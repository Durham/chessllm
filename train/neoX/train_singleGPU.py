import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler,IterableDataset
#from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.optim import Adam
from transformers import AdamW, get_linear_schedule_with_warmup
from neuthink.functional import load_lines
import json
import random
import numpy as np
from typing import List
import datetime
import gc
#from apex import amp

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
import torch.nn.functional as F
import torch.backends.cuda as cuda


def transform_chess_string(chess_string):
    splitted_string = chess_string.split(' ')
    transformed_string = []
    last_move_number = None

    for word in splitted_string:
        if '.' in word:
            move_number, move = word.split('.')
            if move_number != last_move_number:
                transformed_string.append(word)
                last_move_number = move_number
            else:
                transformed_string.append(move)
        else:
            transformed_string.append(word)
    
    return ' '.join(transformed_string)

def _attn_wrapper(self, query, key, value, attention_mask=None, head_mask=None):
#    assert attention_mask is None and head_mask is None, "Not implemented"
    with cuda.sdp_kernel(enable_math=False):
        out = F.scaled_dot_product_attention(
            query.half(),
            key.half(),
            value.half(),
            is_causal=True,
        ).float()
    return out, None

# patch attention to save a lot of memory
#GPTNeoXAttention._attn = _attn_wrapper
old = GPTNeoXAttention._attn
GPTNeoXAttention._attn = _attn_wrapper
#model_path = "EleutherAI/pythia-410m-deduped"
model_path = "./410M"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


train_dataset_name = '/home/user/downloader/output.txt'
test_dataset_name = '/home/user/downloader/test_set.txt'

numrecords = 1000000

def load_data(filename:str)->List[str]:
    data = load_lines(filename)
    common_data = [transform_chess_string(x.strip()) for x in data] # place from chess anotations file
    return common_data


def load_data_txt(filename:str)->List[str]:
    f = open(filename)
    common_txt = []
    i = 0
    while True:
        line = f.read(1400)
        i = i + 1
        if line =="" or i >350:
            break
        common_txt.append(line)
  
    return common_txt



class GPT2Dataset_validation(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []
#    self.filevar = open(filename)
    self.numepoch=0
    i = 0
    for txt in txt_list:
#      i = i + 1
#      print(i)
#      text = json.loads(txt)['text']
      encodings_dict = tokenizer(txt , truncation=True, max_length=max_length, padding="max_length")
#      print(encodings_dict['input_ids'])
#      print(type(encodings_dict['input_ids']))

      labels = encodings_dict['input_ids']
#      labels[labels==tokenizer.pad_token_id]=-100
      self.input_ids.append(torch.tensor(labels))

      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
   
    return (self.input_ids[idx], self.attn_masks[idx])

tokens = 0
class fileloader():
      def __init__(self, filename, tokenizer, max_length):
        self.filevar = open(filename)
#        self.filevar.seek(1400*2*120000)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.filename = filename
        self.tokens = 0

      def __next__(self):
            global tokens
            data = self.filevar.readline()
            if data =="":
               self.filevar = open(self.filename)
               data = self.filevar.readline()

            data = transform_chess_string(data.strip())

            encodings_dict = self.tokenizer( data,  truncation=True, add_special_tokens=True, max_length=self.max_length, padding="max_length")
#            print(encodings_dict)
            labels = encodings_dict['input_ids']
#            labels[labels==tokenizer.pad_token_id]=-100

            input_ids = torch.tensor(labels) 
            tokens = tokens + input_ids[0]
            attn_masks = torch.tensor(encodings_dict['attention_mask'])
            return input_ids, attn_masks 
            

class GPT2Dataset(IterableDataset):

  def __init__(self, filename, tokenizer, gpt2_type=model_path, max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []
    self.max_length =  max_length
    self.filename = filename
    self.numepoch=0


  def __iter__(self):
    return fileloader(self.filename, self.tokenizer, self.max_length)
  
    


batch_size = 8
print("building train")
train_dataset = GPT2Dataset(train_dataset_name, tokenizer, max_length=768)
print("building validation")
data = load_data(test_dataset_name)
print(data[0:3])
validation_dataset = GPT2Dataset_validation(data, tokenizer, max_length=768)



# Split into training and validation sets

train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            batch_size = batch_size # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            validation_dataset, # The validation samples.
            sampler = SequentialSampler(validation_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

print("Preprairing model")
#configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False)

# instantiate the model
#model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration, ignore_mismatched_sizes=True)
model = GPTNeoXForCausalLM.from_pretrained(model_path)

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

epochs = 5
learning_rate = 1e-5
warmup_steps = 100
epsilon = 1e-8

# this produces sample output every 100 steps
sample_every = 300

optimizer = Adam(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon, betas=(0.9,0.999)
                )


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
import time

total_t0 = time.time()

training_stats = []

model = model.to(device)
output_dir = "./410Ma"

for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    j= 0

    start_time = time.time()
    batch_loss=0
    for step, batch in enumerate(train_dataloader):

        j = j + 1                                                                                                                       
        print(j,' ',end='\r')                                                                                                               
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_labels[b_labels==tokenizer.pad_token_id]=-100
        b_masks = batch[1].to(device)

        model.zero_grad()        
        
        with torch.autocast(device_type='cuda', enabled=True):
         outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          
                        )
         loss = outputs['loss']
        scaled_loss = scaler.scale(loss)
#        print(scaled_loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()
        batch_loss =loss.item()
        total_train_loss += batch_loss
        

#        scheduler.step()
        
        
        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:
            
            print("Iter time", time.time()-start_time)
            print("Tokens", tokens)
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, 1000000, batch_loss, elapsed))

            model.eval()
            GPTNeoXAttention._attn = old
            new_user_input_ids = tokenizer.encode("Score:", return_tensors='pt').to('cuda')

            sample_outputs = model.generate(new_user_input_ids,
                                    
                                    do_sample=True,   
                                    top_k=2, 
                                    max_length = 260,
#                                    top_p=1.0, 
                                    
                                    num_return_sequences=1
                                )
            for i, sample_output in enumerate(sample_outputs):
                  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            model.train()
            GPTNeoXAttention._attn = _attn_wrapper


            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        if step % 1000 == 0 and not step == 0:
             gc.collect()
             torch.cuda.empty_cache()

             # Calculate the average loss over all of the batches.
             avg_train_loss = total_train_loss /1000#/ len(train_dataloader)       
             total_train_loss=0
             # Measure how long this epoch took.
             training_time = format_time(time.time() - t0)

             print("")
             print("  Average training loss: {0:.2f}".format(avg_train_loss))
             print("  Training 2000 iters took: {:}".format(training_time))
        

             print("")
             print("Running Validation...")

             t0 = time.time()

             model.eval()

             total_eval_loss = 0
             nb_eval_steps = 0
             z = 0
             # Evaluate data for one epoch
             for batch in validation_dataloader:
        
                 b_input_ids = batch[0].to(device)
                 b_labels = batch[0].to(device)
                 b_masks = batch[1].to(device)
                 b_labels[b_labels==tokenizer.pad_token_id]=-100
        
                 with torch.no_grad():        

                   outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
                   loss = outputs['loss']
            
                   batch_loss = loss.item() 
                   total_eval_loss += batch_loss        
                   z  = z +1

             avg_val_loss = total_eval_loss / z
    
             validation_time = format_time(time.time() - t0)    
#             print(z,total_eval_loss, agv_val_loss)
             print("  Validation Loss: {0:.2f}".format(avg_val_loss))
             print("  Validation took: {:}".format(validation_time))

             f = open('stats.csv','a')
             f.write(str(step)+'('+str(int(step/3))+')' +'\t'+ str(avg_train_loss) + '\t' + str(avg_val_loss)+'\n')
             f.close()
    

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
