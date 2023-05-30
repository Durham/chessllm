#This script can be used to play from console

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler,IterableDataset
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from neuthink.functional import load_lines
import json
import random
import numpy as np
from typing import List
import datetime

model_path= 'BlueSunflower/gpt2-medium-chess'
tokenizer = GPT2Tokenizer.from_pretrained(model_path, bos_token='<|startoftext|>', eos_token='<|endoftext|>')
tokenizer.pad_token=tokenizer.eos_token

print("Preprairing model")
configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration)


model.eval()


import time
start = time.time()
model.eval()

def generate_text(context, length, nb=1):
 tokens = tokenizer.encode(context,return_tensors='pt')
 sample_outputs = model.generate(
                                    tokens,
                                    do_sample=False,   
#                                    top_k=2, 
                                    max_length = len(tokens[0])+length,                                    
                                    top_p=0. ,
                                    temperature=1.0,
                                    num_beams=nb,
#                                    no_repeat_ngram_size=2,
                                    num_return_sequences=nb
                                )
  
 for i, sample_output in enumerate(sample_outputs):
      print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
 return [tokenizer.decode(x, skip_special_tokens=True) for x in sample_outputs]

def model_prob(text):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    sentence_prob = loss.item()
    return sentence_prob


def get_model_move(md,i):
      model_move = (md.split(str(i+1)+".")[1].split(' ')[1])
      
      return model_move


def extend(context, parties,i):
    total=[]
    print(len(parties))
    j=0
    for party in parties:
       move = get_model_move(party,i)

       j=j+1     
       enemy = "1-0 2700 1350 "+ context[14:].strip() + ' ' + move
       print(enemy,"****",move, j)
       enemyplays = generate_text(enemy,6,1)[0]
       print(enemyplays)
       score = model_prob(enemyplays)
       qscore = model_prob("0-1 2350 1350 "+enemyplays[14:])
       total.append((move, score,qscore,qscore-score))
    return total

      
    
   
 
def play():           
#     context='0-1 '
    for i in range(0,108):
      move = input("Enter your move:")
      context = context +str(i+1)+'.' + move
      md = generate_text(context, 6,1)[0]
      model_move = (md.split(str(i+1)+".")[1].split(' ')[1])
      print("Model move:", model_move)
      print("------")
      print(md)
      context = context + " " + model_move + " "
      print(context)

play()


