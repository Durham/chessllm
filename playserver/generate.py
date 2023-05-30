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

#model_path = './b1-1'
#model_path = './15'
model_path= 'BlueSunflower/gpt2-medium-chess'
tokenizer = GPT2Tokenizer.from_pretrained(model_path, pad_token='<|pad|>')

print("Preprairing model")
configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration)
print(len(model.base_model.h)) 
print(model.base_model.h[1])

model.resize_token_embeddings(len(tokenizer))
#model.to('cuda:0')

model.eval()

def generate(context, topk=10,topp=0.9,beam_size=2,length=40,numsamples=3):
#    gc.collect()
    encoded_input = tokenizer.encode(context,return_tensors='pt')
    max_length=len(encoded_input[0])+length
    sample_outputs = model.generate(
                                    encoded_input,#.to('cuda:1'),
                                    do_sample=True,
                                    top_k=topk,
                                    max_length = max_length,
                                    top_p=topp,
                                    num_beams=2,
                                    no_repeat_ngram_size=4,
                                    num_return_sequences=numsamples
                                )
    results = []
    for i, sample_output in enumerate(sample_outputs):
       results.append(tokenizer.decode(sample_output[len(encoded_input[0]):], skip_special_tokens=True))

    return results