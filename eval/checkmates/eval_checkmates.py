#This scripts evaluates model abitlity to make checkmates in one move

import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler,IterableDataset
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from neuthink.functional import load_lines
import json
import random
import numpy as np
from typing import List, Dict
import datetime
import requests
import time
import traceback


# model.eval()
# context = "1.e4 e5 2.d4 exd4 3.Qxd4 Nc6 4.Qxg7"
# #ODcontext = "
# import time
# start = time.time()
# model.eval()
model_path="BlueSunflower/gpt2-medium-chess"

def getboards()-> List[Dict[str,str]]:
    """Makes a request to a json file and gets data about the board, possible moves, and checkmate"""

    result = requests.get("https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/checkmate_in_one/task.json")
    all_mat_boards = result.json()
    example_board = all_mat_boards['examples'][1]
    print("number of tasks:", len(all_mat_boards['examples']))
    return all_mat_boards


def verifyallboards(all_mat_boards, file_result:str = 'res_model_static.txt')-> None:
    '''The function receives and writes the result to the file the result of the model run '''
    start_time = time.time()
    for count,example_board in enumerate(all_mat_boards['examples']):
        try:
            boardanalize(example_board, file_result)
            end_time = time.time()
            print("Line N " + str(count)+' '+ str(end_time-start_time/60)+ 'time minute', end = '\r')
        except:
            print("Line N " + str(count)+' error')
            print(traceback.format_exc())

    


def generate_text(context, length): #generates a move based on the given context

    tokens = tokenizer.encode(context,return_tensors='pt').to('cuda:1')
    print(context)
    new_length = len(tokens[0])+length
    sample_outputs = model.generate(
                                    tokens,
                                    do_sample=False,   
#                                    top_k=2, 
                                    max_length = new_length,
#                                    top_p=0.99, 
#                                    temperature=0.7
#                                    num_beams=4,
#                                    no_repeat_ngram_size=2,
                                    num_return_sequences=1
                                )
  
 #for i, sample_output in enumerate(sample_outputs):
 #     print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
 
def filtermoves(board):
    filter_movies = (board.replace('.',' ')).split() 
    filter_movies = [x for x in filter_movies if not x.isdigit()]
    print(filter_movies)
    return filter_movies


def moveanalize(context_rate,board,next_move,mat,all_moves, file_result:str = 'res_model_static.txt'):
    """Compares the progress of the model with the expected ones and writes the result to a file"""
    if (next_move is None):
        correct_mate = 0
        unrecognized_mate = 0
        legal_moves = 0
    else:
        correct_mate = 1 if next_move==mat else 0
        unrecognized_mate = isunrecognizedmate(next_move,mat)
        legal_moves = islegalmove(next_move,all_moves)
    
    total["correct"] = correct_mate + total.get("correct",0)
    total["legal"] = legal_moves + total.get("legal",0)
    total["unrecognized"] = unrecognized_mate + total.get("unrecognized",0)


    print("is checkmate?: " + mat + " Model move: " + str(next_move))
    print("is checkmate: " + str(correct_mate) + " Recognized checkmate: " + str(unrecognized_mate) + " Legal move: " + str(legal_moves))

    file = open(file_result, "a")
    file.write(context_rate +";"+ board + ";" + mat + ";" + str(next_move) + ";" + str(correct_mate) + ";"+ str(unrecognized_mate) + ";" + str(legal_moves) + "\n")
    file.close()

def isunrecognizedmate(next_move,mat):
    if (mat[0:-1] in next_move) and (mat[-1] != next_move[-1]):
        return 1
    else:
        return 0 

def islegalmove(next_move,all_moves):
    next_move = next_move.replace("+","").replace("#","")
    all_moves = [i.replace("+","").replace("#","") for i in all_moves]    
    if (next_move in all_moves):
        return 1
    else:
        return 0 

def boardanalize(example_board,file_result:str= 'res_model_static.txt'):
    board = example_board['input'].replace('. ','.')
    mat = example_board['target']
    all_moves = example_board['target_scores']

    model_side_white: bool = getmodelside(board)
    context_rate: str = getratepath(model_side_white,model_rate,enemy_rate)
    context_rate_and_board: str =context_rate + board

    #print(board)
    model_moves = generate_text(context_rate_and_board, 6)
    #print(model_moves)

    move_list = filtermoves(model_moves)

    move_count       = len(filtermoves(context_rate_and_board))
    move_count_model = len(filtermoves(model_moves)) 

    if (move_count_model > move_count):
        next_move = move_list[move_count]
    else:
        next_move = None

    # try:
    #     next_move = move_list[move_count]
    # except:
    #     next_move = "None"

    #print(move_count)
    #print(next_move)

    moveanalize(context_rate,board,next_move,mat,all_moves, file_result)

def getratepath(model_side_white: bool, model_rate: str,enemy_rate: str) -> str:
    """Accepts the side of the model in the party, the rating of the model, the rating of the opponent.
    IN: model_side = 1, model_rate = 2300, enemy_rate=2200
    Returns a string with the prediction of the winner.
    OUT: rete_path = '1-0 2200 2300' """
    result: str  = '1-0' if model_side_white else '0-1'
#    result: str  = '0-1' if model_side_white else '1-0'

    if (result == '1-0'):
        rate_white: str  = model_rate
        rate_black: str  = enemy_rate
    else:
        rate_white: str  = enemy_rate
        rate_black: str  = model_rate

    rate_context: str = result + " " + rate_white + " " + rate_black + " "
    return rate_context

def getmodelside(board: str) -> bool:
    """Based on the current board, determines the side for which the model plays
    IN: board = '1. d4 d5 2. Nf3 Nf6 3.',
    Returns a pointer to whether the model is playing White.
    OUT: model_side_white = true"""
    movies: str = (board.replace('.',' ')).split() # доска с ходами и номерами ходов 
    model_side_white: bool = True if movies[-1].isdigit() else False

    return model_side_white


#Старт
model_path = input('Input model_path')
file_result = input('Input file_result')

model_rate = input('model rate')
enemy_rate = input('enemy rate')


print("Preprairing model")
configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False)
# instantiate the model
model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration)

model.to('cuda:0')

tokenizer = GPT2Tokenizer.from_pretrained(model_path, bos_token='<|startoftext|>',eos_token='<|endoftext|>')
tokenizer.pad_token=tokenizer.eos_token

all_mat_boards =  getboards()

verifyallboards(all_mat_boards, file_result) 
print(total)



