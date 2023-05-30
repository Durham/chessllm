from lib2to3.pgen2 import token
import re
from unicodedata import name
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

model_path = 'engine2x_800K' #python_saved

# model.eval()
# context = "1.e4 e5 2.d4 exd4 3.Qxd4 Nc6 4.Qxg7"
# #ODcontext = "
# import time
# start = time.time()
# model.eval()

def getboards()-> List[Dict[str,str]]:
    """Makes a request to a json file and gets data about the board, possible moves, and checkmate"""
    result = requests.get("https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/checkmate_in_one/task.json")
    all_mat_boards = result.json()
    example_board = all_mat_boards['examples'][1]
    print("number of tasks:", len(all_mat_boards['examples']))
    return all_mat_boards


def verifyallboards(all_mat_boards, file_result:str = 'res_model_static.txt')-> None:
    '''The function receives and writes the result to the file the result of the model run '''


    start_time = time.time() #timer to track the time of game games
    for count,example_board in enumerate(all_mat_boards['examples']):
        try:
            boardanalize(example_board, file_result)
            end_time = time.time()
            print("Line N: " + str(count)+' '+ str(end_time-start_time/60)+ 'time minute', end = '\r')
        except:
            print("Line N: " + str(count)+' Error')
            print(traceback.format_exc())

    


def generate_text(context, length, num_ret_seq): #generates a move based on the given context
    tokens = tokenizer.encode(context,return_tensors='pt').to('cuda:1')
    print(context,num_ret_seq,length)
    new_length = len(tokens[0])+length
    do_sample=False if num_ret_seq==1 else True
    if num_ret_seq==1:
     sample_outputs = model.generate(
                                    tokens,
                                    do_sample=False,   
#                                    top_k=2, 
                                    max_length = len(tokens[0])+length,                                    
                                    top_p=0.8, 
                                    temperature=1.0,
#                                    num_beams=num_ret_seq,
#                                    output = 15,
#                                    no_repeat_ngram_size=2,
                                    num_return_sequences=num_ret_seq
                                )
    else:
     sample_outputs = model.generate(
                                    tokens,
                                    do_sample=False,   
#                                    top_k=2, 
                                    max_length = len(tokens[0])+length,                                    
                                    top_p=0.8, 
                                    temperature=1.1,
                                    num_beams=num_ret_seq,
#                                    output = 15,
#                                    no_repeat_ngram_size=2,
                                    num_return_sequences=num_ret_seq
                                )
  
 #for i, sample_output in enumerate(sample_outputs):
 #     print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
 #   return tokenizer.decode(sample_outputs[0], skip_special_tokens=True) возвращаем 1 элемент
    return [tokenizer.decode(text, skip_special_tokens=True) for text in sample_outputs] #возвращаем список

    
 
def filtermoves(board):
    filter_movies = (board.replace('.',' ')).split() 
    filter_movies = [x for x in filter_movies if not x.isdigit()]

    return filter_movies



total={}

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


     file = open(file_result, "a")
    file.write(context_rate + ";" + mat + ";" + str(next_move) + ";" + str(correct_mate) + ";"+ str(unrecognized_mate) + ";" + str(legal_moves) + "\n")
    file.close()

def isunrecognizedmate(next_move,mat):
    if (mat[0:-1] in next_move) and (mat[-1] != next_move[-1]):
        return 1
    else:
        return 0 

def islegalmove(next_move,all_moves):
    next_move = next_move.replace("+","").replace("#","")
    all_moves = [i.replace("+","").replace("#","") for i in all_moves]    
#    all_moves = [i for i in all_moves]    


    if (next_move in all_moves):
        return 1
    else:
        return 0 
#generate_text('1.e4 c5 2.Nf3 Nc6 3.Nc3 e5 4.Bb5 Nge7 5.0-0 g6 6.Nd5 a6',80)

def boardanalize(example_board,file_result:str= 'res_model_static.txt'):
    board = example_board['input'].replace('. ','.')
    mat = example_board['target']
    all_moves = example_board['target_scores']

    #print(board)
    next_move: str =  getmove(board)[0]

    #print(move_count)
    #print(next_move)

    # анализ и запись в файл
    moveanalize(board,next_move,mat,all_moves, file_result)

def getmove(context: str, num_ret_seq:int = 1, num_move=1) -> List:
    """Gets the context and number of moves returned,
    based on this, returns 1 or more moves of the model"""

    model_side_white: bool = getmodelside(context)
#    context_rate: str = getratepath(model_side_white,model_rate,enemy_rate)  #доп контекст с рейтингом
    context_rate = "0-1 1350 2710 "
#    context_rate="0-1 "
    context_rate_and_board: str =context_rate + context
    print('context',context_rate_and_board)
    print(context_rate_and_board)
    #t = token_len if num_ret_seq!=6 else 15
    model_boards_list: List = generate_text(context_rate_and_board, token_len,num_ret_seq)
    #print(model_boards_list)
    model_moves_list: List = []
    #print(model_moves)
    if len(model_boards_list) == 1:
        #вернуть 1 значение
        model_moves: str = model_boards_list[0]
        model_move: List = model_moves_list + [getfiltermove(context_rate_and_board,model_moves)]
        return model_move
    else:
        for board_model in model_boards_list:
            model_move: str = getfiltermove(context_rate_and_board,board_model)
            model_moves_list: List = model_moves_list + [model_move]
        return model_moves_list


def getfiltermove(context: str,model_moves: str) -> str:
    """Accepts the moves of ONE game and determines the course of the model from them
    IN: context = game before model move, model_moves = game with model move
    Returns 1 model move.
    OUT: model_move = 'e5' """
    move_list: List = filtermoves(model_moves)
    #print(move_list)

    move_count: int      = len(filtermoves(context))
    move_count_model: int  = len(filtermoves(model_moves)) 

    next_move:str = ''
    if (move_count_model > move_count):
        next_move = move_list[move_count]
    else:
        next_move = None
        print("ERROR **********************************************************************")
        print(context, model_moves)
        print("*****************************************************************************")
    return next_move


def getratepath(model_side_white: bool, model_rate: str,enemy_rate: str) -> str:
    """Accepts the side of the model in the party, the rating of the model, the rating of the opponent.
    IN: model_side = 1, model_rate = 2300, enemy_rate=2200
    Returns a string with the prediction of the winner.
    OUT: rete_path = '1-0' """

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
    movies: List = (board.replace('.',' ')).split()
    model_side_white: bool = True if movies[-1].isdigit() else False

    return model_side_white

#model_path = 'python_saved'
#model_path = 'model1_2'
#model_path = 'rated_copy' 


token_len = 7

configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False)
# instantiate the model
model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration)
tokenizer = GPT2Tokenizer.from_pretrained(model_path, bos_token='<|startoftext|>',eos_token='<|endoftext|>')
tokenizer.pad_token=tokenizer.eos_token
model.to('cuda:0')

if __name__ == '__main__':
    model_rate = input('model rate')
    enemy_rate = input('enemy rate')

    #Старт
    model_path = input('Input model_path')
    file_result = input('Input file_result')

    print("Preprairing model") #вызвать 1 раз
    configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False)
# instantiate the model
    model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration)
# print(len(model.base_model.h)) 
# print(model.base_model.h[1])
# # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# # otherwise the tokenizer and model tensors won't match up
# model.resize_token_embeddings(len(tokenizer))
    model.to('cuda:0')
#if need_gpu != '' and (num_gpu == '0' or num_gpu == '1'):
#    model.to(f'cuda:{num_gpu}')




