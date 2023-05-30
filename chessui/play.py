from lib2to3.pgen2 import token
import re
from unicodedata import name

import json
import random
import numpy as np
from typing import List, Dict
import datetime
import requests
import time
import traceback
from  model_api_request import get_api_generate
import logging

def load_lines(filename):
  f = open(filename)
  lines = [x.strip() for x in f.readlines()]
  f.close()
  return lines


# model.eval()
# context = "1.e4 e5 2.d4 exd4 3.Qxd4 Nc6 4.Qxg7"
# #ODcontext = "
# import time
# start = time.time()
# model.eval()
logging.basicConfig(filename = "weblogging.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
            print("Session N: " + str(count)+' '+ str(end_time-start_time/60)+ 'time minute', end = '\r')
        except:
            print("Session N: " + str(count)+' error')
            print(traceback.format_exc())

 
def filtermoves(board):# selects a list of moves without numbering and other signs

    filter_movies = (board.replace('.',' ')).split() 
    filter_movies = [x for x in filter_movies if not x.isdigit()]
    return filter_movies


def moveanalize(context_rate,next_move,mat,all_moves, file_result:str = 'res_model_static.txt'):
    """Compares the progress of the model with the expected ones and writes the result to a file"""
    if (next_move is None):
        correct_mate = 0
        unrecognized_mate = 0
        legal_moves = 0
    else:
        correct_mate = 1 if next_move==mat else 0
        unrecognized_mate = isunrecognizedmate(next_move,mat)
        legal_moves = islegalmove(next_move,all_moves)
    
    print("Checkmate: " + mat + " model move: " + str(next_move))
    print("Is checkmate: " + str(correct_mate) + " Recognized checkmate: " + str(unrecognized_mate) + " is legal move?: " + str(legal_moves))

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


    moveanalize(board,next_move,mat,all_moves, file_result)

def getmove(context: str, num_ret_seq:int = 1) -> List:
    """Получает контекст и кол во возвращаемых ходов, 
    исходя из этого возвращает 1 или несколько ходов модели"""

    context_rate = "0-1 1350 2700 "
    context_rate_and_board: str =context_rate + context
    print('контекст срейтингом',context_rate_and_board)
    
    try:
       model_boards_list: List = get_api_generate(context_rate_and_board,token_len,num_ret_seq)
       if model_boards_list is None or model_boards_list == []:
            model_boards_list: List = get_api_generate(context_rate_and_board,token_len,num_ret_seq)
            if model_boards_list is None:
                return None
    except:
        print(traceback.format_exc())
        logging.error('API request error:'+ traceback.format_exc())
        
    print('model returned ',model_boards_list)
    #print(model_boards_list)
    model_moves_list: List = []
    #print(model_moves)
    if len(model_boards_list) == 1:

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
    move_list: List = filtermoves(context + model_moves)
    print(move_list)

    move_count: int      = len(filtermoves(context))
    move_count_model: int  = len(filtermoves(context + model_moves)) 
    print("moves in board: " + str(move_count) + " model moves: " + str(move_count_model))

    next_move:str = ''
    if (move_count_model > move_count):
        next_move = move_list[move_count]
        print(next_move)
    else:
        next_move = None
    return next_move


def getratepath(model_side_white: bool, model_rate: str,enemy_rate: str) -> str:
    """Accepts the side of the model in the party, the rating of the model, the rating of the opponent.
    IN: model_side = 1, model_rate = 2300, enemy_rate=2200
    Returns a string with the prediction of the winner.
    OUT: rete_path = '1-0 2200 2300 ' """

    result: str  = '1-0' if model_side_white else '0-1'

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


model_path = 'rating_last' #python_saved

token_len = 6





