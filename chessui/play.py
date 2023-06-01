from lib2to3.pgen2 import token
import re
from unicodedata import name
#import torch
#from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler,IterableDataset
#from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
#from transformers import AdamW, get_linear_schedule_with_warmup
from neuthink.functional import load_lines
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
            boardanalize(example_board, file_result)#здесь метод запускающий анализ.
            end_time = time.time()
            print("Session N: " + str(count)+' '+ str(end_time-start_time/60)+ 'time minute', end = '\r')
        except:
            print("Session N: " + str(count)+' error')
            print(traceback.format_exc())
    
 
def filtermoves(board): # selects a list of moves without numbering and other signs
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
        correct_mate = 1 if next_move==mat else 0 #check if the move was checkmate,
        unrecognized_mate = isunrecognizedmate(next_move,mat) #check if the model recognizes its move as checkmate
        legal_moves = islegalmove(next_move,all_moves) #check if the move was allowed


    
    print("is checkmate: " + mat + " Model move: " + str(next_move))
    print("is checkmate:: " + str(correct_mate) + " Recognized checkmate: " + str(unrecognized_mate) + " Legal move: " + str(legal_moves))
    #запись в файл (это мат?, это реальный ход?)
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

    #model_side_white: bool = getmodelside(context) #определяем за белую ли сторону играет модель
    #context_rate: str = getratepath(model_side_white,model_rate,enemy_rate)  #доп контекст с рейтингом
    context_rate = "0-1 1350 2700 " #Доп контекст для рейтинга
    context_rate_and_board: str =context_rate + context
    print('контекст срейтингом',context_rate_and_board)
    
    #Генерация напрямую от модели
    #model_boards_list: List = generate_text(context_rate_and_board, token_len,num_ret_seq)
    try:
        #Апи на генерацию
        model_boards_list: List = get_api_generate(context_rate_and_board,token_len,num_ret_seq) #Запрос к модели через api
        if model_boards_list is None or model_boards_list == []:
            model_boards_list: List = get_api_generate(context_rate_and_board,token_len,num_ret_seq) #Запрос к модели через api
            if model_boards_list is None:
                return None
    except:
        print(traceback.format_exc())
        logging.error('ошибка запроса к модели через api:'+ traceback.format_exc())
        
    print('модель вернула',model_boards_list)
    #print(model_boards_list)
    model_moves_list: List = []
    #print(model_moves)
    if len(model_boards_list) == 1:
        #вернуть 1 значение
        model_moves: str = model_boards_list[0] #[берем первый элемент] #генерация ходов от модели исходя из контекста
        model_move: List = model_moves_list + [getfiltermove(context_rate_and_board,model_moves)]
        return model_move
    else:
        for board_model in model_boards_list:
            model_move: str = getfiltermove(context_rate_and_board,board_model)
            model_moves_list: List = model_moves_list + [model_move]
        return model_moves_list
    
def getfiltermove(context: str,model_moves: str) -> str:
    """Принимает ходы ОДНОЙ партии и определяет из них ход модели
    IN: context = партия до хода модели, model_moves = партия с ходом модели
    Возвращает 1 ход модели.
    OUT: model_move = 'e5' """
    move_list: List = filtermoves(context + model_moves)  #выборка только ходов без нумерации и прочих знаков
    print(move_list)

    move_count: int      = len(filtermoves(context)) #определение количества ходов в контексте
    move_count_model: int  = len(filtermoves(context + model_moves)) 
    print("Ходов в доске: " + str(move_count) + " Ходов модели: " + str(move_count_model))

    next_move:str = ''
    if (move_count_model > move_count):
        next_move = move_list[move_count]
        print(next_move)
    else:
        next_move = None
    return next_move


def getratepath(model_side_white: bool, model_rate: str,enemy_rate: str) -> str:
    """Принимает сторона модели в партии, рейтинг модели, рейтинг соперника.
    IN: model_side = 1, model_rate = 2300, enemy_rate=2200
    Возвращает строку с предсказанием победителя.
    OUT: rete_path = '1-0 2200 2300 ' """

    result: str  = '1-0' if model_side_white else '0-1' # результат партии исходя из стороны модели(модель должна выйграть)

    if (result == '1-0'): # проставление рейтинга исходя из результата партии
        rate_white: str  = model_rate
        rate_black: str  = enemy_rate
    else:
        rate_white: str  = enemy_rate
        rate_black: str  = model_rate

    rate_context: str = result + " " + rate_white + " " + rate_black + " " # составление части с рейтингом для дополнения контекста
    return rate_context

def getmodelside(board: str) -> bool:
    """Исходя из текущий доски определяет сторону за которуюю играет модель
    IN: board = '1. d4 d5 2. Nf3 Nf6 3.', 
    Возвращает указатель, за белых ли играет модель.
    OUT: model_side_white = true"""
    movies: List = (board.replace('.',' ')).split() # доска с ходами и номерами ходов 
    model_side_white: bool = True if movies[-1].isdigit() else False

    return model_side_white

#model_path = 'python_saved'
#model_path = 'model1_2'
#model_path = 'rated_copy' 

model_path = 'rating_last' #python_saved

token_len = 6

# configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False)
# # instantiate the model
# model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration)
# tokenizer = GPT2Tokenizer.from_pretrained(model_path, bos_token='<|startoftext|>',eos_token='<|endoftext|>')
# tokenizer.pad_token=tokenizer.eos_token

# model_rate = input('model rate') #глобальная переменная рейтинга для модели
# enemy_rate = input('enemy rate') #Глобальная переменная рейтинга соперника модели

if __name__ == '__main__':
    #Старт
    model_path = input('Input model_path')
    file_result = input('Input file_result')

    need_gpu = input('need_gpu?')
    num_gpu = input('num_gpu?')

    model_rate = input('model rate') #глобальная переменная рейтинга для модели
    enemy_rate = input('enemy rate') #Глобальная переменная рейтинга соперника модели


    print("Preprairing model") #вызвать 1 раз
    # configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False)
    # instantiate the model
    # model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration)
    # print(len(model.base_model.h)) 
    # print(model.base_model.h[1])
    # # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # # otherwise the tokenizer and model tensors won't match up
    # model.resize_token_embeddings(len(tokenizer))

    # if need_gpu != '' and (num_gpu == '0' or num_gpu == '1'):
    #     model.to(f'cuda:{num_gpu}')

    # tokenizer = GPT2Tokenizer.from_pretrained(model_path, bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>')

    all_mat_boards =  getboards() #получаем доску

    verifyallboards(all_mat_boards, file_result) 




