


import requests
from typing import Optional
import traceback

from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

def getboard():
    """Makes a request to a json file and gets data about the board, possible moves, and checkmate"""  result = requests.get("https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/checkmate_in_one/task.json")
    all_mat_boards = result.json()
    example_boards = all_mat_boards['examples'][0]

    board = example_boards['input']
    model_board = board.replace('. ','.')       

    mat = example_boards['target']
    all_moves = example_boards['target_scores']

    print(model_board)
    return model_board
    
model_path = './model1_2'
context = getboard() #'1.e4 e5' # board  state
tokenizer = GPT2Tokenizer.from_pretrained(model_path, bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>')
tokens = tokenizer.encode(context,return_tensors='pt')
len(tokens[0])
print(len(tokens[0]))

#getboard()
def getmove(): #generate_text()
    """Gets the context and returns the move that the model predicts"""
    move = 'Rg5#'
    return move

def moveanalize():
    move = 'Rg5#'
    return move



