#The module changes the fields on the site, fills in the history of the model's move and keeps track of moves

#import sys

from play import getmove
from typing import Optional, Tuple, List, Dict, Union
import traceback

def getchessboardform(chessboard_form):
    '''change the fields on the site fills in the history of the model's move and keeps track of moves
    IN: site form with user progress
    OUT: the same site form with the completed model move and move history'''
    return chessboard_form



def get_model_move(game: str, legal_moves: List[str]) -> Optional[List[str]]:
    """Get the history of moves and possible moves in the given situation
    Returns after checking the legal move that the model suggested"""

    moves = get_model_legal_moves(game,legal_moves)
    model_legal_moves: List = moves if (moves)!=[] else get_model_legal_moves(game,legal_moves, num_moves=5)
    return model_legal_moves
    
def get_model_legal_moves(game: str,legal_moves: List,num_moves=1)->  Optional[List[str]]:
    """Generates a list of moves, checks them for legality
    Returns a list of all legal moves proposed by the model"""
    model_answer: List  = getmove(game.replace('. ','.'),num_moves)
    if model_answer is None:
        return None
    if None in model_answer:
        return []
    # if model_answer is []:
    #     return []
    print('model_answer',model_answer)
    model_legal_list: List =[]
    for model_move in model_answer:
        if  model_move.replace('#','').replace('+','').replace('!','').replace('?','') in legal_moves: # !, ?
            model_legal_list += [model_move]
            print('legal  ',model_move)
        else:
            print(traceback.format_exc())

    return model_legal_list
    

