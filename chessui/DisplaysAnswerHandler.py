from posixpath import split
from typing import List
import tornado.web
import json
import scratch.forms as forms
import time
import traceback
import logging
import uuid

from chessboard_change import get_model_move

import chess
import chess.pgn
import io
users = {}
#import datetime
from datetime import datetime, timedelta

#board = chess.Board() #'board1'
#board_start = chess.Board()
logging.basicConfig(filename = "weblogging.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

file_html = "reviewlink.html"

def create_uuid()->str:
    """ Generates a string to identify the user who visited the page """
    return str(uuid.uuid4())

def save_games():
    try:
        file = open('all_users_games.txt', "w")
        file.write(str(users)) 
        file.close()
    except:
        print(traceback.format_exc())
        logging.error('Error:'+ traceback.format_exc())

def clear_users():
    """Clears the list of users whose games started more than 5 hours ago"""
    now = datetime.now()
    users_for_pop :List =[]

    for id_user in users:
        #print(users[id_user][2])
        if ((users[id_user][2] + timedelta(minutes=300))  <= now): 
            users_for_pop += [id_user]

    for id_user in users_for_pop:
        popped_value = users.pop(id_user)

    


class DisplayFormHandler(tornado.web.RequestHandler):
    '''This is a handler with a form for user input'''


    def get(self):
        
        #if not self.get_secure_cookie("user"):
        uuid_user = create_uuid()
        print('uuid_user: ', uuid_user)
        self.set_secure_cookie("user", uuid_user)

        print('user cookie: ', self.get_secure_cookie("user"))
        #self.clear_cookie("user")

        #chessboard_form =  forms.NodeForm({"history_move":"label","count_move":"hidden","human_move":"editbox","model_move":"editbox"})
        clear_users()


        self.render("reviewlink.html") #, chessboard_form = chessboard_form


    def post(self):
        try:
            #board = chess.Board()
            id_user = str(self.get_secure_cookie("user"))
            print('user cookie after deletion: ',id_user)

            if id_user in users:
                print('User already exists')
            else:
                start_time = datetime.now()
                print('start_time', start_time)
                users[id_user] = [chess.Board(),chess.Board(),start_time]
                print('new user, new board',users[id_user])
            
            print(users)
            board = users[id_user][0]
            board_start = users[id_user][1]
            print(board)

            target = self.get_argument("target")
            source = self.get_argument("source")
            print(source,target)


            human_move = get_move_model_note(source,target,board )
            history_move, board, board_start = push_and_write(human_move, board, board_start)
            game_status = get_board_status(check_board(history_move))
            if (game_status !=""):
                result = {"history_move1":history_move , "gamestatus":game_status}
                print('json ',result)
                self.write(json.dumps(result, ensure_ascii=False).encode('utf8'))
            else:
                legal_moves = list(board.legal_moves)
                legal_moves = [board.san((legal_move)).replace('#','').replace('+','').replace('!','').replace('?','') for legal_move in legal_moves]
                print('Legal moves',legal_moves)

                model_moves:List = get_model_move(history_move.replace('. ','.'),legal_moves)
                
                if (model_moves is None) or (model_moves == []) :      
                    model_lose = 'Model can not do legal move.<br>White wins' if model_moves is not None else 'Server error. Try again later'
                    model_lose = game_status if game_status !="" else model_lose
                    result = {"history_move1":history_move , "gamestatus":model_lose}
                    print('json ',result)
                    self.write(json.dumps(result, ensure_ascii=False).encode('utf8'))
                else:
                    model_move:str = model_moves[0]
                    game_status = 'Game in progress'
                
                note_board_move = [i for i in  str(board.parse_san(model_move))]
                note_board_move.insert(2, "-")
                print('move: ', "".join(note_board_move))

                history_move, board, board_start = push_and_write(model_move, board, board_start)
                game_status = get_board_status(check_board(history_move))

                users[id_user] = [board,board_start,users[id_user][2]]

                result = {"model_move": "".join(note_board_move)[:5], "history_move1":history_move , "movewithq":"".join(note_board_move), "gamestatus":game_status}
                print('json ',result)

                self.write(json.dumps(result, ensure_ascii=False).encode('utf8'))

        except:
            print(traceback.format_exc())
            logging.error('error post:'+ traceback.format_exc())
            logging.info('moves history: '+ history_move)
    
def get_move_model_note(source: str,target: str, board)-> str:
    """A function that translates the move from 1 notation to model notation"""
    try:     
        note_board_move = source + target
        #board.push(note_board_move)
        print('input move', note_board_move)
        print(board.piece_at(chess.parse_square(note_board_move[:2])))
        #if (note_board_move == 'a7a8' or note_board_move == 'b7b8'):
        print("".join(c for c in note_board_move if  c.isdecimal()))

        if (str("".join(c for c in note_board_move if  c.isdecimal())) == '78' and  str(board.piece_at(chess.parse_square(note_board_move[:2]))) == 'P'):
            note_model_move = board.san(chess.Move.from_uci(note_board_move)) + '=Q'
        else:    
            note_model_move = board.san(chess.Move.from_uci(note_board_move)) # Move.from_uci('g1h3')
        #board.push_san(note_model_move)
        testnote = board.san(chess.Move.from_uci(note_board_move+'q')) # Move.from_uci('g1h3')
        return note_model_move
    except:
        print(traceback.format_exc())
        logging.error('error:'+ traceback.format_exc())

def push_and_write(move: str, board, board_start) -> str:
    """Sends moves to the board and records them in history"""
    board.push_san(move)
    print('sends moves',board.move_stack)
    writer_game = board_start.variation_san( board.move_stack)
    print('writer_game: ',writer_game)
    return writer_game, board, board_start

def check_board(text):
    print('moves ',text )
    pgn = io.StringIO(text)
    game = chess.pgn.read_game(pgn)
    print(game)
    board = game.board()
    for move in game.mainline_moves():
       board.push(move)
    return board.outcome()

def get_board_status(outcome):
    if outcome is not None:
        wins = outcome.result()
        termination = str(outcome.termination).replace('Termination.','')
        status = wins + termination
        return status
    else:
        return ''