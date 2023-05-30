'''
Chess data generation

'''

import chess
import chess.engine
from typing import Tuple, List, Dict, Union
import time
import json
import traceback
import uuid


from play import getmove

realtime=0
def play_game(engine, board_start, gameid:str = "", limit_time:float = 0.05)-> Tuple[ List[Dict[str,str]] , Dict[str,List[str]], Dict[str,str] ]:
    '''
    Creating and playing a new game

    board_start - the starting position of the pieces on the board
    board - the current position of the pieces on the board
    limit_time - time to think about a move

    data_gamemark = []
    Assign a party id.

    If the board.is_game_over() is not the end of the game on the board and less than 130 moves have passed since the beginning of the game, then:
        The engine makes another move engine.play(on the board, and chess.engine.Limit(time=limit_time))
        Fix the result on the board board.push(move.move)

        We remember the state of the board after the move and write str(board)
        Assign moveid = gameid + i

        Add to data_gamemark: moveid, gameid, board state, current and previous move

     We get the full game in the standard chess annotation and send it to board_start.variation_san( board.move_stack)
          
          
    '''
    global realtime
    def clousedmove_mark(board_conditions:List[str],move_conditions:List[str], \
                                last_move, gameid:str, number_move:int)-> Dict[str, str]:
        '''Composes markup to predict the next move'''
        if number_move > 1:
            data_move = {'gameid': gameid}
            data_move['moveid'] = gameid + '_' + str(number_move)
            data_move['last_board'] = board_conditions[-1]
            data_move['last_move'] = move_conditions[-1]
            data_move['current_move'] = last_move
            return data_move
        else: return {}
        
    number_move = 1
    board_conditions: List[str] = []
    move_conditions: List[str] = []
    data_gamemark = []
    board = chess.Board()
    game_over = 0

#    engine.configure({"Skill Level": 0})
    engine.configure({"UCI_LimitStrength": True, "Skill Level": 0,})
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": 1350})
#    engine.configure({"Threads":20})
    num_move = 0
    print('game_id: ',gameid)
#    realtime = 0
    while not board.is_game_over() and number_move < 130 and game_over==0:
        num_move = num_move+1
        #result = engine.play(board,chess.engine.Limit(time=limit_time, depth = 5))
        q = time.time()
        result = engine.play(board, chess.engine.Limit(time=0.01))
        realtime = realtime+time.time()-q
        print('result: ',result)
        board.push(result.move)

        winblack = 0
        writer_game,last_move,datamove,data_gamemark,move_conditions,board_conditions,number = \
        gamewrite(board,board_start,gameid,clousedmove_mark,number_move,board_conditions,move_conditions,data_gamemark)

        #start model move
        if  not board.is_game_over():
            legal_moves = list(board.legal_moves)
            #print('легальные ходы 1 нотация',legal_moves)
            legal_moves = [board.san((legal_move)) for legal_move in legal_moves]
            print('writer_game',writer_game)
            print('legal moves',legal_moves)


            model_legal_moves = get_model_move(writer_game,legal_moves, num_move)
            print('model_legal_moves ',model_legal_moves)
            if model_legal_moves == []:
                 game_over = 1
            else:
                board.push_san(model_legal_moves[0])

                winblack = 1
                writer_game,last_move,datamove,data_gamemark,move_conditions,board_conditions,number = \
                gamewrite(board,board_start,gameid,clousedmove_mark,number_move,board_conditions,move_conditions,data_gamemark) #Запись результатов ходов в доску, для составления разметки
    
    file = open('win_rate_model_static.txt', "a")
    file.write(gameid + ";" + writer_game + ";" + str(winblack) + "\n")
    file.close()

    return (data_gamemark , {'gameid': gameid, 'board': board_conditions}, {'gameid': gameid, 'moves': writer_game})

#print(realtime)
######################################################################
def get_model_move(game: str, legal_moves: List, num_move) -> List:
    """Get the history of moves and possible moves in the given situation
    Returns after checking the legal move that the model suggested"""

    moves = get_model_legal_moves(game,legal_moves,num_moves=1, num_move=num_move)
    model_legal_moves: List = moves if (moves)!=[] else get_model_legal_moves(game,legal_moves, num_moves=10, num_move=num_move)#лько раз модель может переходить
    return model_legal_moves

def get_model_legal_moves(game: str,legal_moves: List,num_moves=1, num_move=1)-> List:
    """Generates a list of moves, checks them for legality
    Returns a list of all legal moves proposed by the model"""


    model_answer: List  = getmove(game.replace('. ','.'),num_moves, num_move)

    print('model_answer',model_answer)
    model_legal_list: List =[]
    count_move:int = 0
    while count_move < num_moves: 
        if  model_answer[count_move] in legal_moves:
            model_legal_list += [model_answer[count_move]]
            #print('model_answer[0]',model_answer[count_move])
            #print('model_legal_list',model_legal_list)
            print('legal ', model_answer[count_move])
            count_move = countmove(count_move)
        else:
            count_move = countmove(count_move)
            print(traceback.format_exc())

    return model_legal_list

def countmove(count_move: int)-> int:
    print(count_move)
    return count_move +1

def gamewrite(board,board_start,gameid, clousedmove_mark,number_move,board_conditions,move_conditions,data_gamemark): #как проставить типизацию функции clousedmove_mark?
    """Recording the results of moves on the board, for compiling markup for predicting the next move"""
    writer_game = board_start.variation_san( board.move_stack)
    print('writer_game:',writer_game)

    last_move = writer_game.split(' ')[-1]
    print('last_move:',last_move)

    datamove = clousedmove_mark(board_conditions, move_conditions, last_move, gameid, number_move)
    data_gamemark = data_gamemark if datamove == {} else data_gamemark + [datamove]

    move_conditions += [last_move]
    board_conditions += [str(board)]
    number_move += 1
    
    return writer_game,last_move,datamove,data_gamemark,move_conditions,board_conditions,number_move

#####################################################################################

def savemark_injson(data:List[Union[str, Dict[str, str]]], filename:str)->None:
    '''Save data for predicting the move in json format'''
    with open(filename, 'a') as marktotal:
        for move in data:
            marktotal.write( json.dumps(move) + '\n')
 
    
def get_news_chess_games(limit_time:float = 0.005, file_marktotal:str='chessmarktotal.json', file_markdoard:str='markdoard.json', file_anot:str = 'anotationmark.json'):
    '''
    We generate chess games and write them down for markup
    
    '''
    board_start_config = chess.Board()
    #engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    engine = chess.engine.SimpleEngine.popen_uci("/home/user/alice/stockfish-11-linux/Linux/stockfish_20011801_x64")


    
    try:
        i = 1
        while i==1:
            game_id:str = str( uuid.uuid1() )
            info_mark, board_conditions, writer_game = play_game(engine, board_start_config, game_id, limit_time=limit_time)
            savemark_injson(info_mark, file_marktotal)
            savemark_injson([board_conditions], file_markdoard)
            savemark_injson([writer_game], file_anot)
            print(i, end='\r')
            i += 1
    except:
        print(traceback.format_exc())
        engine.quit()
       
            
    
    