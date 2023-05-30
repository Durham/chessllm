#A module that makes an api request to the model and receives from it boards generated on request
# shift + alt highlight column

import requests
from typing import List, Optional
import typing
import traceback
import json
import config
import logging
logging.basicConfig(filename = "weblogging.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_api_generate(context,token_len,num_ret_seq) -> List:
    """Makes a request to the model and gets a list of boards generated by the request
    in: context to generate, size of tokens, number of boards to generate
    out: List of generated boards"""
    try:    
        print("")
        #params = {'context': context, 'token_len': token_len, 'num_ret_seq':num_ret_seq}
        #'key':apikey,
        request1 = {'action':'generate', 'context':context,"top_p":1.0, 'temperature':0.5, 'model_name':'chess', 'maxrepeat':8, 'length':token_len,'numsamples':num_ret_seq, 'beam_size':num_ret_seq}
        request_url = config.api_request_generate_model

        #result = requests.post("http://192.168.1.45:8082/", data=params)
        answer = requests.post(request_url, data=json.dumps(request1, ensure_ascii=False).encode('utf8'))
        r = answer.json()
        model_boards_list:List = r['text']
        print('Boards:',model_boards_list)
        return model_boards_list
    except:
        print(traceback.format_exc() + "\n" + "Something went wrong.")
        logging.error('API error:'+ traceback.format_exc())
        exept_for_user = None
        return exept_for_user
        

#get_api_generate()
