from generate import generate
from typing import Dict
import threading
import tornado
import tornado.ioloop
import tornado.web
import logging
import config

import json

class StartHandler(tornado.web.RequestHandler):
    def post(self):


       # self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
       # t = time.time()
        returned_result = {'status': 'unknown request'}
        payload = json.loads(self.request.body.decode('utf-8'))
        returned_result = execute_action(payload)

        logging.log(logging.INFO,"Hello!")
        self.write(json.dumps(returned_result, ensure_ascii=False).encode('utf8'))


def execute_generate(payload: Dict[str, str]) -> Dict[str, str]:

    if 'length' not in payload:
        return {'status': 'fail', 'error': 'length of generated text not specified'}
    topp = payload['top_p'] if 'top_p' in payload else 0.9
    topk = payload['top_k'] if 'top_k' in payload else 10
    beam_size = payload['beam_size'] if 'beam_size' in payload else 10
    numsamples = payload['numsamples'] if 'numsamples' in payload else 5
    length = payload['length'] if 'length' in payload else 50
    context = payload['context']
    context= context.lower()
    result = generate(payload['context'],topp=topp,topk=topk,beam_size=beam_size,numsamples=numsamples,length=length)

    if result is None:
        returned_result = {'status': 'failed',
                           'error': 'requested model is not loaded'}
        if config.control_permissions:
            db.record_user_query(payload['key'], 'generate', 'failed')
    else:
        returned_result = {'status': 'ok', 'text': result}
        if config.control_permissions:
            db.record_user_query(payload['key'], 'generate', payload['context'] + '||' + result)
    return returned_result

def execute_action(payload: Dict[str, str]) -> Dict[str, str]:

    if payload['action'] == 'generate':
        returned_result = execute_generate(payload)

    return returned_result
