import json
import tornado
import tornado.ioloop
import tornado.web
import os
import traceback
import logging

from typing import Dict
import threading
from generatehandler import StartHandler

print("JJJJJJJJJJJJJJ")
settings = dict()
application = tornado.web.Application([
    (r"/", StartHandler),

],
    template_path=os.path.join(os.path.dirname(__file__), "templates"),
    static_path=os.path.join(os.path.dirname(__file__), "static"),
    autoreload=True,
    debug=True,
    **settings
)


if __name__ == "__main__":
     try:
        f = open("config.json", "r")
        # ~ a = f.read()
        config = json.load(f)
        print(config)
        server_ip = config["server-ip"]
        server_port = config["server-port"]
        protocol = 'http://'
        host_url = protocol + server_ip + ":" + server_port + "/"
        print("start with:     " + host_url)
#        db.first_run()

     except:
         print("Can not load file config.json")
         traceback.print_exc()
     else:

        print("starting server...")

        http_server = tornado.httpserver.HTTPServer(application)
        http_server.listen(server_port)


        tornado.ioloop.IOLoop.instance().start()
