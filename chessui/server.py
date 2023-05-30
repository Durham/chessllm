import json
import tornado
import tornado.ioloop
import tornado.web
import config
import os
import traceback
import ssl
from DisplaysAnswerHandler import DisplayFormHandler


application = tornado.web.Application([
    (r"/", DisplayFormHandler),
    (r'/img/(.*)', tornado.web.StaticFileHandler, {'path': 'static/img'})

 ],
 template_path=os.path.join(os.path.dirname(__file__), "templates"),
 autoreload=True,
 autoescape=None,
 debug=True, cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__"

)

if __name__ == "__main__":
    try:
        f = open("config.json", "r")
        # ~ a = f.read()
        config_data = json.load(f)
        server_ip = config_data["server-ip"]
        server_port = config_data["server-port"]
        protocol = 'http://'
        host_url = protocol + server_ip + ":" + server_port + "/" 
        print("start with:     " + host_url)
    except:
        print("Can not load file config.json")
        traceback.print_exc()
    else:
        print("loading server")
        #application.listen(int(server_port))
        if config.is_https:
            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain(os.path.join(os.path.dirname(__file__), "/fullchain.pem"), os.path.join(os.path.dirname(__file__), "/privkey.pem"))
            http_server = tornado.httpserver.HTTPServer(application, ssl_options=ssl_ctx)
            http_server.listen(443)
        application.listen(80 if config.is_https else server_port) 
        print("starting")
        tornado.ioloop.IOLoop.instance().start()

  # tornado.ioloop.IOLoop.current().start()
