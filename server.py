import http.server
import socketserver


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()

class PragraphProcessor():
    def __init__(self):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        self.embed = hub.Module(module_url)
    
    def get_vector(self, paragraph):
        print("Input for vector calculation: {}".format(paragraph))
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            paragraph_embedding = session.run(self.embed([paragraph]))

            return paragraph_embedding

globalPragraphProcessor = PragraphProcessor()

class PragraphProcessorHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        content_len = int(self.headers.get('Content-Length'))
        body = str(self.rfile.read(content_len),'utf-8')

        vector = globalPragraphProcessor.get_vector(body)

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(''.join(str(d) for d in vector), 'utf-8'))
        print("Vectorization request finished.", PORT)
    
    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        body = str(self.rfile.read(content_len),'utf-8')

        vector = globalPragraphProcessor.get_vector(body)

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(''.join(str(d) for d in vector), 'utf-8'))
        print("Vectorization request finished.", PORT)

    
PORT = 1020

with socketserver.TCPServer(("", PORT), PragraphProcessorHandler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()




#docker pull tomchl/vectorizationpyserver
#docker run --publish=1020:1020 vectorization