import http.server
import socketserver

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import numpy as np
from scipy.spatial import distance
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()

class PragraphProcessor():
    def __init__(self):
        self.g = tf.Graph()
        with self.g.as_default():
            # We will be feeding 1D tensors of text into the graph.
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
            self.embedded_text = embed(self.text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        self.g.finalize()
        self.session = tf.Session(graph=self.g)
        self.session.run(init_op)

    def get_vector(self, paragraph):
        print("Input for vector calculation: {}".format(paragraph))
        paragraph_embedding = self.session.run(self.embedded_text, feed_dict={self.text_input: [paragraph]})
        return paragraph_embedding

    def get_similarity(self, paragraphs):
        if(len(paragraphs) != 2):
            return 0.0

        vectors = []
        for paragraph in paragraphs:
            paragraph_embedding = self.session.run(self.embedded_text, feed_dict={self.text_input: [paragraph]})
            vectors.append(paragraph_embedding)

        return distance.cosine(vectors[0], vectors[1])


globalPragraphProcessor = PragraphProcessor()

class PragraphProcessorHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if(self.path == "/vector"):
            content_len = int(self.headers.get('Content-Length'))
            body = str(self.rfile.read(content_len),'utf-8')

            vector = globalPragraphProcessor.get_vector(body)

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(''.join(str(d) for d in vector), 'utf-8'))
            print("Vectorization request finished.", PORT)
        elif(self.path == "/cosine"):
            content_len = int(self.headers.get('Content-Length'))
            body = str(self.rfile.read(content_len),'utf-8').split("&&")

            similarity = globalPragraphProcessor.get_similarity(body)

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(str(similarity), 'utf-8'))
            print("Similarity request finished.", PORT)
        else:
            self.send_response(401)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write("not supported endpoint", 'utf-8')
            print("WrongEndpointCalled.", PORT)

    
PORT = 1020

with socketserver.TCPServer(("", PORT), PragraphProcessorHandler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()




#docker pull tomchl/vectorizationpyserver
#docker run --publish=1020:1020 vectorization