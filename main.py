import json
import urllib.request
from flask import Flask
from flask import request
import coronacop

app = Flask(__name__)

@app.route('/')
def handler():
    
    image = request.args['image']
    urllib.request.urlretrieve(image, './tmp/image.jpg')
    args = coronacop.def_args
    args["picture"] = './tmp/image.jpg'
    args["write"]   = './tmp/out'
    unmasked = coronacop.run(args)
    return json.dumps({
        "unmasked": unmasked>0.5
    })