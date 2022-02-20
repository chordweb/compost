from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def do_ml(bio, groups):
    # groups = [{description: "", _id: ""}]
    # bio = ""
    # return an array of ids
    return []

@app.route("/", methods=["POST"])
def hello_world():
    bio = request.json["bio"]
    groups = request.json["groups"]
    res = do_ml(bio, groups)
    return {"matches": res}

app.run(port=7070)