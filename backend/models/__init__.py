from flask import Flask #create the Flask app
from flask_cors import CORS

app = Flask(__name__)
CORS(app, methods=['GET','POST','PUT','DELETE'],headers=['Content-Type'], origins="*")

# run routes
from backend.models import main