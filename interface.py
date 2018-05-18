from flask import Flask
app = Flask(__name__)

@app.route("/")
def interface():
	return "Hello World!"