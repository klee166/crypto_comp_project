from flask import Flask, render_template
app = Flask(__name__)

@app.route("/send", methods=['GET','POST'])
def send():
	if request.method == 'POST':
		if request.form['submit'] == '1: Produce matrix of web crawling':
			pass
	#return 'hello world'
		elif request.form['submit'] == '2: Query Search':
			pass
		else:
			pass
	return render_template('index.html')

if __name__ == "__main__":
	app.run()
