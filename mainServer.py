## do this first
# pip install pyngrok
# ngrok authtoken 1nxJ3mokhH53txkpzU7qos8xNFF_2LGE2FBWpJf7hdZi5pwZZ

from flask import Flask, request
from quick_flask_server import run_with_ngrok
from CloudPubSub.myPublish import publish
app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

@app.route("/")
def hello():
    return "Hello World2!"

@app.route('/add', methods=['GET', 'POST'])
def parse_request():
    url = request.args.get('url')
    # if request.method == "GET":
    publish(url)
    return f"Hi {url}"

if __name__ == '__main__':
    app.run(debug = False)