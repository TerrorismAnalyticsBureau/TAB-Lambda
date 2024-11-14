from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/execute', methods=['GET'])
def execute_command():
    response = {
        'output': 'Hello, world!'
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

