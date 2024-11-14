from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/execute', methods=['GET'])
def execute_command():
    response = {
        'output': 'DEMO OUTPUT: Next predicted event will be in Country: Iraq with Primary Attack Type: Bombing/ Explosion by Perpetrators: Unknown on Date: 06/15/2018'
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

