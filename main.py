from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)

CORS(app)


@app.route('/execute', methods=['GET'])
def get_predictions_string():
    # Load predictions
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)

    # Create a string representation
    result_string = (
        f"Attack Type: {predictions['attack_type']}"
        f"\nGroup Name: {predictions['group_name']}"
        f"\nCity: {predictions['city_name']}"
        f"\nProvince/ State: {predictions['provstate_name']}"
        f"\nDate: {predictions['predicted_date']}"
        f"\n"
        f"\nPrediction was last updated on {predictions['last_updated']}"
    )
    return result_string


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
