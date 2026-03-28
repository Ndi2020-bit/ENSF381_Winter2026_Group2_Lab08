# Group Number: 2
# Name: Somrit Dewan, UCID: 30230926
# Name: Nnamdi Onaga, UCID: 30254264

from copy import deepcopy
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

SEEDED_USERS = {
    "1": {"id": "1", "first_name": "Ava", "user_group": 11},
    "2": {"id": "2", "first_name": "Ben", "user_group": 22},
    "3": {"id": "3", "first_name": "Chloe", "user_group": 33},
    "4": {"id": "4", "first_name": "Diego", "user_group": 44},
    "5": {"id": "5", "first_name": "Ella", "user_group": 55},
}

MODEL_PATH = Path(__file__).resolve().parent / "src" / "random_forest_model.pkl"
PREDICTION_COLUMNS = [
    "city",
    "province",
    "latitude",
    "longitude",
    "lease_term",
    "type",
    "beds",
    "baths",
    "sq_feet",
    "furnishing",
    "smoking",
    "cats",
    "dogs",
]

app = Flask(__name__)
CORS(app)
users = deepcopy(SEEDED_USERS)


# Exercise 1 

@app.route('/users', methods=['GET'])
def get_users():
    return jsonify(list(users.values())), 200


@app.route('/users', methods=['POST'])
def create_user():
    data = request.json

    # Checks if data is valid
    if data is None:
        return jsonify({"message": "Request body is missing"}), 400
    
    # Checks for missing values
    try:
        data['id'], data['first_name'], data['user_group']
    except KeyError:
        return jsonify({"message": "Missing id, first_name, or user_group"}), 400
    
    # Checks if the fields are left empty
    if data['id'] == "" or data['first_name'] == "" or data['user_group'] is None:
        return jsonify({"message": "id, first_name, or user_group cannot be left empty"}), 400

    user_id = str(data['id'])

    # Checks for duplicate entry
    if user_id in users:
        return jsonify({"message": f"User {user_id} already exists."}), 409
    
    users[user_id] = {"id": user_id, "first_name": data['first_name'], "user_group": data['user_group']}
    return jsonify({"id": user_id, "first_name": data['first_name'], "user_group": data['user_group'], "message": f"Created user {user_id}."}), 201


@app.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.json

    # Checks if data is valid
    if data is None:
        return jsonify({"message": "Request body is missing"}), 400

    # Checks if request body is invalid
    try:
        data['first_name'], data['user_group']
    except KeyError:
        return jsonify({"message": "Missing first_name or user_group"}), 400
    
    # Checks if the fields are left empty
    if data['first_name'] == "" or data['user_group'] is None:
        return jsonify({"message": "first_name or user_group cannot be left empty"}), 400

    # Checks if the user_id exists
    if user_id not in users:
        return jsonify({"message": f"User {user_id} was not found."}), 404
    
    users[user_id] = {"id": user_id, "first_name": data['first_name'], "user_group": data['user_group']}
    return jsonify({"id": user_id, "first_name": data['first_name'], "user_group": data['user_group'], "message": f"Updated user {user_id}."}), 200


@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    # Checks if the user_id exists
    if user_id not in users:
        return jsonify({"message": f"User {user_id} was not found."}), 404
    del users[user_id]
    return jsonify({"message": f"Deleted user {user_id}."}), 200


# Exercise 2

@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
    try:
        model = joblib.load(MODEL_PATH)
        data = request.json
        pets = bool(data['pets'])
        sample_data = [
            str(data['city']),
            str(data['province']),
            float(data['latitude']),
            float(data['longitude']),
            str(data['lease_term']),
            str(data['type']),
            float(data['beds']),
            float(data['baths']),
            float(data['sq_feet']),
            str(data['furnishing']),
            str(data['smoking']),
            pets,
            pets,
        ]

        sample_df = pd.DataFrame([sample_data], columns=[
            'city', 'province', 'latitude', 'longitude', 'lease_term',
            'type', 'beds', 'baths', 'sq_feet', 'furnishing',
            'smoking', 'cats', 'dogs'
            ])
        predicted_price = model.predict(sample_df)
        return jsonify({"predicted_price": float(predicted_price[0])}), 200
    
    except Exception as e:
        return jsonify({"message": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5050)