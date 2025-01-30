from flask import Flask, jsonify, request
from ml import make_prediction
from testrepo1.supportfn.constant import load_data
import os

def create_app():
    app = Flask(__name__)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        # Add preprocessing using supportive_functions
        prediction = make_prediction(data)
        return jsonify({'prediction': prediction})
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)