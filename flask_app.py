from flask import Flask, request, jsonify
import pandas as pd
from src.utils.constant import model_local_path
import tensorflow as tf


app = Flask(__name__)


@app.route('/health', methods=["GET"])
def health_check():
    # Health check endpoint
    return "Ok"


@app.route('/predict', methods=["POST"])
def predict_pulsar_json():
    """
    Endpoint to predict if an input in JSON format contains house price data.
    ---
    parameters:
      - name: input_json
        in: body
        type: application/json
        required: true
    responses:
      200:
        description: The predicted value
    """
    try:
        # Extracting input parameters from the JSON request
        input_data = request.get_json()

        # Convert input data to DataFrame
        df_test = pd.DataFrame(input_data)

        # Making predictions using the loaded classifier
        prediction = model.predict(df_test)

        prediction_list = prediction.tolist()  # Convert to list
        cleaned_predictions = [float(pred[0]) for pred in prediction_list]  # Remove array notation and dtype

        # Define the threshold for classification
        threshold = 0.5

        # Apply the threshold to the predictions and create a list of dictionaries with serial numbers
        predicted_chances = [{'For input instance': i + 1, 'Water is': 'Potable' if pred >=
                             threshold else 'Not Potable'} for
                             i, pred in enumerate(cleaned_predictions)]
        # Return the predicted value in JSON format
        return jsonify({"predictions": predicted_chances})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    model = tf.keras.models.load_model(model_local_path)
    # Run the Flask app on all available network interfaces
    app.run(host='0.0.0.0', port=8000)
