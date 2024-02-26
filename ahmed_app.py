from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

loaded_knn_model = joblib.load('knn_model.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Extracting input data from request
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

        # Making predictions with each model
        knn_prediction = loaded_knn_model.predict(input_data)
        print(knn_prediction)

        return jsonify({'knn_prediction': knn_prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
