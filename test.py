import requests
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import json

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

url_knn = 'https://8b1d-37-166-85-250.ngrok-free.app/predict'
url_gmm = 'http://127.0.0.1:5002/predict'
url_svm = 'http://127.0.0.1:5003/predict'

def get_prediction(url, params):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def load_model_info():
    with open('model_balances.json', 'r') as file:
        return json.load(file)

def save_model_info(model_info):
    with open('model_balances.json', 'w') as file:
        json.dump(model_info, file, indent=4)

def adjust_weights_and_slash(all_predictions, actual_labels, model_info):
    accuracies = {}
    for model, preds in all_predictions.items():
        accuracies[model] = accuracy_score(actual_labels, preds)

    avg_accuracy = np.mean(list(accuracies.values()))
    for model, acc in accuracies.items():
        if acc < avg_accuracy:
            model_info[model]['balance'] -= 100  # Apply slashing
        model_info[model]['weight'] = acc / avg_accuracy  # Adjust weights

def aggregate_predictions_weighted(all_predictions, model_info):
    weighted_votes = np.zeros((len(X_test), len(np.unique(y_test))))
    for model, preds in all_predictions.items():
        weight = model_info[model]['weight']
        for i, pred in enumerate(preds):
            weighted_votes[i, pred] += weight
    consensus_predictions = np.argmax(weighted_votes, axis=1)
    return consensus_predictions

model_info = load_model_info()
all_predictions = {model: [] for model in model_info}

for instance in X_test:
    params = {
        'sepal_length': instance[0],
        'sepal_width': instance[1],
        'petal_length': instance[2],
        'petal_width': instance[3]
    }
    for model, url in zip(['knn', 'gmm', 'svm'], [url_knn, url_gmm, url_svm]):
        response = get_prediction(url, params)
        if response:
            all_predictions[model].append(response[f'{model}_prediction'][0])

for model in all_predictions:
    all_predictions[model] = np.array(all_predictions[model])

adjust_weights_and_slash(all_predictions, y_test, model_info)

save_model_info(model_info)

consensus_predictions = aggregate_predictions_weighted(all_predictions, model_info)

accuracy = accuracy_score(y_test, consensus_predictions)
print(f"Accuracy of aggregated meta-model: {accuracy}")
