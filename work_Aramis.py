from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Charger l'ensemble de données Iris
iris = load_iris()
X = iris.data
y = iris.target
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser et entraîner le modèle GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_train)

# Prédire les étiquettes de cluster pour les données de test
cluster_labels = gmm.predict(X_test)

# Assigner une classe à chaque cluster en utilisant la majorité des votes
def assign_majority_class(cluster_labels, true_labels):
    unique_clusters = np.unique(cluster_labels)
    assigned_classes = np.zeros_like(unique_clusters)
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)
        cluster_classes = true_labels[cluster_indices]
        assigned_classes[cluster] = np.argmax(np.bincount(cluster_classes))
    return assigned_classes

assigned_classes = assign_majority_class(cluster_labels, y_test)

# Associer les étiquettes de classe prédites avec les clusters
predicted_labels = np.array([assigned_classes[label] for label in cluster_labels])

# Calculer l'exactitude (accuracy) de la prédiction
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Exactitude (Accuracy) : {accuracy:.2f}")
import joblib

joblib.dump(gmm, 'gmm_model.pkl')

loaded_knn_model = joblib.load('gmm_model.pkl')