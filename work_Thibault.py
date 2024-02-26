# Importation des bibliothèques nécessaires
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import joblib  # Importation de la bibliothèque pickle pour la sauvegarde du modèle

# Chargement du jeu de données Iris
iris = load_iris()
X = iris.data  # Utilisation de toutes les caractéristiques pour l'entraînement
y = iris.target

# Séparation des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)

# Création et entraînement du modèle SVM avec toutes les caractéristiques
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test et évaluation du modèle
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# Affichage des performances du modèle
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
print('Confusion Matrix:')
print(conf_matrix)

# Pour la visualisation, réduction des dimensions à 2D avec PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualisation des données réduites avec PCA

# Sauvegarde du modèle entraîné avec pickle
with open('svm_model.pkl', 'wb') as file:
    joblib.dump(model, file)
