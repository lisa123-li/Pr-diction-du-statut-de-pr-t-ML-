import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Charger le dataset
dataset = pd.read_csv('data-set.csv')
pd.set_option('display.max_columns', None)

# Afficher les valeurs manquantes
print("Valeurs manquantes avant traitement :")
print(dataset.isnull().sum())

# Supprimer les lignes avec des valeurs manquantes
dataset = dataset.dropna()

# Vérifier que les valeurs manquantes ont été supprimées
print("\nValeurs manquantes après traitement :")
print(dataset.isnull().sum())

# Remplacement de certaines valeurs texte par des valeurs numériques
dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
dataset.replace(to_replace='3+', value=4, inplace=True)

# Afficher les dépendants après remplacement
print("\nValeurs de 'Dependents' :")
print(dataset['Dependents'].value_counts())

# Visualisation
sns.countplot(x='Education', hue='Loan_Status', data=dataset)
plt.show()

# Conversion des colonnes texte en valeurs numériques
dataset.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)

# Séparation des données (features et étiquette)
x = dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
y = dataset['Loan_Status']

# Division en données d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, stratify=y, random_state=2)

# Entraînement du modèle
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Prédictions sur les données d'entraînement
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('\nAccuracy sur les données d’entraînement :', training_data_accuracy)

# Prédictions sur les données de test
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy sur les données de test :', test_data_accuracy)

# Faire une prédiction sur de nouvelles données (système prédictif)
# Exemple : [Male, 1 dépendant, marié, diplômé, salarié, 5000 revenu, 2000 coapplicant, 128 prêt demandé, 360 durée, 1 historique crédit, 1 urbain]
input_data = (1, 1, 1, 1, 0, 5000, 2000, 128, 360, 1.0, 2)

# Conversion en array numpy
input_data_as_numpy_array = np.asarray(input_data)

# Reshape pour prédire un seul échantillon
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Prédiction
prediction = classifier.predict(input_data_reshaped)

# Affichage
if prediction[0] == 1:
    print('\n Le prêt est approuvé.')
else:
    print('\n Le prêt n’est pas approuvé.')