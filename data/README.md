# Data

Notre dataset est constitué de deux fichiers CSV :

- [**`Cardiovascular Disease Prediction - For model creation.csv`**](Cardiovascular%20Disease%20Prediction%20-%20For%20model%20creation.csv) (303 observations, séparateur `;`) : Jeu de données principal destiné à l'exploration, au prétraitement et à l'entraînement des algorithmes de Machine Learning.

- [**`Cardiovascular Disease Prediction - For prediction.csv`**](Cardiovascular%20Disease%20Prediction%20-%20For%20prediction.csv) (91 observations, séparateur `,`) : Jeu de données secondaire utilisé pour évaluer le modèle et réaliser des prédictions sur de nouveaux patients.

## Structure des variables (14 colonnes)

Les deux fichiers partagent la même structure :
- **Données démographiques et symptômes :** âge, sexe, type de douleur thoracique, angine induite par l'exercice.
- **Mesures biologiques et cliniques :** tension artérielle au repos, cholestérol sérique, glycémie à jeun, thalassémie.
- **Examens cardiaques :** résultats ECG au repos, fréquence cardiaque maximale atteinte, dépression et pente du segment ST, nombre de vaisseaux principaux colorés par fluoroscopie.
- **Variable cible :** **`GOAL-Heart Disease`** (`Yes` / `No`), indiquant la présence ou non d'une maladie cardiovasculaire.