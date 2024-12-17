# Importing the necessary libraries.

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Load your dataset (you can replace 'data.csv' with your own file)
df = pd.read_csv("E:/MINI PROJECT 2/Data/features_30_sec.csv")

# Assuming the target variable is in column 'label'
X = df.drop(columns=['filename', 'label'])
y = df['label']

# Standardize features (optional but recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a decision tree classifier for feature selection
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_scaled, y)

# Feature selection using decision tree
selector = SelectFromModel(clf, threshold='mean')
selector.fit(X_scaled, y)

# Get selected features
selected_features = X.columns[selector.get_support()][:15]

# Extract features (X) and target labels (y)
X1 = df[selected_features]
y1 = df['label']

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Set up the hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],          # Number of trees in the forest
    'max_features': ['auto', 'sqrt', 'log2'], # Number of features to consider at every split
    'max_depth': [None, 10, 20, 30],          # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10],          # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],            # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]                 # Whether bootstrap samples are used when building trees
}

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the model
grid_search.fit(X1, y1)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Using the best estimator from grid search
best_rfc = grid_search.best_estimator_

x1_train, x1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2)
best_rfc.fit(x1_train, y1_train)
y_pred = best_rfc.predict(x1_test)
print(accuracy_score(y_pred, y1_test))

# Save the trained model to a file
joblib.dump(best_rfc, 'best_rf_model.pkl')
print("Model saved successfully.")
