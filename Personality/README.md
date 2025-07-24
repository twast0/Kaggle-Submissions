# Personality Prediction - Random Forest Solution

This repository contains a baseline solution for a personality classification task using a Random Forest classifier and robust preprocessing.

## Approach

- **Data Loading:** Reads `train.csv` and `test.csv` into pandas DataFrames.
- **Preprocessing:**
  - Converts binary categorical columns (`Stage_fear`, `Drained_after_socializing`) from `'Yes'/'No'` to `1/0`.
  - Uses `KNNImputer` to fill missing values in numeric columns.
  - Ensures binary columns are rounded to `0` or `1` after imputation.
- **Model:** Trains a `RandomForestClassifier` on the processed data.
- **Evaluation:** Prints accuracy and a classification report on a validation split.
- **Prediction:** Applies the same preprocessing to the test set and predicts the `Personality` class.
- **Submission:** Saves predictions as `test_predictions.csv` with `id` and `Predicted_Personality`.

## Features Used

- All columns except the target (`Personality`) are used as features.
- Binary columns are mapped and rounded to ensure correct format.

## How to Use

1. Place `train.csv` and `test.csv` in the same directory as the notebook/script.
2. Run the notebook or script.
3. The output file `test_predictions.csv` will be created, ready for submission or further analysis.

## Code Summary

```python
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv('train.csv')
y = df['Personality']
X = df.drop('Personality', axis=1)

# Convert binary columns
X['Stage_fear'] = X['Stage_fear'].map({'Yes': 1, 'No': 0})
X['Drained_after_socializing'] = X['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Impute missing values
imputer = KNNImputer(n_neighbors=5)
numeric_cols = X.select_dtypes(include=['number']).columns
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

# Round binary columns
X['Stage_fear'] = X['Stage_fear'].round().astype(int)
X['Drained_after_socializing'] = X['Drained_after_socializing'].round().astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict on test set
test_df = pd.read_csv('test.csv')
X_test_final = test_df.copy()
X_test_final['Stage_fear'] = X_test_final['Stage_fear'].map({'Yes': 1, 'No': 0})
X_test_final['Drained_after_socializing'] = X_test_final['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
numeric_cols_test = X_test_final.select_dtypes(include=['number']).columns
X_test_final[numeric_cols_test] = imputer.transform(X_test_final[numeric_cols_test])
X_test_final['Stage_fear'] = X_test_final['Stage_fear'].round().astype(int)
X_test_final['Drained_after_socializing'] = X_test_final['Drained_after_socializing'].round().astype(int)
test_predictions = rf.predict(X_test_final)
test_df['Predicted_Personality'] = test_predictions
test_df[['id', 'Predicted_Personality']].to_csv('test_predictions.csv', index=False)
