# Spaceship Titanic - XGBoost Solution

This repository contains a baseline solution for the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic), using XGBoost and robust feature engineering.

## Approach

- **Consistent Preprocessing:** Train and test data are concatenated for uniform handling of missing values and categorical encoding.
- **Feature Engineering:** 
  - Categorical columns are encoded as integers.
  - The `Cabin` column is split into `CabinDeck`, `CabinNum`, and `CabinSide`.
  - Boolean columns (`CryoSleep`, `VIP`) are filled and converted to integers.
  - Numerical columns with missing values are filled with median or zero.
- **Model:** XGBoost Classifier with tuned hyperparameters.

## Features Used

- `HomePlanet`, `CryoSleep`, `Destination`, `Age`, `VIP`
- `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
- `CabinDeck`, `CabinSide`

## Code Summary

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Target
y = train['Transported'].astype(int)

# Combine train and test for consistent preprocessing
test['Transported'] = np.nan
full = pd.concat([train, test], sort=False)

# Fill missing values and encode categorical features
full['HomePlanet'] = full['HomePlanet'].fillna('Unknown')
full['CryoSleep'] = full['CryoSleep'].fillna(False).astype(int)
full['Cabin'] = full['Cabin'].fillna('Unknown/Unknown/Unknown')
full['Destination'] = full['Destination'].fillna('Unknown')
full['Age'] = full['Age'].fillna(full['Age'].median())
full['VIP'] = full['VIP'].fillna(False).astype(int)
for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    full[col] = full[col].fillna(0)

# Split Cabin into 3 features
full[['CabinDeck', 'CabinNum', 'CabinSide']] = full['Cabin'].str.split('/', expand=True)
full['CabinDeck'] = full['CabinDeck'].astype('category').cat.codes
full['CabinSide'] = full['CabinSide'].astype('category').cat.codes

# Encode categorical variables
for col in ['HomePlanet', 'Destination']:
    full[col] = full[col].astype('category').cat.codes

# Feature list
features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'CabinDeck', 'CabinSide'
]

# Split back to train and test by using the target variable
X = full.loc[full['Transported'].notnull(), features]
X_test = full.loc[full['Transported'].isnull(), features]

# Train the model
model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X, y)

# Predict 'Transported' for test data
test_preds = model.predict(X_test)
test['Transported'] = test_preds.astype(bool)

# Save the submission
test[['PassengerId', 'Transported']].to_csv('submission.csv', index=False)
