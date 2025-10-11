import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import pickle

# Load dataset
df = pd.read_csv("cars_ridge.csv")

# Log-transform torque
df['torque'] = np.log1p(df['torque'])

# Handle outliers
discrete_cols = ['doors']
continuous_cols = ['weight', 'fuel_efficiency']

for col in discrete_cols:
    mode_val = df[col].mode()[0]
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df[col] = df[col].apply(lambda x: mode_val if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR else x)

for col in continuous_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

# Features and target
X = df[['mileage','engine_size','horsepower','torque','doors','airbags',
        'weight','fuel_efficiency','brand_score','luxury_index']]
y = df['price_k']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RidgeCV
alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, scoring='r2', cv=10)
ridge_cv.fit(X_train_scaled, y_train)

# Save model and scaler
with open('ridge_model.pkl', 'wb') as f:
    pickle.dump(ridge_cv, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Ridge model and scaler saved successfully!")
