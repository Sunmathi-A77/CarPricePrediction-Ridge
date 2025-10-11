# Car Price Prediction Using Ridge and OLS Regression

## 🚗 Car Price Prediction App

Predict the price of a car (in ₹k) using multiple features with Ridge Regression.

Live App: https://ridge-carpriceprediction.streamlit.app/

## Project Overview

This project aims to predict car prices (price_k) using multiple features such as mileage, engine size, horsepower, torque, doors, airbags, weight, fuel efficiency, brand score, and luxury index.

We explore both Ordinary Least Squares (OLS) and Ridge Regression to handle multicollinearity and improve predictive performance.

## Dataset

Source: Kaggle

## Features:

mileage

engine_size

horsepower

torque

doors

airbags

weight

fuel_efficiency

brand_score

luxury_index

Target Variable: price_k (Car price in thousands)

## Steps Performed

## Exploratory Data Analysis (EDA)

Checked data types, missing values, and duplicates.

Skewness of numeric columns calculated.

## Visualizations:

Histograms for numeric columns.

Correlation Heatmap to identify relationships.



## Data Preprocessing

Log transformation applied to torque to reduce skewness.

Outliers handled:

Discrete column (doors) → replaced extreme values with mode.

Continuous columns (weight, fuel_efficiency) → capped using IQR.

## Feature selection

all 10 features used for prediction.

## Train-Test Split

Train-Test ratio: 80:20

Features scaled using StandardScaler for regression models.

## Modeling

## 1. OLS Regression

Linear regression fitted to scaled features.

Performance Check:

R² Score, RMSE, Intercept, Coefficients for each feature captured.

## 2. Ridge Regression (RidgeCV)

Ridge regression fitted with 10-fold cross-validation to select best alpha.

Performance:

R² Score, RMSE, Intercept, Coefficients are shrunk compared to OLS to reduce overfitting.

## Residual Analysis

Histogram of Ridge residuals shows roughly normal distribution.

<img width="531" height="393" alt="image" src="https://github.com/user-attachments/assets/77613d3f-44b8-4fcf-b79b-a81a2cfe045b" />

Scatter plot of residuals vs predicted price indicates random distribution with no major pattern.

<img width="543" height="393" alt="image" src="https://github.com/user-attachments/assets/dfcb24e1-c053-4d4f-97f1-4c53d46e4377" />

## Libraries and Tools Used

Python 3.x

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

## Conclusion

Both OLS and Ridge provide reasonable predictions of car prices.

Ridge regression helps handle correlated features, while OLS slightly outperforms in this dataset.

The model can be further improved using polynomial features, target transformation, or advanced regression techniques (ElasticNet, Gradient Boosting).