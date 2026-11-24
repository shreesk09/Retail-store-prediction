# =========================================================
# STEP 0: INSTALL (only if needed)
# =========================================================
# pip install pandas numpy matplotlib seaborn scikit-learn joblib


# =========================================================
# STEP 1: IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Install first:
# pip install xgboost

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# =========================================================
# STEP 2: LOAD DATASET
# 
# =========================================================
df = pd.read_csv(r"D:/Major_Project_Nyeras/Dataset/train.csv", encoding='latin1')

print("First 5 rows:")
print(df.head())


# =========================================================
#  PHASE 1: DATA CLEANING + EDA
# =========================================================

# -------------------------------
# BASIC INFO
# -------------------------------
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
df.info()

# -------------------------------
# MISSING VALUES
# -------------------------------
print("\nMissing Values:\n", df.isnull().sum())

# Fix missing Postal Code
df['Postal Code'].fillna(df['Postal Code'].median(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# -------------------------------
# FIX DATE FORMAT
# -------------------------------
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day


# =========================================================
#  EDA VISUALIZATIONS
# =========================================================

# Sales Distribution
plt.figure()
sns.histplot(df['Sales'], kde=True)
plt.title("Sales Distribution")
plt.show()

# Category vs Sales
plt.figure()
sns.barplot(x='Category', y='Sales', data=df)
plt.xticks(rotation=45)
plt.title("Sales by Category")
plt.show()

# Region vs Sales
plt.figure()
sns.barplot(x='Region', y='Sales', data=df)
plt.title("Sales by Region")
plt.show()

# Monthly Trend
monthly_sales = df.groupby('Month')['Sales'].sum()
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.show()

# Correlation
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Matrix")
plt.show()


# =========================================================
# SAVE CLEANED DATA
# =========================================================
df.to_csv("cleaned_superstore1.csv", index=False)


# =========================================================
#  PHASE 2: MACHINE LEARNING
# =========================================================

# =========================================================
# FEATURE SELECTION (UPDATED FOR YOUR DATASET)
# =========================================================

df_model = df[['Category', 'Sub-Category', 'Region', 'Segment', 'Month', 'Year', 'Sales']]

# Encode categorical variables
df_model = pd.get_dummies(df_model, drop_first=True)

# Define features and target
X = df_model.drop('Sales', axis=1)
y = df_model['Sales']


# =========================================================
# TRAIN-TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================================
# SCALING (for Linear Regression only)
# =========================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# MODEL 1: LINEAR REGRESSION (FIXED)
# =========================================================
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)


# =========================================================
# MODEL 2: RANDOM FOREST (IMPROVED)
# =========================================================
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# =========================================================
# MODEL 3: DECISION TREE
# =========================================================

dt = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

# =========================================================
# MODEL 4: GRADIENT BOOSTING
# =========================================================

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)

# =========================================================
# MODEL 5: XGBOOST
# =========================================================

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)


# =========================================================
# EVALUATION FUNCTION
# =========================================================
def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2


# =========================================================
# RESULTS
# =========================================================

lr_results = evaluate(y_test, y_pred_lr)

rf_results = evaluate(y_test, y_pred_rf)

dt_results = evaluate(y_test, y_pred_dt)

gb_results = evaluate(y_test, y_pred_gb)

xgb_results = evaluate(y_test, y_pred_xgb)

print("\nLinear Regression Results:")
print("MAE:", lr_results[0])
print("RMSE:", lr_results[1])
print("R2:", lr_results[2])

print("\nDecision Tree Results:")
print("MAE:", dt_results[0])
print("RMSE:", dt_results[1])
print("R2:", dt_results[2])

print("\nRandom Forest Results:")
print("MAE:", rf_results[0])
print("RMSE:", rf_results[1])
print("R2:", rf_results[2])

print("\nGradient Boosting Results:")
print("MAE:", gb_results[0])
print("RMSE:", gb_results[1])
print("R2:", gb_results[2])

print("\nXGBoost Results:")
print("MAE:", xgb_results[0])
print("RMSE:", xgb_results[1])
print("R2:", xgb_results[2])


# =========================================================
# MODEL COMPARISON
# =========================================================
results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Decision Tree',
        'Random Forest',
        'Gradient Boosting',
        'XGBoost'
    ],

    'MAE': [
        lr_results[0],
        dt_results[0],
        rf_results[0],
        gb_results[0],
        xgb_results[0]
    ],

    'RMSE': [
        lr_results[1],
        dt_results[1],
        rf_results[1],
        gb_results[1],
        xgb_results[1]
    ],

    'R2 Score': [
        lr_results[2],
        dt_results[2],
        rf_results[2],
        gb_results[2],
        xgb_results[2]
    ]
})

print("\nModel Comparison:")
print(results)

plt.figure(figsize=(10,5))

sns.barplot(
    data=results,
    x='Model',
    y='R2 Score'
)

plt.title("Model Performance Comparison (R² Score)")
plt.xticks(rotation=20)

plt.show()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top Features Influencing Sales")
plt.show()


# =========================================================
# SAVE MODEL
# =========================================================
joblib.dump(xgb, "best_sales_model.pkl")

print("\nModel saved successfully!")

joblib.dump(
    X.columns.tolist(),
    "feature_columns.pkl"
)

print("Model Saved")
print("Feature Columns Saved")

results.to_csv("model_results.csv", index=False)
print("Model Results Saved")


# =========================================================
#  END
# =========================================================