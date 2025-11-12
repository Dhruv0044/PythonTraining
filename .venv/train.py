import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

print("Script started...")

# --- 1. Load Data ---
try:
    data = pd.read_csv('quikr_car.csv')
except FileNotFoundError:
    print("Error: 'car_data.csv' not found.")
    print("Please make sure your data file is in the same directory.")
    exit()

print("Data loaded.")

# --- START: DATA CLEANING BLOCK ---
# We will clean all columns that are supposed to be numeric.

# Clean 'Price' column
print("Cleaning 'Price' column...")
data['Price'] = data['Price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
print("'Price' column cleaned.")

# Clean 'kms_driven' column
print("Cleaning 'kms_driven' column...")
data['kms_driven'] = data['kms_driven'].astype(str).str.replace(r'[^\d.]', '', regex=True)
data['kms_driven'] = pd.to_numeric(data['kms_driven'], errors='coerce')
print("'kms_driven' column cleaned.")

# Clean 'year' column
print("Cleaning 'year' column...")
data['year'] = data['year'].astype(str).str.replace(r'[^\d.]', '', regex=True)
data['year'] = pd.to_numeric(data['year'], errors='coerce')
print("'year' column cleaned.")

# Drop any rows where 'Price', 'kms_driven', or 'year' could not be converted
critical_columns = ['Price', 'kms_driven', 'year']
initial_rows = len(data)
data = data.dropna(subset=critical_columns)
final_rows = len(data)
print(f"Dropped {initial_rows - final_rows} rows due to missing numeric data.")

# Ensure correct data types
data['Price'] = data['Price'].astype(float)
data['kms_driven'] = data['kms_driven'].astype(float)
data['year'] = data['year'].astype(int)

# --- END: DATA CLEANING BLOCK ---


# --- 2. Preprocessing (from your script) ---
# Drop the unnecessary 'index' column if it exists
if 'index' in data.columns:
    data = data.drop('index', axis=1)

# Remove outliers using IQR method
Q1 = data['Price'].quantile(0.25)
Q3 = data['Price'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['Price'] >= Q1 - 1.5 * IQR) & (data['Price'] <= Q3 + 1.5 * IQR)]

print("Data outliers removed.")

# Define features (x) and target (y)
x = data.drop('Price', axis=1)
y = data['Price']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Data split into train/test sets.")

# --- 3. Model Training (from your script) ---
# Create column transformer with OHE and scaling
columns_transform = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['name', 'company', 'fuel_type']),
    ('scaler', StandardScaler(), ['year', 'kms_driven'])
], remainder='passthrough') # Use remainder='passthrough' if there are other columns

# Try multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=21, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=21),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=21, max_depth=10)
}

results = {}
for name, model in models.items():
    pipe = make_pipeline(columns_transform, model)
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    score = r2_score(y_test, y_pred)
    results[name] = score
    print(f'{name}: R² Score = {score:.4f}')

# Use the best model
best_model_name = max(results, key=results.get)
print(f'\nBest Model: {best_model_name} with R² Score: {results[best_model_name]:.4f}')

# Train final model with best performer
final_pipe = make_pipeline(columns_transform, models[best_model_name])
final_pipe.fit(x_train, y_train)

print("Final model trained.")

# --- 4. Save (Dump) the Model and Data ---
# Save the final pipeline
with open('final_pipe.pkl', 'wb') as f:
    pickle.dump(final_pipe, f)

# Save the 'x' dataframe (pre-split) for dropdown options in the app
with open('data_features.pkl', 'wb') as f:
    pickle.dump(x, f)

print("\nScript finished.")
print("Saved 'final__pipe.pkl' (the model)")
print("Saved 'data__features.pkl' (for app UI options)")