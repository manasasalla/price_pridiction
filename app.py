import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Step 1: Load Your Dataset
# Replace 'car_data.csv' with the actual path of your dataset
data = pd.read_csv("C:\\Users\\Saiteja\\Downloads\\car_data.csv")

# Step 2: Inspect the dataset and select the correct feature columns
print(data.columns)

# Step 3: Preprocess the data by converting categorical columns to numerical
# One-Hot Encoding for categorical variables (Fuel_Type, Transmission, Owner, Selling_type)
# Convert categorical columns into binary columns using One-Hot Encoding
data = pd.get_dummies(data, columns=['Car_Name', 'Fuel_Type', 'Transmission', 'Owner', 'Selling_type'], drop_first=True)




# Step 4: Define the features (X) and target (y)
# Here, I assume 'Selling_Price' is your target variable and other columns are features
X = data[['Year', 'Driven_kms', 'Present_Price', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Transmission_Manual', 'Owner_1', 'Selling_type_Individual']]
y = data['Selling_Price']

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'rf_model.pkl')

# Step 8: Evaluate the Model
score = model.score(X_test, y_test)
print(f"Model R^2 score: {score:.2f}")