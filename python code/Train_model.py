import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Read the Excel file
file_path = "Crop_prediction.xlsx"
crop = pd.read_excel(file_path, engine='openpyxl')

# Preprocess the data
le = LabelEncoder()
crop['Region'] = le.fit_transform(crop['Region'])
crop['WeatherCondition'] = le.fit_transform(crop['WeatherCondition'])
crop['Bulk Nutrient'] = le.fit_transform(crop['Bulk Nutrient'])
crop['SeedQuality'] = le.fit_transform(crop['SeedQuality'])
crop['Crop Cultivation'] = le.fit_transform(crop['Crop_Cultivation'])

# Train-test split
X = crop.drop(['Crop_Cultivation','Crop Cultivation','Irrigation Method'], axis=1)
y1 = crop['Crop_Cultivation']
y2 = crop['Irrigation Method']
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, random_state=1, test_size=0.2)
X1_train, X1_test, y2_train, y2_test = train_test_split(X, y2, random_state=1, test_size=0.2)

# Train the model for Crop Cultivation
model_cultivation = RandomForestClassifier()
model_cultivation.fit(X_train, y1_train)

# Train the model for Irrigation Method
model_irrigation = RandomForestClassifier()
model_irrigation.fit(X1_train, y2_train)

# Save the models
joblib.dump(model_cultivation, 'model_cultivation.pkl')
joblib.dump(model_irrigation, 'model_irrigation.pkl')
