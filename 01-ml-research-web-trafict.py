# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load Dataset
data = pd.read_csv('website_data.csv')

# 2. Exploratory Data Analysis (EDA)
print("=== Info Dataset ===")
print(data.info())

print("\n=== Deskripsi Statistik ===")
print(data.describe())

print("\n=== Distribusi Target ===")
print(data['Seasonal Fluctuations'].value_counts())

# Visualisasi distribusi target
plt.figure(figsize=(8, 5))
sns.countplot(x='Seasonal Fluctuations', data=data)
plt.title('Distribusi Kategori Seasonal Fluctuations')
plt.show()

# 3. Preprocessing Data
# Pisahkan fitur dan target
X = data.drop('Seasonal Fluctuations', axis=1)
y = data['Seasonal Fluctuations']

# Identifikasi kolom numerik dan kategorikal
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Buat transformer untuk preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Bangun Pipeline Model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Untuk handle class imbalance
    ))
])

# 6. Training Model
model.fit(X_train, y_train)

# 7. Evaluasi Model
# Prediksi pada data test
y_pred = model.predict(X_test)

# Print classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Print accuracy
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 8. Feature Importance
# Dapatkan nama fitur setelah preprocessing
# Untuk fitur numerik
num_feature_names = numeric_features.tolist()

# Untuk fitur kategorikal yang sudah di-encode
if len(categorical_features) > 0:
    cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = num_feature_names + cat_feature_names
else:
    feature_names = num_feature_names

# Dapatkan feature importance
importances = model.named_steps['classifier'].feature_importances_

# Buat DataFrame untuk visualisasi
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Visualisasi feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importance')
plt.show()

# 9. Prediksi Contoh Data Baru (Opsional)
# Contoh data baru dengan struktur yang sama dengan X
new_data = pd.DataFrame({
    'Customer Behavior': [50.0],
    'Market Trends': [1.0],
    'Product Availability': [75],
    'Customer Demographics': ['36-45'],
    'Website Traffic': [200],
    'Engagement Rate': [0.05],
    'Discount Rate': [0.1],
    'Advertising Spend': [1000],
    'Social Media Engagement': [50],
    'Returning Customers Rate': [0.2],
    'New Customers Count': [10],
    'Average Order Value': [150],
    'Shipping Speed': [2],
    'Customer Satisfaction Score': [4],
    'Economic Indicator': [1.0]
})

# Prediksi kategori seasonal fluctuations
prediction = model.predict(new_data)
print(f"\nPrediksi untuk data baru: {prediction[0]}")