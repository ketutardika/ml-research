# Import library
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
diabetes = load_diabetes()
X = diabetes.data  # Fitur
y = diabetes.target  # Target (nilai progresi diabetes)

# Convert ke DataFrame untuk analisis
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y  # Tambahkan kolom target

# Cek 5 baris pertama
print(df.head())

# Statistik dasar
print(df.describe())

# Bagi data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model
model = LinearRegression()

# Latih model dengan data training
model.fit(X_train, y_train)

# Prediksi data test
y_pred = model.predict(X_test)

X_bmi = X_train[:, 2].reshape(-1, 1)  # Ambil kolom BMI saja
model.fit(X_bmi, y_train)

# Hitung Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")

# Koefisien regresi (pentingnya setiap fitur)
print("Koefisien fitur:", model.coef_)

import matplotlib.pyplot as plt

# Plot prediksi vs nilai sebenarnya
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Nilai Sebenarnya")
plt.ylabel("Prediksi")
plt.title("Actual vs Predicted")
plt.show()