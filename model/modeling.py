import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

class ChurnModel:
    """
    Logistic Regression Model untuk Telco Customer Churn Prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_data(self, df):
        """
        Persiapan data untuk training
        """
        # Drop unnecessary columns
        df = df.drop(['CustomerID', 'Churn Label', 'Churn Reason'], axis=1, errors='ignore')
        
        # Convert Total Charges to numeric
        if 'Total Charges' in df.columns:
            df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
            df['Total Charges'] = df['Total Charges'].fillna(df['Total Charges'].median())
        
        return df
    
    def train(self, X, y):
        """
        Training model Logistic Regression
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Calculate feature importance (coefficients)
        self.feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(self.model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        return self.model, X_test_scaled, y_test
    
    def predict(self, X):
        """
        Melakukan prediksi
        """
        if self.model is None:
            raise ValueError("Model belum dilatih!")
        
        # Ensure X is a DataFrame with correct columns
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Prediksi dengan probabilitas
        """
        if self.model is None:
            raise ValueError("Model belum dilatih!")
        
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, model_path):
        """
        Simpan model ke file
        """
        joblib.dump(self.model, model_path + '_model.pkl')
        joblib.dump(self.scaler, model_path + '_scaler.pkl')
        joblib.dump(self.feature_names, model_path + '_features.pkl')
    
    def load_model(self, model_path):
        """
        Load model dari file
        """
        self.model = joblib.load(model_path + '_model.pkl')
        self.scaler = joblib.load(model_path + '_scaler.pkl')
        self.feature_names = joblib.load(model_path + '_features.pkl')
    
    def get_top_features(self, n=10):
        """
        Dapatkan top n features berdasarkan importance
        """
        if self.feature_importance is not None:
            return self.feature_importance.head(n)
        return None


def load_and_prepare_data(csv_path):
    """
    Load dan prepare data dari CSV
    """
    df = pd.read_csv(csv_path)
    model = ChurnModel()
    df = model.prepare_data(df)
    return df, model
