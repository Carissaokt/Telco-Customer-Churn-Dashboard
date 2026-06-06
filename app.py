from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
import dagshub
import os
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from model.modeling import ChurnModel

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'telco_churn_analysis_2026'

# DagsHub configuration
dagshub.init(repo_owner='Carissaokt', repo_name='Telco-Customer-Churn-Dashboard', mlflow=True)

# Global model instance
churn_model = ChurnModel()
model_stats = {}

# Feature mapping untuk prediction
FEATURE_MAPPING = {
    'Contract_Month-to-month': 0,
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'Internet Service_DSL': 0,
    'Internet Service_Fiber optic': 0,
    'Internet Service_No': 0,
    'Tech support_Yes': 0,
    'Tech support_No': 0,
    'Online security_Yes': 0,
    'Online security_No': 0,
    'Payment method_Electronic check': 0,
    'Payment method_Mailed check': 0,
    'Payment method_Bank transfer (automatic)': 0,
    'Payment method_Credit card (automatic)': 0,
    'Device protection_Yes': 0,
    'Device protection_No': 0,
    'Online backup_Yes': 0,
    'Online backup_No': 0,
}

# Sample features list untuk training model
SAMPLE_FEATURES = [
    'Tenure Months',
    'Monthly Charges',
    'Total Charges',
    'Contract_One year',
    'Contract_Two year',
    'Internet Service_Fiber optic',
    'Internet Service_No',
    'Tech support_Yes',
    'Online security_Yes',
    'Payment method_Bank transfer (automatic)',
    'Payment method_Credit card (automatic)',
    'Device protection_Yes',
    'Online backup_Yes'
]


def train_model_with_data():
    """
    Train model dengan sample data dari Telco dataset
    """
    try:
        # Generate sample training data
        np.random.seed(42)
        n_samples = 7043
        
        # Create synthetic data similar to Telco dataset
        data = {
            'Tenure Months': np.random.randint(0, 73, n_samples),
            'Monthly Charges': np.random.uniform(18, 119, n_samples),
            'Total Charges': np.random.uniform(0, 10000, n_samples),
            'Contract_One year': np.random.choice([0, 1], n_samples),
            'Contract_Two year': np.random.choice([0, 1], n_samples),
            'Internet Service_Fiber optic': np.random.choice([0, 1], n_samples),
            'Internet Service_No': np.random.choice([0, 1], n_samples),
            'Tech support_Yes': np.random.choice([0, 1], n_samples),
            'Online security_Yes': np.random.choice([0, 1], n_samples),
            'Payment method_Bank transfer (automatic)': np.random.choice([0, 1], n_samples),
            'Payment method_Credit card (automatic)': np.random.choice([0, 1], n_samples),
            'Device protection_Yes': np.random.choice([0, 1], n_samples),
            'Online backup_Yes': np.random.choice([0, 1], n_samples),
        }
        
        X = pd.DataFrame(data)
        
        # Create target variable with realistic distribution
        # Churn probability based on features
        churn_prob = (
            0.5 -
            (X['Tenure Months'] / 100) -
            (X['Contract_Two year'] * 0.3) -
            (X['Tech support_Yes'] * 0.15) -
            (X['Online security_Yes'] * 0.1) +
            (X['Monthly Charges'] / 200) +
            np.random.normal(0, 0.1, n_samples)
        )
        churn_prob = np.clip(churn_prob, 0, 1)
        y = (churn_prob > 0.4).astype(int)
        
        # Train model
        model, X_test, y_test = churn_model.train(X, y)
        
        # Evaluate model on test split
        y_pred = churn_model.predict(X_test)
        y_proba = churn_model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        train_samples = len(X) - len(X_test)
        test_samples = len(X_test)

        # Store training and evaluation statistics
        model_stats.update({
            'model_type': 'Logistic Regression',
            'feature_count': len(churn_model.feature_names),
            'training_samples': len(X),
            'training_data_count': train_samples,
            'testing_data_count': test_samples,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        })

        # Log metrics to DagsHub / MLflow
        with mlflow.start_run(run_name='telco_churn_training'):
            try:
                mlflow.log_param('model', 'Logistic Regression')
                mlflow.log_param('feature_count', len(churn_model.feature_names))
                mlflow.log_param('training_samples', len(X))
                mlflow.log_metric('accuracy', float(accuracy))
                mlflow.log_metric('precision', float(precision))
                mlflow.log_metric('recall', float(recall))
                mlflow.log_metric('f1', float(f1))
                mlflow.log_metric('roc_auc', float(roc_auc))
                mlflow.log_metric('training_data_count', float(train_samples))
                mlflow.log_metric('testing_data_count', float(test_samples))
                
                # Try to log model
                try:
                    mlflow.sklearn.log_model(churn_model.model, 'churn_model')
                except Exception as model_err:
                    print(f"⚠️ Warning: Could not log model artifact: {str(model_err)}")
                
            finally:
                mlflow.end_run()

        print("✅ Model trained successfully!")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Feature names: {len(churn_model.feature_names)} features")
        
        return True
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        return False


def prepare_prediction_data(form_data):
    """
    Prepare form data untuk prediction
    """
    try:
        # Create prediction data
        pred_data = {
            'Tenure Months': float(form_data.get('tenure', 32)),
            'Monthly Charges': float(form_data.get('monthly_charges', 64.80)),
            'Total Charges': float(form_data.get('total_charges', 2100)),
            'Contract_One year': 1 if form_data.get('contract') == 'One year' else 0,
            'Contract_Two year': 1 if form_data.get('contract') == 'Two year' else 0,
            'Internet Service_Fiber optic': 1 if form_data.get('internet_service') == 'Fiber optic' else 0,
            'Internet Service_No': 1 if form_data.get('internet_service') == 'No' else 0,
            'Tech support_Yes': 1 if form_data.get('tech_support') == 'Yes' else 0,
            'Online security_Yes': 1 if form_data.get('online_security') == 'Yes' else 0,
            'Payment method_Bank transfer (automatic)': 1 if form_data.get('payment_method') == 'Bank transfer (automatic)' else 0,
            'Payment method_Credit card (automatic)': 1 if form_data.get('payment_method') == 'Credit card (automatic)' else 0,
            'Device protection_Yes': 1 if form_data.get('device_protection') == 'Yes' else 0,
            'Online backup_Yes': 1 if form_data.get('online_backup') == 'Yes' else 0,
        }
        
        X_pred = pd.DataFrame([pred_data])
        return X_pred
    except Exception as e:
        print(f"Error preparing prediction data: {str(e)}")
        return None


# ===== ROUTES =====

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page with analytics"""
    # Menampilkan statistik metrik di halaman dashboard Flask Anda jika diperlukan
    return render_template('dashboard.html', stats=model_stats)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    prediction_result = None
    form_data = {}
    
    if request.method == 'POST':
        try:
            # Ambil data dari form HTML
            form_data = request.form.to_dict()
            X_pred = prepare_prediction_data(form_data)
            
            if X_pred is None:
                return jsonify({'error': 'Gagal memproses data input'}), 400
                
            # Lakukan prediksi menggunakan model yang sudah dilatih
            prediction = int(churn_model.predict(X_pred)[0])
            probability = float(churn_model.predict_proba(X_pred)[0][1])
            
            # Simpan hasil prediksi
            prediction_result = {
                'prediction': prediction,
                'probability': probability,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    # Tampilkan halaman form predict dengan data (kosong jika GET, terisi jika POST)
    return render_template('form_prediction.html', 
                         prediction_result=prediction_result,
                         form_data=form_data)


# ===== TRIGER TRAINING SAAT APLIKASI JALAN =====
if __name__ == '__main__':
    print("⏳ Melatih model dan mengirim metrik ke DagsHub...")
    # Ini akan memicu pengiriman nilai f1, precision, recall ke DagsHub
    train_model_with_data() 
    
    # Jalankan server Flask
    app.run(debug=True, port=5000)

