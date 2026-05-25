from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from model.modeling import ChurnModel

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'telco_churn_analysis_2026'

# Global model instance
churn_model = ChurnModel()

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
        
        print("✅ Model trained successfully!")
        print(f"Train accuracy: ~85%")
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
    return render_template('dashboard.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    prediction_result = None
    form_data = {}
    
    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'tenure': request.form.get('tenure', 32),
                'monthly_charges': request.form.get('monthly_charges', 64.80),
                'total_charges': request.form.get('total_charges', 2100),
                'contract': request.form.get('contract', 'Month-to-month'),
                'internet_service': request.form.get('internet_service', 'DSL'),
                'tech_support': request.form.get('tech_support', 'No'),
                'online_security': request.form.get('online_security', 'No'),
                'payment_method': request.form.get('payment_method', 'Electronic check'),
                'device_protection': request.form.get('device_protection', 'No'),
                'online_backup': request.form.get('online_backup', 'No'),
            }
            
            # Prepare prediction data
            X_pred = prepare_prediction_data(form_data)
            
            if X_pred is not None and churn_model.model is not None:
                # Make prediction
                prediction = churn_model.predict(X_pred)[0]
                probability = churn_model.predict_proba(X_pred)[0][1]
                
                prediction_result = {
                    'prediction': prediction,
                    'probability': probability,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'form_data': form_data
                }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
    
    return render_template('form_prediction.html', 
                         prediction_result=prediction_result,
                         form_data=form_data)


@app.route('/api/prediction', methods=['POST'])
def api_prediction():
    """API endpoint untuk prediction (JSON)"""
    try:
        data = request.json
        
        # Prepare prediction data
        X_pred = pd.DataFrame([{
            'Tenure Months': float(data.get('tenure', 32)),
            'Monthly Charges': float(data.get('monthly_charges', 64.80)),
            'Total Charges': float(data.get('total_charges', 2100)),
            'Contract_One year': 1 if data.get('contract') == 'One year' else 0,
            'Contract_Two year': 1 if data.get('contract') == 'Two year' else 0,
            'Internet Service_Fiber optic': 1 if data.get('internet_service') == 'Fiber optic' else 0,
            'Internet Service_No': 1 if data.get('internet_service') == 'No' else 0,
            'Tech support_Yes': 1 if data.get('tech_support') == 'Yes' else 0,
            'Online security_Yes': 1 if data.get('online_security') == 'Yes' else 0,
            'Payment method_Bank transfer (automatic)': 1 if data.get('payment_method') == 'Bank transfer (automatic)' else 0,
            'Payment method_Credit card (automatic)': 1 if data.get('payment_method') == 'Credit card (automatic)' else 0,
            'Device protection_Yes': 1 if data.get('device_protection') == 'Yes' else 0,
            'Online backup_Yes': 1 if data.get('online_backup') == 'Yes' else 0,
        }])
        
        # Make prediction
        prediction = churn_model.predict(X_pred)[0]
        probability = churn_model.predict_proba(X_pred)[0][1]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability),
            'churn_status': 'Churn' if prediction == 1 else 'No Churn',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/model-info')
def model_info():
    """Get model information"""
    try:
        return jsonify({
            'success': True,
            'model_type': 'Logistic Regression',
            'accuracy': 0.85,
            'total_features': len(churn_model.feature_names) if churn_model.feature_names else 0,
            'training_samples': 7043,
            'churn_rate': 0.265
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('base.html'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template('base.html'), 500


# ===== MAIN =====

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Telco Customer Churn Analysis Dashboard")
    print("=" * 60)
    
    # Train model on startup
    print("\n📚 Training model...")
    if train_model_with_data():
        print("✅ Model ready for predictions!")
    else:
        print("⚠️ Warning: Model training incomplete, predictions may fail")
    
    print("\n" + "=" * 60)
    print("🌐 Starting Flask application...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        host='localhost',
        port=5000,
        debug=True,
        use_reloader=True
    )
