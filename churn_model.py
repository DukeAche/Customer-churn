"""
Core ML module for churn prediction using XGBoost and SHAP explainability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap
import pickle
import os


class ChurnPredictor:
    """
    Churn prediction model using XGBoost with SHAP explainability.
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_encoder = LabelEncoder()
        self.shap_explainer = None
        self.shap_values = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def preprocess_data(self, df, target_col='churned', fit=True):
        """
        Preprocess the data: encode categoricals and scale numerics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of the target column
        fit : bool
            Whether to fit encoders/scalers or just transform
        
        Returns:
        --------
        X : np.ndarray
            Processed features
        y : np.ndarray
            Encoded target (if present)
        """
        df = df.copy()
        
        # Separate customer ID if present
        if 'customer_id' in df.columns:
            df = df.drop('customer_id', axis=1)
        
        # Separate target if present
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(target_col, axis=1)
            
            if fit:
                y = self.target_encoder.fit_transform(y)
            else:
                y = self.target_encoder.transform(y)
        else:
            y = None
            X = df
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Encode categorical variables
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle unseen categories
                X[col] = X[col].astype(str)
                known_labels = set(self.label_encoders[col].classes_)
                X[col] = X[col].apply(lambda x: x if x in known_labels else self.label_encoders[col].classes_[0])
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Convert to numpy array
        X_array = X[self.feature_names].values
        
        # Scale numerical features
        if fit:
            X_array = self.scaler.fit_transform(X_array)
        else:
            X_array = self.scaler.transform(X_array)
        
        return X_array, y
    
    def train(self, df, target_col='churned', test_size=0.2, random_state=42):
        """
        Train the XGBoost model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        target_col : str
            Name of the target column
        test_size : float
            Proportion of test set
        random_state : int
            Random seed
        
        Returns:
        --------
        dict
            Training metrics
        """
        # Preprocess data
        X, y = self.preprocess_data(df, target_col, fit=True)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss'
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate
        metrics = self.evaluate()
        
        # Initialize SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.shap_explainer.shap_values(self.X_test)
        
        return metrics
    
    def evaluate(self):
        """
        Evaluate the model on test set.
        
        Returns:
        --------
        dict
            Evaluation metrics
        """
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, df):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        
        Returns:
        --------
        predictions : np.ndarray
            Predicted classes
        probabilities : np.ndarray
            Prediction probabilities
        """
        X, _ = self.preprocess_data(df, fit=False)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Decode predictions
        predictions = self.target_encoder.inverse_transform(predictions)
        
        return predictions, probabilities
    
    def get_shap_values(self, df):
        """
        Get SHAP values for the given data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        
        Returns:
        --------
        shap_values : np.ndarray
            SHAP values
        """
        X, _ = self.preprocess_data(df, fit=False)
        shap_values = self.shap_explainer.shap_values(X)
        return shap_values
    
    def explain_prediction(self, customer_data, customer_index=0):
        """
        Explain why a specific customer is predicted to churn.
        
        Parameters:
        -----------
        customer_data : pd.DataFrame
            Customer data (can be multiple customers)
        customer_index : int
            Index of the customer to explain
        
        Returns:
        --------
        dict
            Explanation with top contributing features
        """
        predictions, probabilities = self.predict(customer_data)
        shap_values = self.get_shap_values(customer_data)
        
        # Get SHAP values for the positive class (churn = Yes)
        customer_shap = shap_values[customer_index]
        
        # Get feature contributions
        feature_contributions = [
            (self.feature_names[i], customer_shap[i])
            for i in range(len(self.feature_names))
        ]
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanation = {
            'customer_index': customer_index,
            'prediction': predictions[customer_index],
            'churn_probability': probabilities[customer_index][1],
            'top_factors': feature_contributions[:5],
            'all_factors': feature_contributions
        }
        
        return explanation
    
    def save_model(self, filepath='churn_model.pkl'):
        """
        Save the trained model and preprocessors.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_encoder': self.target_encoder,
            'shap_explainer': self.shap_explainer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='churn_model.pkl'):
        """
        Load a trained model and preprocessors.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_encoder = model_data['target_encoder']
        self.shap_explainer = model_data['shap_explainer']
        
        print(f"✓ Model loaded from {filepath}")


if __name__ == '__main__':
    # Test the model with dummy data
    print("Testing ChurnPredictor...")
    
    # Load data
    if os.path.exists('sample_churn_data.csv'):
        df = pd.read_csv('sample_churn_data.csv')
        print(f"Loaded data with shape: {df.shape}")
        
        # Initialize and train
        predictor = ChurnPredictor()
        metrics = predictor.train(df)
        
        print("\n=== Model Performance ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
        
        # Save model
        predictor.save_model()
        
    else:
        print("Please run generate_dummy_data.py first to create sample data.")
