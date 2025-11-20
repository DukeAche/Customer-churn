"""
Churn-Buster: Interactive Churn Prediction Dashboard
Built with Streamlit, XGBoost, and SHAP for explainable AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from churn_model import ChurnPredictor
import os

# Page configuration
st.set_page_config(
    page_title="Churn-Buster Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = ChurnPredictor()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None


def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Churn-Buster Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Predict customer churn with AI-powered explainability**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.title("Navigation")
        page = st.radio(
            "Select a page:",
            ["📊 Data Upload & Training", "🔍 Model Insights", "👤 Customer Analysis", "📈 Batch Predictions"]
        )
        
        st.markdown("---")
        st.info("**About Churn-Buster**\n\nThis dashboard uses XGBoost and SHAP to predict customer churn and explain why customers are at risk.")
    
    # Route to different pages
    if page == "📊 Data Upload & Training":
        data_upload_page()
    elif page == "🔍 Model Insights":
        model_insights_page()
    elif page == "👤 Customer Analysis":
        customer_analysis_page()
    elif page == "📈 Batch Predictions":
        batch_predictions_page()


def data_upload_page():
    """Page for uploading data and training the model."""
    st.header("📊 Data Upload & Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Step 1: Upload Your Data")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file with customer data",
            type=['csv', 'xlsx', 'xls'],
            help="Your file should contain customer features and a 'churned' column (Yes/No)"
        )
        
        # Option to use sample data
        use_sample = st.checkbox("Or use sample dataset for demo", value=False)
        
        if use_sample and os.path.exists('sample_churn_data.csv'):
            st.session_state.data = pd.read_csv('sample_churn_data.csv')
            st.success("✓ Sample data loaded successfully!")
        elif uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                else:
                    st.session_state.data = pd.read_excel(uploaded_file)
                st.success(f"✓ File uploaded successfully! Shape: {st.session_state.data.shape}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        # Display data preview
        if st.session_state.data is not None:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
            
            # Data statistics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Customers", len(st.session_state.data))
            with col_b:
                churn_count = (st.session_state.data['churned'] == 'Yes').sum()
                st.metric("Churned Customers", churn_count)
            with col_c:
                churn_rate = churn_count / len(st.session_state.data) * 100
                st.metric("Churn Rate", f"{churn_rate:.2f}%")
    
    with col2:
        st.subheader("Step 2: Train Model")
        
        if st.session_state.data is not None:
            if st.button("🚀 Train XGBoost Model", use_container_width=True):
                with st.spinner("Training model... This may take a moment."):
                    try:
                        # Train the model
                        metrics = st.session_state.predictor.train(st.session_state.data)
                        st.session_state.metrics = metrics
                        st.session_state.model_trained = True
                        
                        st.success("✓ Model trained successfully!")
                        
                        # Display metrics
                        st.markdown("### Model Performance")
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                        st.metric("Precision", f"{metrics['precision']:.2%}")
                        st.metric("Recall", f"{metrics['recall']:.2%}")
                        st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
                        st.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")
                        
                    except Exception as e:
                        st.error(f"Error training model: {e}")
        else:
            st.warning("Please upload data first!")
    
    # Show training status
    if st.session_state.model_trained:
        st.success("✅ Model is trained and ready to use!")
        st.info("Navigate to other pages to explore insights and make predictions.")


def model_insights_page():
    """Page for displaying global model insights using SHAP."""
    st.header("🔍 Model Insights & Feature Importance")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first on the 'Data Upload & Training' page.")
        return
    
    st.markdown("### Global Feature Importance")
    st.write("This SHAP summary plot shows which features are most important for predicting churn across all customers.")
    
    # Generate SHAP summary plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            st.session_state.predictor.shap_values,
            st.session_state.predictor.X_test,
            feature_names=st.session_state.predictor.feature_names,
            show=False
        )
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        **How to read this plot:**
        - Features are ranked by importance (top to bottom)
        - Each dot represents a customer
        - Color indicates feature value (red = high, blue = low)
        - Position on X-axis shows impact on churn prediction
        """)
        
    except Exception as e:
        st.error(f"Error generating SHAP plot: {e}")
    
    # Feature importance bar chart
    st.markdown("### Feature Importance Ranking")
    
    try:
        # Calculate mean absolute SHAP values
        feature_importance = pd.DataFrame({
            'feature': st.session_state.predictor.feature_names,
            'importance': np.abs(st.session_state.predictor.shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_importance['feature'][:10], feature_importance['importance'][:10], color='#667eea')
        ax.set_xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontsize=12)
        ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"Error generating feature importance chart: {e}")
    
    # Model performance metrics
    if st.session_state.metrics:
        st.markdown("### Model Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{st.session_state.metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{st.session_state.metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{st.session_state.metrics['recall']:.2%}")
        with col4:
            st.metric("F1 Score", f"{st.session_state.metrics['f1_score']:.2%}")
        with col5:
            st.metric("ROC AUC", f"{st.session_state.metrics['roc_auc']:.2%}")


def customer_analysis_page():
    """Page for analyzing individual customer churn risk."""
    st.header("👤 Individual Customer Analysis")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first on the 'Data Upload & Training' page.")
        return
    
    st.markdown("### Analyze Why Specific Customers Are at Risk")
    
    # Customer selection
    if st.session_state.data is not None:
        # Filter for customers
        df = st.session_state.data.copy()
        
        # Option to filter by churn status
        filter_option = st.selectbox(
            "Filter customers by:",
            ["All Customers", "High Risk (Predicted Churn)", "Low Risk (Predicted Retention)"]
        )
        
        # Get predictions for filtering
        predictions, probabilities = st.session_state.predictor.predict(df.drop('churned', axis=1, errors='ignore'))
        df['predicted_churn'] = predictions
        df['churn_probability'] = probabilities[:, 1]
        
        if filter_option == "High Risk (Predicted Churn)":
            df = df[df['predicted_churn'] == 'Yes']
        elif filter_option == "Low Risk (Predicted Retention)":
            df = df[df['predicted_churn'] == 'No']
        
        # Customer selector
        if 'customer_id' in df.columns:
            customer_ids = df['customer_id'].tolist()
            selected_customer_id = st.selectbox("Select a customer:", customer_ids)
            customer_idx = df[df['customer_id'] == selected_customer_id].index[0]
            customer_data = df.loc[[customer_idx]]
        else:
            customer_idx = st.number_input("Select customer index:", min_value=0, max_value=len(df)-1, value=0)
            customer_data = df.iloc[[customer_idx]]
        
        # Display customer details
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Customer Details")
            st.dataframe(customer_data.T, use_container_width=True)
        
        with col2:
            st.subheader("Churn Risk Assessment")
            
            # Get prediction and explanation
            explanation = st.session_state.predictor.explain_prediction(
                customer_data.drop(['churned', 'predicted_churn', 'churn_probability'], axis=1, errors='ignore'),
                customer_index=0
            )
            
            # Display prediction
            churn_prob = explanation['churn_probability']
            
            # Risk level
            if churn_prob > 0.7:
                risk_level = "🔴 HIGH RISK"
                risk_color = "red"
            elif churn_prob > 0.4:
                risk_level = "🟡 MEDIUM RISK"
                risk_color = "orange"
            else:
                risk_level = "🟢 LOW RISK"
                risk_color = "green"
            
            st.markdown(f"### {risk_level}")
            st.metric("Churn Probability", f"{churn_prob:.1%}")
            
            # Top contributing factors
            st.markdown("### 🎯 Top Risk Factors")
            st.write("These factors contribute most to this customer's churn risk:")
            
            for i, (feature, contribution) in enumerate(explanation['top_factors'][:5], 1):
                direction = "↑ Increases" if contribution > 0 else "↓ Decreases"
                st.markdown(f"**{i}. {feature}**: {direction} churn risk (SHAP: {contribution:.4f})")
        
        # SHAP waterfall plot for individual explanation
        st.markdown("### 📊 SHAP Waterfall Plot")
        st.write("This plot shows how each feature pushes the prediction higher or lower.")
        
        try:
            # Get SHAP values for this customer
            customer_shap = st.session_state.predictor.get_shap_values(
                customer_data.drop(['churned', 'predicted_churn', 'churn_probability'], axis=1, errors='ignore')
            )[0]
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get base value (expected value)
            base_value = st.session_state.predictor.shap_explainer.expected_value
            
            # Use SHAP's waterfall plot
            shap.plots._waterfall.waterfall_legacy(
                base_value,
                customer_shap,
                feature_names=st.session_state.predictor.feature_names,
                show=False
            )
            
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Error generating waterfall plot: {e}")


def batch_predictions_page():
    """Page for making predictions on new data."""
    st.header("📈 Batch Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first on the 'Data Upload & Training' page.")
        return
    
    st.markdown("### Upload New Customer Data for Predictions")
    
    # File uploader for new data
    new_file = st.file_uploader(
        "Upload CSV or Excel file with new customer data",
        type=['csv', 'xlsx', 'xls'],
        key='new_data'
    )
    
    if new_file is not None:
        try:
            if new_file.name.endswith('.csv'):
                new_data = pd.read_csv(new_file)
            else:
                new_data = pd.read_excel(new_file)
            
            st.success(f"✓ File uploaded! Contains {len(new_data)} customers.")
            
            # Make predictions
            if st.button("🔮 Generate Predictions"):
                with st.spinner("Making predictions..."):
                    predictions, probabilities = st.session_state.predictor.predict(new_data)
                    
                    # Add predictions to dataframe
                    results = new_data.copy()
                    results['predicted_churn'] = predictions
                    results['churn_probability'] = probabilities[:, 1]
                    results['risk_level'] = pd.cut(
                        probabilities[:, 1],
                        bins=[0, 0.4, 0.7, 1.0],
                        labels=['Low Risk', 'Medium Risk', 'High Risk']
                    )
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(results, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        high_risk = (results['risk_level'] == 'High Risk').sum()
                        st.metric("High Risk Customers", high_risk)
                    with col2:
                        medium_risk = (results['risk_level'] == 'Medium Risk').sum()
                        st.metric("Medium Risk Customers", medium_risk)
                    with col3:
                        low_risk = (results['risk_level'] == 'Low Risk').sum()
                        st.metric("Low Risk Customers", low_risk)
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Upload a file to get started with batch predictions.")


if __name__ == '__main__':
    main()
