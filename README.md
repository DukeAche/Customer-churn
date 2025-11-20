# 🎯 Churn-Buster Dashboard

A powerful, AI-driven customer churn prediction dashboard that not only predicts which customers are likely to leave, but also explains **why** using SHAP (SHapley Additive exPlanations). Built for data scientists and business analysts who need actionable insights.

## 🌟 Features

### 📊 Data Ingestion
- Upload customer data via CSV or Excel files
- Automatic data validation and preprocessing
- Sample dataset included for quick demo

### 🤖 Predictive Modeling
- **XGBoost Classifier**: State-of-the-art gradient boosting for accurate churn prediction
- Automated feature engineering and scaling
- Train-test split with comprehensive evaluation metrics
- Model persistence for reuse

### 🔍 Explainable AI
- **SHAP Integration**: Understand model decisions at global and local levels
- **Global Insights**: Feature importance rankings and summary plots
- **Individual Analysis**: Waterfall plots showing why specific customers are at risk
- **Risk Factors**: Top contributing features for each high-risk customer

### 📈 Interactive Dashboard
- **Multi-page Streamlit Interface**: Intuitive navigation across different views
- **Real-time Predictions**: Upload new data and get instant churn predictions
- **Risk Segmentation**: Categorize customers into Low/Medium/High risk
- **Export Results**: Download predictions as CSV for further analysis

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /home/dukeray66/anti_3
   ```
   **Create a virtual environment and activate it**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
    pip install -r requirements.txt
   ```

### Generate Sample Data

Create a synthetic churn dataset for testing:

```bash
python generate_dummy_data.py
```

This creates `sample_churn_data.csv` with 1200 customers and realistic churn patterns.

### Run the Dashboard

Launch the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📖 User Guide

### 1. Data Upload & Training Page

- **Upload Data**: Click "Browse files" to upload your customer data (CSV/Excel)
  - Required column: `churned` (Yes/No)
  - Include customer features: tenure, charges, login activity, etc.
- **Use Sample Data**: Check the box to load the demo dataset
- **Train Model**: Click "Train XGBoost Model" to build the predictor
- **View Metrics**: See accuracy, precision, recall, F1, and ROC AUC scores

### 2. Model Insights Page

- **SHAP Summary Plot**: See which features matter most globally
  - Red dots = high feature values
  - Blue dots = low feature values
  - X-axis = impact on prediction
- **Feature Importance**: Bar chart ranking features by average impact
- **Performance Metrics**: Quick reference for model quality

### 3. Customer Analysis Page

- **Select Customer**: Choose from dropdown or by index
- **Filter Options**: View all customers, or filter by risk level
- **Risk Assessment**: See churn probability and risk level (Low/Medium/High)
- **Top Risk Factors**: Understand the top 5 drivers for this customer
- **Waterfall Plot**: Visual breakdown of how features contribute to the prediction

### 4. Batch Predictions Page

- **Upload New Data**: Provide a file with customers to score
- **Generate Predictions**: Get churn predictions for all customers
- **Risk Segmentation**: See count of High/Medium/Low risk customers
- **Download Results**: Export predictions with probabilities and risk levels

## 📊 Data Format

Your input CSV/Excel should have columns like:

| Column | Description | Type |
|--------|-------------|------|
| `customer_id` | Unique identifier (optional) | String |
| `tenure_months` | Months as customer | Integer |
| `monthly_charges` | Monthly bill amount | Float |
| `contract_type` | Month-to-Month, One Year, Two Year | Categorical |
| `login_count_last_month` | Login frequency | Integer |
| `support_tickets` | Number of support requests | Integer |
| `days_since_last_login` | Recency of activity | Integer |
| `churned` | Yes/No (training data only) | Categorical |

See `sample_churn_data.csv` for a complete example.

## 🧠 How It Works

### XGBoost Model
- **Algorithm**: Gradient boosted decision trees
- **Hyperparameters**: 100 estimators, max depth 6, learning rate 0.1
- **Preprocessing**: Label encoding for categoricals, standard scaling for numerics
- **Evaluation**: 80/20 train-test split with stratification

### SHAP Explainability
- **TreeExplainer**: Optimized for tree-based models like XGBoost
- **SHAP Values**: Measure each feature's contribution to individual predictions
- **Summary Plot**: Global view of feature importance and effects
- **Waterfall Plot**: Local explanation for individual customers

## 🎓 Understanding SHAP Values

**SHAP (SHapley Additive exPlanations)** assigns each feature an importance value for a specific prediction:

- **Positive SHAP value**: Feature increases churn probability
- **Negative SHAP value**: Feature decreases churn probability
- **Magnitude**: How much the feature affects the prediction

### Example Interpretation

If a customer has a SHAP waterfall plot showing:
- `days_since_last_login`: +0.8 (high value, pushes toward churn)
- `tenure_months`: -0.5 (long tenure, pushes away from churn)
- `support_tickets`: +0.3 (many tickets, pushes toward churn)

**Interpretation**: This customer is at risk primarily because they haven't logged in recently, but their long tenure helps retention. Reducing support issues could help.

## 🔧 Customization

### Adding Features
Edit `generate_dummy_data.py` to include domain-specific features for your business.

### Model Tuning
Modify hyperparameters in `churn_model.py`:
```python
self.model = xgb.XGBClassifier(
    n_estimators=150,  # Increase for more trees
    max_depth=8,       # Increase for more complexity
    learning_rate=0.05 # Decrease for more conservative learning
)
```

### Custom Styling
Update the CSS in `app.py` to match your brand colors.

## 📦 Project Structure

```
anti_3/
├── app.py                      # Main Streamlit dashboard
├── churn_model.py              # XGBoost model and SHAP logic
├── generate_dummy_data.py      # Sample data generator
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── sample_churn_data.csv       # Generated sample data
```

## 🤝 Contributing

Feel free to extend this dashboard:
- Add Random Forest as an alternative model
- Implement AutoML for hyperparameter tuning
- Add time-series analysis for churn trends
- Create retention campaign suggestions

## 📄 License

This project is open-source. Use it for learning, business, or research!

## 🙋 Support

For questions or issues:
1. Check the sample data format
2. Ensure all dependencies are installed
3. Verify your data has a `churned` column for training

---

**Built with ❤️ using XGBoost, SHAP, and Streamlit**
