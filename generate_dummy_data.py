"""
Generate a dummy churn dataset for testing the Churn-Buster dashboard.
Creates realistic synthetic data with typical churn indicators.
"""

import pandas as pd
import numpy as np

def generate_churn_dataset(n_samples=1000, random_state=42):
    """
    Generate a synthetic customer churn dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Synthetic churn dataset
    """
    np.random.seed(random_state)
    
    # Generate features
    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples, p=[0.5, 0.3, 0.2]),
        'login_count_last_month': np.random.randint(0, 60, n_samples),
        'support_tickets': np.random.randint(0, 15, n_samples),
        'days_since_last_login': np.random.randint(0, 90, n_samples),
        'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check', 'Cash'], n_samples, p=[0.35, 0.25, 0.3, 0.1]),
        'internet_service': np.random.choice(['DSL', 'Fiber Optic', 'No'], n_samples, p=[0.35, 0.4, 0.25]),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.45, 0.2]),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate total charges based on tenure and monthly charges
    df['total_charges'] = df['tenure_months'] * df['monthly_charges'] * np.random.uniform(0.95, 1.05, n_samples)
    
    # Generate churn label with realistic patterns
    # Higher churn probability for:
    # - Short tenure
    # - High support tickets
    # - Month-to-month contracts
    # - Low login activity
    # - Long time since last login
    
    churn_probability = np.zeros(n_samples)
    
    # Base probability
    churn_probability += 0.1
    
    # Tenure effect (shorter tenure = higher churn)
    churn_probability += (72 - df['tenure_months']) / 144
    
    # Contract type effect
    churn_probability += (df['contract_type'] == 'Month-to-Month').astype(float) * 0.3
    churn_probability -= (df['contract_type'] == 'Two Year').astype(float) * 0.2
    
    # Support tickets effect (more tickets = higher churn)
    churn_probability += (df['support_tickets'] / 30)
    
    # Login activity effect (less activity = higher churn)
    churn_probability += (60 - df['login_count_last_month']) / 120
    churn_probability += df['days_since_last_login'] / 180
    
    # Payment method effect (electronic check has higher churn)
    churn_probability += (df['payment_method'] == 'Electronic Check').astype(float) * 0.15
    
    # No services effect
    churn_probability += (df['online_security'] == 'No').astype(float) * 0.1
    churn_probability += (df['tech_support'] == 'No').astype(float) * 0.1
    
    # Senior citizen effect
    churn_probability += df['senior_citizen'] * 0.1
    
    # Clip probabilities to [0, 1]
    churn_probability = np.clip(churn_probability, 0, 1)
    
    # Generate binary churn outcome
    df['churned'] = (np.random.random(n_samples) < churn_probability).astype(int)
    df['churned'] = df['churned'].map({0: 'No', 1: 'Yes'})
    
    return df


if __name__ == '__main__':
    # Generate dataset
    print("Generating dummy churn dataset...")
    df = generate_churn_dataset(n_samples=1200)
    
    # Display statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"\nChurn distribution:")
    print(df['churned'].value_counts())
    print(f"\nChurn rate: {(df['churned'] == 'Yes').sum() / len(df) * 100:.2f}%")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nDataset info:")
    print(df.info())
    
    # Save to CSV
    output_file = 'sample_churn_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Dataset saved to {output_file}")
