"""
Generate a realistic synthetic telecom customer churn dataset.
Why synthetic: no internet access in this environment to pull the real
Kaggle Telco dataset, so we simulate one with realistic relationships
baked in (e.g. month-to-month contracts churn more, tenure reduces churn,
high monthly charges + low tenure = high risk). This mirrors the real
dataset's structure so every technique you learn transfers directly.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20])
tenure = np.random.exponential(scale=20, size=n).astype(int)
tenure = np.clip(tenure, 0, 72)
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.20])
monthly_charges = np.round(np.random.normal(65, 25, n).clip(18, 120), 2)
payment_method = np.random.choice(
    ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n, p=[0.35, 0.2, 0.225, 0.225]
)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52])
tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.51, 0.20])
online_security = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.51, 0.20])
paperless_billing = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
gender = np.random.choice(['Male', 'Female'], n)

total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.9, 1.0, n), 2)

# Build churn probability from realistic business drivers (this is what EDA should uncover)
churn_prob = 0.03
churn_prob += (contract == 'Month-to-month') * 0.16
churn_prob += (contract == 'One year') * 0.03
churn_prob += (internet_service == 'Fiber optic') * 0.07
churn_prob += (payment_method == 'Electronic check') * 0.09
churn_prob += (tech_support == 'No') * 0.06
churn_prob += (online_security == 'No') * 0.05
churn_prob += (tenure < 6) * 0.12
churn_prob += (tenure > 48) * -0.08
churn_prob += (monthly_charges > 80) * 0.05
churn_prob += (senior_citizen == 1) * 0.03
churn_prob += (paperless_billing == 'Yes') * 0.03
churn_prob = np.clip(churn_prob, 0.02, 0.85)

churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    'customer_id': [f'CUST{i:05d}' for i in range(n)],
    'gender': gender,
    'senior_citizen': senior_citizen,
    'partner': partner,
    'tenure_months': tenure,
    'contract': contract,
    'internet_service': internet_service,
    'online_security': online_security,
    'tech_support': tech_support,
    'paperless_billing': paperless_billing,
    'payment_method': payment_method,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'churn': np.where(churn == 1, 'Yes', 'No')
})

df.to_csv('telco_churn.csv', index=False)
print(f"Generated {len(df)} rows")
print(f"Churn rate: {(df['churn']=='Yes').mean():.1%}")
print(df.head())
