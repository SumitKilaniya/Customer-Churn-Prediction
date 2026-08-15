import pandas as pd
import joblib

df = pd.read_csv('telco_churn.csv')
rf = joblib.load('churn_model.pkl')
features = joblib.load('model_features.pkl')

df_model = df.drop(columns=['customer_id']).copy()
df_model['churn_flag'] = df_model['churn'].map({'Yes': 1, 'No': 0})
cat_cols = df.drop(columns=['customer_id', 'churn']).select_dtypes(include='object').columns.tolist()
X_full = pd.get_dummies(df_model.drop(columns=['churn', 'churn_flag']), columns=cat_cols, drop_first=True)
X_full = X_full.reindex(columns=features, fill_value=0)

df['churn_risk_score'] = rf.predict_proba(X_full)[:, 1]

# Business scenario: target top 20% highest-risk customers with a retention offer
top20 = df.sort_values('churn_risk_score', ascending=False).head(int(len(df)*0.2))
actual_churners_in_top20 = (top20['churn'] == 'Yes').sum()
revenue_at_risk_top20 = top20.loc[top20['churn']=='Yes', 'monthly_charges'].sum() * 12

# Assume a retention campaign (discount/outreach) costs $15/customer and saves 30% of true churners
campaign_cost = len(top20) * 15
customers_saved = int(actual_churners_in_top20 * 0.30)
revenue_saved = customers_saved * top20.loc[top20['churn']=='Yes', 'monthly_charges'].mean() * 12
roi = (revenue_saved - campaign_cost) / campaign_cost

print(f"Targeting top 20% highest-risk customers ({len(top20)} customers):")
print(f"  Actual churners captured in this group: {actual_churners_in_top20} ({actual_churners_in_top20/len(top20):.1%} of targeted group)")
print(f"  Annualized revenue at risk in this group: ${revenue_at_risk_top20:,.0f}")
print(f"  Campaign cost (${15}/customer outreach): ${campaign_cost:,.0f}")
print(f"  Estimated customers retained (30% save rate): {customers_saved}")
print(f"  Estimated revenue saved: ${revenue_saved:,.0f}")
print(f"  Estimated ROI: {roi:.1f}x")

df[['customer_id','churn_risk_score','contract','tenure_months','monthly_charges','churn']].to_csv('scored_customers.csv', index=False)
print("\nSaved scored_customers.csv with risk scores for every customer")
