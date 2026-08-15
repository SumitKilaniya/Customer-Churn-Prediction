"""
Customer Churn Prediction & Retention Dashboard
Deploy free on Streamlit Community Cloud: https://streamlit.io/cloud

Local run:
    pip install streamlit pandas scikit-learn plotly joblib
    streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("telco_churn.csv")
    scored = pd.read_csv("scored_customers.csv")
    return df, scored

@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

df, scored = load_data()
model, features = load_model()

st.title("📉 Customer Churn Prediction Dashboard")
st.caption("End-to-end churn risk scoring with business impact estimation")

# ---- KPI row ----
col1, col2, col3, col4 = st.columns(4)
churn_rate = (df["churn"] == "Yes").mean()
revenue_at_risk = df.loc[(df["churn"] == "Yes") & (df["contract"] == "Month-to-month"), "monthly_charges"].sum() * 12
col1.metric("Overall Churn Rate", f"{churn_rate:.1%}")
col2.metric("Revenue at Risk (M2M)", f"${revenue_at_risk:,.0f}/yr")
col3.metric("Customers Scored", f"{len(scored):,}")
col4.metric("Est. Campaign ROI", "10.9x")

st.divider()

# ---- Segment analysis ----
left, right = st.columns(2)
with left:
    st.subheader("Churn Rate by Contract Type")
    contract_churn = df.groupby("contract")["churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
    fig = px.bar(contract_churn, x="contract", y="churn", color="churn",
                 color_continuous_scale=["#3b7a6b", "#c65d3b"], labels={"churn": "Churn Rate (%)"})
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Churn Risk Score Distribution")
    fig2 = px.histogram(scored, x="churn_risk_score", color="churn", nbins=30,
                         color_discrete_map={"Yes": "#c65d3b", "No": "#3b7a6b"})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---- Customer lookup / what-if tool ----
st.subheader("🔍 Score a Customer")
c1, c2, c3 = st.columns(3)
with c1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
with c2:
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
with c3:
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("Predict Churn Risk", type="primary"):
    input_row = pd.DataFrame([{
        "gender": "Male", "senior_citizen": 0, "partner": "No", "tenure_months": tenure,
        "contract": contract, "internet_service": internet_service, "online_security": "No",
        "tech_support": tech_support, "paperless_billing": "Yes", "payment_method": payment_method,
        "monthly_charges": monthly_charges, "total_charges": monthly_charges * tenure,
    }])
    input_encoded = pd.get_dummies(input_row).reindex(columns=features, fill_value=0)
    risk = model.predict_proba(input_encoded)[0][1]
    st.metric("Predicted Churn Probability", f"{risk:.1%}")
    if risk > 0.5:
        st.error("⚠️ High risk — recommend proactive retention outreach")
    elif risk > 0.3:
        st.warning("⚡ Moderate risk — monitor and consider a check-in")
    else:
        st.success("✅ Low risk")

st.divider()
st.subheader("Top At-Risk Customers")
top_risk = scored.sort_values("churn_risk_score", ascending=False).head(20)
st.dataframe(top_risk[["customer_id", "churn_risk_score", "contract", "tenure_months", "monthly_charges"]],
             use_container_width=True)
