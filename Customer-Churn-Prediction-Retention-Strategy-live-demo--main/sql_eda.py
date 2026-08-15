import sqlite3
import pandas as pd

df = pd.read_csv('telco_churn.csv')
conn = sqlite3.connect(':memory:')
df.to_sql('customers', conn, index=False, if_exists='replace')

def run(title, query):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(query.strip())
    print('-'*70)
    result = pd.read_sql_query(query, conn)
    print(result.to_string(index=False))
    return result

# Q1: Overall churn rate (baseline every stakeholder will ask first)
run("Q1: Overall churn rate", """
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
""")

# Q2: Churn by contract type -- GROUP BY + conditional aggregation
run("Q2: Churn rate by contract type", """
SELECT
    contract,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC
""")

# Q3: Revenue at risk -- this is the business-framing move DS interviewers look for
run("Q3: Monthly revenue at risk from churned customers, by contract", """
SELECT
    contract,
    SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END) AS monthly_revenue_at_risk,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END) * 12, 0) AS annualized_revenue_at_risk
FROM customers
GROUP BY contract
ORDER BY annualized_revenue_at_risk DESC
""")

# Q4: Tenure bucket analysis -- CASE WHEN for binning, a very commonly asked SQL pattern
run("Q4: Churn rate by tenure bucket", """
SELECT
    CASE
        WHEN tenure_months < 6 THEN '0-6 months'
        WHEN tenure_months < 12 THEN '6-12 months'
        WHEN tenure_months < 24 THEN '1-2 years'
        WHEN tenure_months < 48 THEN '2-4 years'
        ELSE '4+ years'
    END AS tenure_bucket,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY tenure_bucket
ORDER BY MIN(tenure_months)
""")

# Q5: Highest-risk combined segment -- multi-condition WHERE, the kind of query a PM would ask for
run("Q5: Highest-risk segment (month-to-month + no tech support + electronic check)", """
SELECT
    COUNT(*) AS customers_in_segment,
    ROUND(100.0 * SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customers
WHERE contract = 'Month-to-month'
  AND tech_support = 'No'
  AND payment_method = 'Electronic check'
""")

conn.close()
