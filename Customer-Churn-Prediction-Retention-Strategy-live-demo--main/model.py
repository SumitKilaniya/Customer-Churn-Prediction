"""
Model training with proper evaluation.
Concepts taught inline via prints -- read the comments, this is exam material.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                              precision_score, recall_score, f1_score)
import joblib

df = pd.read_csv('telco_churn.csv')

# ---- Feature engineering ----
# WHY: raw categorical strings can't go into most models directly.
# We use one-hot encoding for nominal categories (no natural order) and
# keep numeric columns as-is. This is the standard approach for
# interpretable models like logistic regression / random forest.
df_model = df.drop(columns=['customer_id'])
df_model['churn'] = df_model['churn'].map({'Yes': 1, 'No': 0})

categorical_cols = df_model.select_dtypes(include='object').columns.tolist()
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(columns=['churn'])
y = df_encoded['churn']

# ---- Train/test split ----
# WHY stratify=y: churn is imbalanced (30.7% / 69.3%). Without stratify,
# a random split could accidentally give train/test different churn rates,
# skewing evaluation. stratify keeps the class ratio identical in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train churn rate: {y_train.mean():.1%}, Test churn rate: {y_test.mean():.1%}")

# ---- Model 1: Logistic Regression (baseline, interpretable) ----
# WHY start here: DS roles value interpretability. Logistic regression
# coefficients directly tell you "this feature increases/decreases churn
# odds by X%" -- exactly what a business stakeholder wants to hear.
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

# ---- Model 2: Random Forest (usually higher accuracy, less interpretable) ----
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}  <- of predicted churners, how many actually churned")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}  <- of actual churners, how many we caught")
    print(f"F1 score:  {f1_score(y_true, y_pred):.3f}  <- harmonic mean of precision & recall")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.3f}  <- ranking quality, threshold-independent")
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n[[TN={cm[0,0]}  FP={cm[0,1]}]\n [FN={cm[1,0]}  TP={cm[1,1]}]]")

evaluate("Logistic Regression", y_test, y_pred_lr, y_proba_lr)
evaluate("Random Forest", y_test, y_pred_rf, y_proba_rf)

# ---- Feature importance (business interpretation) ----
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\n{'='*60}\nTop 10 churn drivers (Random Forest feature importance)\n{'='*60}")
print(importances.head(10).to_string())

coefs = pd.Series(log_reg.coef_[0], index=X.columns).sort_values(ascending=False)
print(f"\n{'='*60}\nTop 5 factors INCREASING churn odds (Logistic Regression)\n{'='*60}")
print(coefs.head(5).to_string())
print(f"\nTop 5 factors DECREASING churn odds\n{'-'*60}")
print(coefs.tail(5).to_string())

# Save artifacts for the dashboard / deployment
joblib.dump(rf, 'churn_model.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')
importances.to_csv('feature_importance.csv')
print("\nSaved: churn_model.pkl, model_features.pkl, feature_importance.csv")
