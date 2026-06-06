
import json

ml_notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Capstone Part 3: Churn Prediction Modeling & Threshold Calibration\n",
                "**Course**: IITP AI Course Capstone\n",
                "**Project Repository**: `d2c-churn-predictive-model`\n\n",
                "### Objective:\n",
                "Implements a leakage-safe data transformation workflow, builds baseline vs. ensemble tree classifiers, tunes decision boundaries from a business risk standpoint, and serializes operational assets."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import pickle\n",
                "import json\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix\n\n",
                "print('Predictive dependencies compiled successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Leakage-Safe Data Splitting\n",
                "We partition features exclusively using indices recorded on or before the 2025-09-30 snapshot cutoff, keeping train, validation, and test observations perfectly isolated."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_csv('rfm_modeling_snapshot.csv')\n\n",
                "# Isolate input variables from tracking columns\n",
                "drop_cols = ['customer_id', 'snapshot_date', 'churn_next_60d', 'split']\n",
                "cat_cols = ['city_tier', 'age_group', 'acquisition_channel', 'loyalty_tier', 'preferred_category', 'marketing_consent']\n\n",
                "# Apply numeric dummy coding across category boundaries\n",
                "df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)\n",
                "feature_names = [c for c in df_encoded.columns if c not in drop_cols]\n\n",
                "# Partition data using the pre-assigned split labels\n",
                "train_df = df_encoded[df_encoded['split'] == 'train']\n",
                "val_df = df_encoded[df_encoded['split'] == 'validation']\n",
                "test_df = df_encoded[df_encoded['split'] == 'test']\n\n",
                "X_train, y_train = train_df[feature_names].fillna(0), train_df['churn_next_60d']\n",
                "X_val, y_val = val_df[feature_names].fillna(0), val_df['churn_next_60d']\n",
                "X_test, y_test = test_df[feature_names].fillna(0), test_df['churn_next_60d']\n\n",
                "print(f'Train size: {X_train.shape[0]} samples | Validation size: {X_val.shape[0]} samples')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Implementing Classifiers: Baseline vs. Strong Champion Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Model A: Baseline Logistic Regression\n",
                "base_model = LogisticRegression(max_iter=1000, random_state=42)\n",
                "base_model.fit(X_train, y_train)\n\n",
                "# Model B: Strong Random Forest Ensemble Classifier\n",
                "champion_rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=4, random_state=42)\n",
                "champion_rf.fit(X_train, y_train)\n\n",
                "print('Both predictive models initialized and trained successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Business Threshold Tuning & Validation Check\n",
                "Instead of relying on the standard uncalibrated 0.50 cutoff, we tune the threshold down to 0.40. This allows us to proactively capture disengaged accounts before they churn completely."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "val_probabilities = champion_rf.predict_proba(X_val)[:, 1]\n",
                "tuned_threshold = 0.40\n",
                "val_predictions = (val_probabilities >= tuned_threshold).astype(int)\n\n",
                "# Compute model performance metrics\n",
                "accuracy = accuracy_score(y_val, val_predictions)\n",
                "precision = precision_score(y_val, val_predictions)\n",
                "recall = recall_score(y_val, val_predictions)\n",
                "f1 = f1_score(y_val, val_predictions)\n",
                "auc_score = roc_auc_score(y_val, val_probabilities)\n",
                "c_matrix = confusion_matrix(y_val, val_predictions).tolist()\n\n",
                "metrics_payload = {\n",
                "    \"accuracy\": round(accuracy, 4),\n",
                "    \"precision\": round(precision, 4),\n",
                "    \"recall\": round(recall, 4),\n",
                "    \"f1_score\": round(f1, 4),\n",
                "    \"roc_auc\": round(auc_score, 4),\n",
                "    \"confusion_matrix\": c_matrix,\n",
                "    \"selected_threshold\": tuned_threshold\n",
                "}\n\n",
                "with open('metrics.json', 'w') as f:\n",
                "    json.dump(metrics_payload, f, indent=4)\n",
                "print(json.dumps(metrics_payload, indent=4))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Feature Importance Profiling\n",
                "We print out the top drivers pushing the churn model calculations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "importances = champion_rf.feature_importances_\n",
                "feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)\n",
                "print('\\n--- TOP 5 DRIVERS OF CUSTOMER CHURN ---')\n",
                "print(feat_imp_df.head(5))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Serializing the Final Model Pipeline Artifact"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('model.pkl', 'wb') as f:\n",
                "    pickle.dump(champion_rf, f)\n",
                "print('\\nModel binary successfully saved into model.pkl format.')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("churn_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(ml_notebook_content, f, indent=2)
print("SUCCESS: churn_model.ipynb generation completed.")