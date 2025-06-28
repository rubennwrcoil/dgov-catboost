# -*- coding: utf-8 -*-
'''
Update path and file names
'''


import os, json, random          
import numpy as np               
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import shap

# Step 1: paths & random seed
DATA_PATH = "D:\\EGOV_DATA\\for_catboolst.xlsx"
OUT_DIR   = "D:\\EGOV_DATA\\outputs_fullSample"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 2025                         
random.seed(SEED)
np.random.seed(SEED)



# Step 2: Load the data and split the data into features and target variable
df = pd.read_excel(DATA_PATH)

df.fillna(value="missing", inplace=True)

target = "MachshevGov"              
dropcols = ["SerialNumber"]         
X = df.drop(columns=[target] + dropcols)
y = df[target]


cat_cols = ["Religion", "Religiosity", "AreaLiving", "Education", "Hebrew_prof", "Gender", "Age", "InternetChannel", "Occupation", "YearSurvey"]  




# Inverse-frequency weight for the minority class
pos_weight = (y == 0).sum() / (y == 1).sum()


# Initiate CatBoost
model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=[1, pos_weight],       
    early_stopping_rounds=50,
    verbose=200
)


train_pool = Pool(X, y, cat_features=cat_cols)
test_pool  = Pool(X,  y,  cat_features=cat_cols)

model.fit(train_pool, eval_set=test_pool)



proba = model.predict_proba(X)[:, 1]
pred  = model.predict(X)

metrics = {
    "roc_auc" :  roc_auc_score(y, proba),
    "pr_auc"  :  average_precision_score(y, proba),
    "f1"      :  f1_score(y, pred),
    "bal_acc" :  balanced_accuracy_score(y, pred),
    "conf_mat":  confusion_matrix(y, pred).tolist()
}

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))




# ---------------------- SHAP explanation values -------------------


explainer = shap.TreeExplainer(model)

X_shap = X.sample(n=min(1000, len(X)), random_state=SEED)
shap_values = explainer.shap_values(X_shap)

# Save mean |SHAP| per feature for reproducibility
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": X_shap.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

shap_df.to_csv(os.path.join(OUT_DIR, "shap_mean_abs.csv"), index=False)






