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
OUT_DIR   = "D:\\EGOV_DATA\\outputs"
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



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)




# Inverse-frequency weight for the minority class
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


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


train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_cols)

model.fit(train_pool, eval_set=test_pool)



proba = model.predict_proba(X_test)[:, 1]
pred  = model.predict(X_test)

metrics = {
    "roc_auc" :  roc_auc_score(y_test, proba),
    "pr_auc"  :  average_precision_score(y_test, proba),
    "f1"      :  f1_score(y_test, pred),
    "bal_acc" :  balanced_accuracy_score(y_test, pred),
    "conf_mat":  confusion_matrix(y_test, pred).tolist()
}

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))




B = 100
imp_mat = np.zeros((B, X.shape[1]))

for b in range(B):
    boot_idx = np.random.choice(len(X_train), len(X_train), replace=True)
    pool_b   = Pool(X_train.iloc[boot_idx], y_train.iloc[boot_idx],
                    cat_features=cat_cols)
    model_b  = model.copy()
    model_b.fit(pool_b, verbose=False)
    imp_mat[b] = model_b.get_feature_importance(type="FeatureImportance")

imp_df = pd.DataFrame(imp_mat, columns=X.columns)
imp_df.to_csv(os.path.join(OUT_DIR, "bootstrap_importance.csv"), index=False)


# ---------------------- SHAP explanation values -------------------


explainer = shap.TreeExplainer(model)

X_shap = X_test.sample(n=min(1000, len(X_test)), random_state=SEED)
shap_values = explainer.shap_values(X_shap)

# Save mean |SHAP| per feature for reproducibility
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": X_shap.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

shap_df.to_csv(os.path.join(OUT_DIR, "shap_mean_abs.csv"), index=False)




# ---------------------- Bootstrap feature importance -------------------


import pandas as pd, matplotlib.pyplot as plt

data = pd.read_csv("D:\\EGOV_DATA\\outputs\\bootstrap_importance.csv")

mean = data.mean().sort_values(ascending=False)[:10]
sd   = data.std().loc[mean.index]

fig, ax = plt.subplots(figsize=(6,4))

bars = ax.barh(mean.index[::-1], mean[::-1], xerr=sd[::-1])

# Widen the x-axis by a small margin so labels fit
extra_space = sd.max() + 2.9         # tweak 0.02 until it looks right
ax.set_xlim(0, mean.max() + extra_space)

for bar, value, err in zip(bars, mean[::-1], sd[::-1]):
    ax.text(bar.get_width() + err + 0.002,
            bar.get_y() + bar.get_height()/2,
            f"{value:.3f}",
            va="center", ha="left", fontsize=8)

ax.set_xlabel("Mean importance (PredictionValuesChange)")
fig.tight_layout()
plt.savefig("D:\\EGOV_DATA\\outputs\\Figure3_importance.png", dpi=300)


