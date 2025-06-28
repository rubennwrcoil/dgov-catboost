# CatBoost models for “Digital Government Use among the Arab Population in Israel”

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15762356.svg)](https://doi.org/10.5281/zenodo.15762356)

This repository contains the training script, pre-trained models, and output
files that support the machine-learning results in our revised
submission to "Online Information Review".

---

## Folder structure
dgov-catboost/

├ script/

  │	├ catboost_full.py # trains on the entire dataset

  │	├ catboost_test.py # 70/30 split with class-weights, bootstraps
  
├ data/

├ outputs_full/ # artefacts for full-sample run

  │	├ metrics_full.json

  │	├ shap_mean_abs_full.csv

├ outputs_test/ # artefacts for hold-out run

  │	├ metrics_test.json

  │	├ shap_mean_abs_test.csv

  │	├ bootstrap_importance_test.csv

├ requirements.txt 

├LICENSE 

├ README.md # you are here




## Note
The CBS Social Survey microdata are under restricted access and
cannot be redistributed here. See data/README_data.txt for instructions.


## How to cite this repository
Harari, R. (2025). The CatBoost training script, outputs, and environment file for the article "Digital Government Use among the Arab Population in Israel" (V2025-06-28). Zenodo. https://doi.org/10.5281/zenodo.15762356

## License 
This project is released under the MIT License (see LICENSE). You are
free to reuse or adapt the code provided you retain attribution.

