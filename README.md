# 🧠 Parkinson’s Disease Prediction using Machine Learning
📋 Overview

This project aims to predict whether an individual has Parkinson’s disease based on biomedical voice measurements.
Using a dataset from Kaggle – Parkinson’s Disease Dataset
, several machine learning algorithms were trained, evaluated, and compared to find the best-performing model.

🎯 Objectives

Perform data preprocessing, feature analysis, and visualization.

Train and compare multiple classification algorithms.

Identify the most effective model for accurate Parkinson’s disease prediction.

Evaluate performance using various metrics such as accuracy, F1-score, ROC-AUC, and PR-AUC.

🧾 Dataset

Source: Kaggle — Parkinson’s Disease Dataset

Description: The dataset contains various biomedical voice measurements from individuals, some diagnosed with Parkinson’s disease and others not.

Target Variable: status (1 = Parkinson’s, 0 = Healthy)

⚙️ Tools & Libraries

This project was implemented in Python using Google Colab.
Key libraries include:

numpy, pandas, matplotlib, seaborn, scipy
sklearn (scikit-learn)
catboost
xgboost

🔍 Machine Learning Models

The following models were trained and evaluated:

Model	Accuracy	Precision	Recall	Specificity	F1	ROC AUC	PR AUC
Logistic Regression	0.7943	0.8272	0.8481	0.7046	0.8375	0.8911	0.9324
Random Forest	0.9177	0.9409	0.9266	0.9030	0.9337	0.9501	0.9554
SVM	0.7864	0.7995	0.8785	0.6329	0.8372	0.8672	0.9151
KNN	0.6456	0.6887	0.7899	0.4051	0.7358	0.6320	0.7044
Decision Tree	0.8544	0.9062	0.8557	0.8523	0.8802	0.8540	0.8656
CatBoost	0.9272	0.9373	0.9468	0.8945	0.9421	0.9667	0.9724
LDA	0.7975	0.8363	0.8405	0.7257	0.8384	0.8899	0.9324
Naive Bayes	0.7690	0.8217	0.8051	0.7089	0.8133	0.8509	0.8966
XGBoost	0.9209	0.9457	0.9266	0.9114	0.9361	0.9624	0.9706
Gradient Boosting	0.9161	0.9340	0.9316	0.8903	0.9328	0.9585	0.9695
AdaBoost	0.9241	0.9370	0.9418	0.8945	0.9394	0.9573	0.9685

✅ Best Model: CatBoost (Highest overall AUC and PR AUC)

🧩 Steps in the Notebook

Data Loading & Cleaning

Checked missing values, duplicates, and outliers.

Exploratory Data Analysis (EDA)

Distribution plots, correlation heatmaps, and voice frequency analysis.

Feature Engineering

Feature scaling (StandardScaler), transformation (PowerTransformer), and selection.

Model Training & Comparison

Trained multiple ML models with default and tuned hyperparameters.

Evaluation

Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices.

Feature Importance

Analyzed top predictive voice parameters.

📈 Results Summary

CatBoost, XGBoost, and AdaBoost outperformed other models.

CatBoost achieved:

Accuracy: 92.7%

ROC AUC: 0.9667

PR AUC: 0.9724

Models performed significantly better after preprocessing and feature transformation.

💡 Insights

Voice-based features (e.g., jitter, shimmer, MDVP) showed strong correlation with Parkinson’s presence.

Ensemble models generalized better than linear classifiers.

Proper scaling and transformation (PowerTransformer) improved model performance.

🧠 Future Work

Integrate deep learning models (e.g., LSTM for voice sequence analysis).

Collect larger datasets for more robust generalization.

Deploy model as a web app using Streamlit or Flask.

📎 How to Run

Clone this repository

git clone https://github.com/yourusername/parkinsons-disease-prediction.git
cd parkinsons-disease-prediction


Install dependencies

pip install -r requirements.txt


Run the notebook or Streamlit app

jupyter notebook


or

streamlit run app.py
