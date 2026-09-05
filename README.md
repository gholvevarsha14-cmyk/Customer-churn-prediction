# Customer-churn-prediction
This project predicts whether a customer will leave (churn) a telecom service using machine learning techniques. The goal is to help businesses identify customers at risk of leaving and take proactive retention actions.

## 📊 Project Overview
Customer churn is a major problem for telecom companies. By analyzing customer data and behavior patterns, this project builds predictive models to identify customers likely to churn.

## 🧰 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 📁 Dataset
The dataset contains information about telecom customers including:
- Customer demographics
- Account information
- Services subscribed
- Monthly charges
- Churn status

## 🔍 Exploratory Data Analysis (EDA)
EDA was performed to understand:
- Distribution of customer demographics
- Relationship between services and churn
- Correlation between features
- Patterns influencing churn

## 🤖 Machine Learning Models
The following models were used for prediction:
- Logistic Regression
- Random Forest
- Decision Tree
- K-Nearest Neighbors

## 📈 Model Evaluation
Models were evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

## 🤖 Machine Learning Models

The following models were used for prediction:

- Logistic Regression
- Random Forest
- XGBoost

## 🎯 Results

Three classification models were trained and evaluated on 7,043 telecom customer records using an 80/20 train-test split:

| Model               | Accuracy | Recall | ROC-AUC |
| ------------------- | -------- | ------ | ------- |
| Logistic Regression | 82.1%    | 55.2%  | 0.735   |
| Random Forest       | 78.9%    | 47.7%  | 0.690   |
| XGBoost              | 78.3%    | 49.1%  | 0.690   |

**Logistic Regression achieved the best overall performance**, with the highest accuracy and ROC-AUC among the three models.

**Key Insight:** Feature importance analysis identified **customer tenure** and **monthly charges** as the strongest predictors of churn — customers with shorter tenure and higher monthly bills are at greatest risk of leaving, a useful signal for targeted retention campaigns.

## 💡 Business Insight
The analysis shows that customers with higher monthly charges and shorter tenure are more likely to churn. Companies can use this information to improve retention strategies.

## 📌 Future Improvements
- Hyperparameter tuning
- Model deployment using a web application
- Real-time churn prediction

## 👩‍💻 Author
Varsha  
Aspiring Data Analyst
