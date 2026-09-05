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

## 🎯 Results
Four classification models were trained and evaluated on 768 patient records (Pima Indians Diabetes dataset) using an 80/20 train-test split:

| Model | Accuracy |
|---|---|
| Logistic Regression | 75.3% |
| Random Forest | ~75% |
| SVM | 73.4% |
| KNN | 69.5% |

**Logistic Regression achieved the best overall performance**, with the classification report showing 81% precision for non-diabetic cases and 65% precision for diabetic cases.

**Feature Importance (via Random Forest):**

| Feature | Importance |
|---|---|
| Plasma Glucose Concentration | 25.4% |
| Body Mass Index (BMI) | 16.2% |
| Age | 13.9% |
| Diabetes Pedigree Function | 12.5% |

**Key Insight:** Glucose level is by far the strongest predictor of diabetes risk, followed by BMI and age — consistent with established clinical understanding of Type 2 diabetes risk factors.

## 💡 Business Insight
The analysis shows that customers with higher monthly charges and shorter tenure are more likely to churn. Companies can use this information to improve retention strategies.

## 📌 Future Improvements
- Hyperparameter tuning
- Model deployment using a web application
- Real-time churn prediction

## 👩‍💻 Author
Varsha  
Aspiring Data Analyst
