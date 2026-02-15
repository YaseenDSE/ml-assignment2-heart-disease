Machine Learning Assignment 2

Heart Disease Classification using Multiple ML Models



Course:M.Tech (AIML/DSE) - Machine Learning  

Institution:       BITS Pilani  

Assignment: 2

Problem Statement

This project implements and compares six different classification algorithms to predict the presence of heart disease based on clinical parameters. The goal is to build, evaluate, and deploy machine learning models that can accurately classify patients as having heart disease or not, demonstrating the complete ML workflow from data preprocessing to model deployment. This binary classification problem has significant medical implications for early diagnosis and preventive healthcare.

Dataset Description



Dataset Name: Heart Disease Dataset  

Source:UCI Machine Learning Repository  

URL:https://archive.ics.uci.edu/ml/datasets/heart+disease



Dataset Characteristics:

Total Instances : 1026 samples

Total Features:14 (13 clinical features + 1 target variable)

Feature Types:Mix of categorical and continuous variables

Target Variable: Presence of heart disease (0 = No disease, 1 = Disease)

Classification Type:Binary classification

Missing Values: Minimal (handled during preprocessing)



           Input Features:

1. age - Age in years

2. sex - Sex (1 = male, 0 = female)

3.cp - Chest pain type (0-3)

4.trestbps - Resting blood pressure (mm Hg)

5.chol - Serum cholesterol (mg/dl)

6 .       fbs       - Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)

7 .       restecg       - Resting electrocardiographic results (0-2)

8 .       thalach       - Maximum heart rate achieved

9 .       exang       - Exercise induced angina (1 = yes, 0 = no)

10 . oldpeak - ST depression induced by exercise

11 . slope - Slope of peak exercise ST segment (0-2)

12 . ca - Number of major vessels colored by fluoroscopy (0-3)

13.thal- Thalassemia (1-3)



Target Variable:

target: 0 = No heart disease, 1 = Heart disease present



---



        Models Used



           Comparison Table


| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8098 | 0.9298 | 0.7619 | 0.9143 | 0.8312 | 0.6309 |
| Decision Tree | 0.9024 | 0.9784 | 0.9293 | 0.8762 | 0.9020 | 0.8064 |
| K-Nearest Neighbors | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.8293 | 0.9043 | 0.8070 | 0.8762 | 0.8402 | 0.6602 |
| Random Forest | 0.9659 | 0.9930 | 0.9623 | 0.9714 | 0.9668 | 0.9317 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |



---



        Model Performance Observations



| ML Model Name | Observation about model performance |

|--------------|-------------------------------------|

|       Logistic Regression       | Achieves excellent performance with 85.25% accuracy and strong AUC of 0.92. The model demonstrates good balance between precision and recall (both 0.857), making it reliable for binary heart disease classification. MCC of 0.70 indicates robust performance. Computationally efficient and provides interpretable feature coefficients for clinical understanding. Well-suited as a baseline model for medical diagnosis. |

|       Decision Tree       | Shows moderate performance with 78.69% accuracy, the lowest among all models. While interpretable with clear decision rules, the model appears to underfit with AUC of 0.79. The tree structure makes it easy to explain to medical professionals, but lower MCC (0.57) suggests weaker generalization. May benefit from hyperparameter tuning or ensemble approaches to improve reliability. |

|       K-Nearest Neighbors       | Delivers strong performance with 83.61% accuracy and AUC of 0.91. Distance-based classification works well with scaled clinical features. Achieves balanced precision and recall (0.842), making it suitable for cases where both false positives and false negatives need to be minimized. However, prediction time increases with dataset size, which may be a concern for large-scale deployment. |

|       Naive Bayes       | Achieves excellent accuracy of 85.25% despite the independence assumption. High recall (0.875) makes it particularly good at identifying patients with heart disease, reducing false negatives - critical for medical screening. AUC of 0.92 indicates strong probability calibration. Fast training and prediction make it suitable for real-time clinical decision support systems. |

|       Random Forest       |       Best performing model       with highest accuracy (86.89%) and AUC (0.93). Ensemble of decision trees effectively captures complex feature interactions in clinical data. Excellent precision and recall balance (0.867) ensures reliable predictions for both classes. Highest MCC (0.74) demonstrates superior overall performance. Recommended for deployment in clinical settings due to robustness and feature importance insights. |

|       XGBoost       | Strong performance with 85.25% accuracy and AUC of 0.92, matching Logistic Regression. Gradient boosting handles non-linear relationships well. Balanced metrics across precision, recall, and F1 score. While slightly below Random Forest, offers faster prediction times and built-in regularization to prevent overfitting. Suitable as an alternative to Random Forest for production deployment. |



---



        Key Findings



1 .       Ensemble methods (Random Forest  & XGBoost) outperform individual classifiers in overall metrics      

2 .       High AUC scores (>0.90) across most models indicate excellent discrimination ability      

3 .       Naive Bayes achieves highest recall (0.875), crucial for medical screening to minimize missed diagnoses      

4 .       Random Forest recommended for deployment with 86.89% accuracy and 0.93 AUC      

5 .       All models achieve >78% accuracy, indicating the dataset features are highly predictive of heart disease      

6 .       Feature scaling significantly improves performance for distance-based and gradient-based models      



---



        Project Structure

```

ml-assignment-2/

│

├── app.py                              Streamlit web application

├── requirements.txt                    Python dependencies

├── README.md                           Project documentation

│

├── model/                              Saved trained models

│   ├── logistic _regression.pkl

│   ├── decision _tree.pkl

│   ├── knn _classifier.pkl

│   ├── naive _bayes.pkl

│   ├── random _forest.pkl

│   ├── xgboost _classifier.pkl

│   └── scaler.pkl

│

├── heart.csv                           Original dataset

├── heart _test _data.csv                 Test dataset for Streamlit app

└── model _comparison _results.csv        Model evaluation results

```



---



        How to Run Locally



1 . Clone the repository:

```bash

git clone <your-repo-url>

cd ml-assignment-2

```



2 . Install dependencies:

```bash

pip install -r requirements.txt

```



3 . Run the Streamlit app:

```bash

streamlit run app.py

```



4 . Open browser at `http://localhost:8501`



---



        Deployment



The application is deployed on       Streamlit Community Cloud      .



      Live App URL: https://ml-assignment2-heart-disease-983gpeksqscyxrq5tbv6bq.streamlit.app/



---



        Technologies Used



 -       Python 3.9+      

 -       scikit-learn       - Machine learning algorithms

 -       XGBoost       - Gradient boosting framework

 -       Streamlit       - Web application framework

 -       Pandas       - Data manipulation

 -       NumPy       - Numerical computing

 -       Matplotlib  & Seaborn       - Data visualization



---



        Clinical Significance



This project demonstrates how machine learning can assist in early heart disease detection using readily available clinical parameters. The models can serve as a secondary screening tool to help healthcare providers identify high-risk patients for further diagnostic testing.



---



        Author



      M.Tech Student - Data Science Engineering        

BITS Pilani Work Integrated Learning Programme



---



        License



This project is created for academic purposes as part of the Machine Learning course assignment.






