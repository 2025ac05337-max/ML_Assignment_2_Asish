# Income Classification using Machine Learning

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for predicting whether an individual's annual income is <=50K or >50K. Five classification algorithms are implemented and evaluated using multiple performance metrics.

## b. Dataset Description

The dataset used for this project is an Income Classification dataset containing demographic, educational, employment, and financial attributes.

The target variable is `income`, with two classes:

- 0: <=50K
- 1: >50K

After preprocessing, the dataset contains 32,561 instances and 97 encoded features.

The dataset was divided into training and testing sets using an 80:20 split.

Training set: 26,048 instances  
Testing set: 6,513 instances

Categorical variables were converted into numerical features using one-hot encoding. Feature scaling was applied before model training.

## c. GitHub Repository Link

GitHub Repository: [Add your GitHub repository link here]

## d. Models Used

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble)

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8549 | 0.9074 | 0.7376 | 0.6167 | 0.6718 | 0.5834 |
| Decision Tree | 0.8567 | 0.9006 | 0.7222 | 0.6582 | 0.6887 | 0.5970 |
| KNN | 0.8263 | 0.8498 | 0.6557 | 0.5867 | 0.6193 | 0.5086 |
| Naive Bayes | 0.4167 | 0.6830 | 0.2889 | 0.9732 | 0.4455 | 0.2332 |
| Random Forest (Ensemble) | 0.8617 | 0.9166 | 0.7986 | 0.5689 | 0.6644 | 0.5936 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved an accuracy of 85.49% and an AUC of 90.74%. It provided balanced overall classification performance with an MCC of 58.34%. |
| Decision Tree | Decision Tree achieved 85.67% accuracy and 90.06% AUC. It provided a good balance between precision and recall and achieved the highest MCC of 59.70%. |
| KNN | KNN achieved 82.63% accuracy and 84.98% AUC. Its performance was lower than Logistic Regression, Decision Tree and Random Forest, but it still provided reasonable classification results. |
| Naive Bayes | Naive Bayes produced the lowest accuracy of 41.67%. Although its recall was very high at 97.32%, its precision was only 28.89%, resulting in weaker overall performance. |
| Random Forest (Ensemble) | Random Forest achieved the highest accuracy of 86.17%, highest AUC of 91.66%, and highest precision of 79.86%. It provided the strongest overall performance based on these metrics. |

## Overall Winner

**Random Forest (Ensemble)** is the overall winner for this dataset based on its highest accuracy, AUC, and precision.

## Streamlit Application

The project includes an interactive Streamlit application that provides:

- Test dataset upload
- Machine learning model selection
- Evaluation metrics
- Confusion matrix
- Classification report
- Prediction summary

### Streamlit App Link

[Add your deployed Streamlit application link here]