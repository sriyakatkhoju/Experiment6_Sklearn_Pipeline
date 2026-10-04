# Experiment 06 - Scikit-learn Pipeline

## Experiment Title
Scikit-learn Pipeline: Design and Implement a Scikit-learn Preprocessing Pipeline for Data Preprocessing and Model Training Using the Pipeline API

## Aim
To design and implement a Scikit-learn Pipeline that performs data preprocessing and model training in a single workflow.

## Objectives
- Learn how to use the Scikit-learn Pipeline API.
- Perform preprocessing and model training together.
- Reduce repetitive code.
- Build a clean and reusable machine learning workflow.

## Dataset
Loan Approval Dataset

## Preprocessing
The pipeline performs:
- Missing value handling for numerical features
- Standard scaling of numerical features
- Missing value handling for categorical features
- One-hot encoding of categorical features

## Model
Decision Tree Classifier

## Pipeline Structure

Loan Approval Dataset
        |
        v
ColumnTransformer
   |           |
Numeric     Categorical
   |           |
Imputation   Imputation
   |           |
Scaling      One-Hot Encoding
   |           |
   +-----+-----+
         |
         v
Decision Tree Classifier
         |
         v
     Prediction
         |
         v
      Accuracy

## Train-Test Split
- Training samples: 491
- Testing samples: 123
- Test size: 20%
- Random state: 42

## Result

Accuracy: 82.11%

The Scikit-learn Pipeline successfully performed data preprocessing and model training in a single workflow.

## Tools Used
- Python
- Pandas
- Scikit-learn
- VS Code
- Git and GitHub

## Conclusion
A Scikit-learn Pipeline was successfully created to combine data preprocessing and Decision Tree model training in a single workflow. The pipeline simplified the machine learning process and achieved an accuracy of 82.11% on the Loan Approval dataset.