# Bank Customer Churn Prediction using Artificial Neural Networks (ANN)

This project focuses on predicting customer churn using a deep learning approach with an Artificial Neural Network (ANN). The goal is to build a model that can identify customers likely to leave a bank, enabling proactive retention strategies.

## Project Overview

Customer churn is a critical metric for businesses. Predicting churn allows companies to understand the factors driving customers away and take targeted actions to retain valuable customers. This project utilizes a publicly available dataset related to bank customer churn and employs an ANN to build a predictive model.

## Dataset

The project uses the [Churn Modelling Dataset](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction). This dataset contains information about bank customers, including their demographics, account details, and whether they have churned or not.

The following columns are used in the analysis and model:

*   `CreditScore`
*   `Geography`
*   `Gender`
*   `Age`
*   `Tenure`
*   `Balance`
*   `NumOfProducts`
*   `HasCrCard`
*   `IsActiveMember`
*   `EstimatedSalary`
*   `Exited` (Target Variable: 1 if the customer churned, 0 otherwise)

## Project Steps

1.  **Data Loading and Initial Exploration:** Load the dataset and perform initial checks for data types, missing values, and duplicates.
2.  **Exploratory Data Analysis (EDA):** Analyze the distribution of features, identify potential relationships between features and the target variable (churn), and visualize the data to gain insights.
3.  **Data Preprocessing:**
    *   Handle categorical features using one-hot encoding.
    *   Address the class imbalance in the target variable by oversampling the minority class (churned customers).
    *   Split the data into training and testing sets.
    *   Scale numerical features to ensure they have similar ranges.
4.  **Model Building:** Construct an Artificial Neural Network (ANN) using the Keras Sequential API. The model consists of multiple Dense layers with different activation functions, Batch Normalization, and Dropout layers for regularization.
5.  **Model Compilation:** Compile the ANN model with an appropriate optimizer (Adam), loss function (binary crossentropy for binary classification), and evaluation metrics (accuracy).
6.  **Model Training:** Train the model on the preprocessed training data. Early stopping is implemented to prevent overfitting and stop training when validation performance plateaus.
7.  **Model Evaluation:** Evaluate the trained model's performance on the test data using metrics such as accuracy, precision, recall, F1-score, and the confusion matrix.

## Model Architecture

The ANN model is a Multilayer Perceptron (MLP) with the following structure:

*   **Input Layer:** 11 input features
*   **Hidden Layers:** Multiple Dense layers with varying numbers of neurons and activation functions (tanh, sigmoid, relu). Batch Normalization and Dropout layers are included for regularization and stability.
*   **Output Layer:** 1 neuron with a sigmoid activation function for binary classification.

## Dependencies

The project requires the following libraries:

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `sklearn`
*   `tensorflow`
*   `pylab`
*   `scipy`

