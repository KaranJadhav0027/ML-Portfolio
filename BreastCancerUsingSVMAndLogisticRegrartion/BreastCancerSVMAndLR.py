# ####====================================================================####
# #                        Breast Cancer Detection                          #
# #     Achieved high accuracy using Logistic Regression & SVM             #
# #                         Author: Karan Sanjay Jadhav                    #
# #                       Last Updated: 09-08-2025                         #
# ####====================================================================####

# #==========================================================================#
# #                          Import Libraries                                #
# #==========================================================================#
# import pandas as pd
# from pathlib import Path
# import joblib
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score, confusion_matrix, classification_report,
#     roc_auc_score, precision_score, recall_score, f1_score
# )

# ARTIFACTS = Path("artifacts_sample")
# ARTIFACTS.mkdir(exist_ok=True)

# MODEL_PATH_LR = ARTIFACTS / "BreastCancer_LogisticRegression_pipeline.joblib"
# MODEL_PATH_SVM = ARTIFACTS / "BreastCancer_SVM_pipeline.joblib"
# RANDOM_STATE = 42
# TEST_SIZE = 0.2

# #################################
# # Function: load_dataset
# # Inputs: path (str) - Path to CSV file
# # Output: pd.DataFrame - Loaded dataset
# # Description: Loads dataset from the provided CSV path
# # Author: Karan Sanjay Jadhav
# #################################
# def load_dataset(path):
#     return pd.read_csv(path,)

# #################################
# # Function: clean_data
# # Inputs: df (pd.DataFrame) - Raw dataframe
# # Output: pd.DataFrame - Cleaned dataframe
# # Description: Removes missing values from the dataframe
# # Author: Karan Sanjay Jadhav
# #################################
# def clean_data(df):
#     return df.dropna()

# #################################
# # Function: separate_features_target
# # Inputs: df (pd.DataFrame) - Cleaned dataframe
# #         target_column (str) - Target column name
# # Output: Tuple of features (X) and target (y)
# # Description: Separates independent and target variables
# # Author: Karan Sanjay Jadhav
# #################################
# def separate_features_target(df, target_column='CancerType'):
#     X = df.drop(columns=[target_column])
#     y = df[target_column].map({'M': 1, 'B': 0})  # Map malignant/benign to 1/0
#     return X, y

# #################################
# # Function: split_data
# # Inputs: X (pd.DataFrame), y (pd.Series)
# # Output: Train-test split datasets
# # Description: Splits data into train and test sets
# # Author: Karan Sanjay Jadhav
# #################################
# def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
#     return train_test_split(X, y, test_size=test_size, random_state=random_state)

# #################################
# # Function: train_logistic_regression_pipeline
# # Inputs: Training and testing datasets
# # Output: y_pred - Predictions
# # Description: Trains and saves Logistic Regression pipeline
# # Author: Karan Sanjay Jadhav
# #################################
# def train_logistic_regression_pipeline(X_train, y_train, X_test, y_test, save_path=MODEL_PATH_LR):
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', LogisticRegression(random_state=RANDOM_STATE))
#     ])
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred)*100)
#     joblib.dump(pipeline, save_path)
#     print(f"Logistic Regression pipeline saved to: {save_path}")
#     return y_pred

# #################################
# # Function: train_svm_pipeline
# # Inputs: Training and testing datasets
# # Output: y_pred - Predictions
# # Description: Trains and saves Support Vector Machine (SVM) pipeline
# # Author: Karan Sanjay Jadhav
# #################################
# def train_svm_pipeline(X_train, y_train, X_test, y_test, save_path=MODEL_PATH_SVM):
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
#     ])
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     print("SVM Accuracy:", accuracy_score(y_test, y_pred)*100)
#     joblib.dump(pipeline, save_path)
#     print(f"SVM pipeline saved to: {save_path}")
#     return y_pred

# #################################
# # Function: evaluate_model
# # Inputs: y_test (True labels), y_pred (Predicted labels)
# # Output: None
# # Description: Prints confusion matrix and evaluation metrics
# # Author: Karan Sanjay Jadhav
# #################################
# def evaluate_model(y_test, y_pred):
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy:", accuracy * 100)
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))
#     print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))
#     print("F1 Score:", f1_score(y_test, y_pred))

# #################################
# # Function: BreastCancerCase
# # Inputs: path (str) - Path to dataset
# # Output: None
# # Description: Main pipeline for breast cancer classification
# # Author: Karan Sanjay Jadhav
# #################################
# def BreastCancerCase(path):
#     df = load_dataset(path)
#     df = clean_data(df)
#     X, y = separate_features_target(df)
#     X_train, X_test, y_train, y_test = split_data(X, y)

#     print("\nTraining Logistic Regression model...")
#     y_pred_lr = train_logistic_regression_pipeline(X_train, y_train, X_test, y_test)
#     print("\nLogistic Regression Model Evaluation:")
#     evaluate_model(y_test, y_pred_lr)

#     print("\nTraining Support Vector Machine model...")
#     y_pred_svm = train_svm_pipeline(X_train, y_train, X_test, y_test)
#     print("\nSVM Model Evaluation:")
#     evaluate_model(y_test, y_pred_svm)

# #################################
# # Function: main
# # Inputs: None
# # Output: None
# # Description: Entry point for breast cancer detection pipeline
# # Author: Karan Sanjay Jadhav
# #################################
# def main():
#     BreastCancerCase("breast-cancer-wisconsin.csv")

# if __name__ == "__main__":
#     main()
####====================================================================####
#                        Breast Cancer Detection                          #
#     Achieved high accuracy using Logistic Regression & SVM             #
#                         Author: Karan Sanjay Jadhav                    #
#                       Last Updated: 09-08-2025                         #
####====================================================================####

#==========================================================================#
#                          Import Libraries                                #
#==========================================================================#
import pandas as pd
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score
)

ARTIFACTS = Path("artifacts_sample")
ARTIFACTS.mkdir(exist_ok=True)

MODEL_PATH_LR = ARTIFACTS / "BreastCancer_LogisticRegression_pipeline.joblib"
MODEL_PATH_SVM = ARTIFACTS / "BreastCancer_SVM_pipeline.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

#################################
# Function: load_dataset
# Inputs: path (str) - Path to CSV file
# Output: pd.DataFrame - Loaded dataset
# Description: Loads dataset from the provided CSV path and converts '?' to NaN
# Author: Karan Sanjay Jadhav
#################################
def load_dataset(path):
    # Convert '?' to NaN while reading csv
    df = pd.read_csv(path, na_values='?')
    return df

#################################
# Function: clean_data
# Inputs: df (pd.DataFrame) - Raw dataframe
# Output: pd.DataFrame - Cleaned dataframe with no missing values
# Description: Removes missing values from the dataframe
# Author: Karan Sanjay Jadhav
#################################
def clean_data(df):
    # Drop rows with NaN values
    df_cleaned = df.dropna()
    
    # Ensure all columns except target are numeric
    for col in df_cleaned.columns:
        if col != 'CancerType':  # Replace with actual target column if different
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # Drop any rows that became NaN after conversion
    df_cleaned = df_cleaned.dropna()
    return df_cleaned

#################################
# Function: separate_features_target
# Inputs: df (pd.DataFrame) - Cleaned dataframe
#         target_column (str) - Target column name
# Output: Tuple of features (X) and target (y)
# Description: Separates independent and target variables
# Author: Karan Sanjay Jadhav
#################################
def separate_features_target(df, target_column='CancerType'):
    X = df.drop(columns=[target_column])
    y = df[target_column].map({'M': 1, 'B': 0})  # Map malignant/benign to 1/0
    return X, y

#################################
# Function: split_data
# Inputs: X (pd.DataFrame), y (pd.Series)
# Output: Train-test split datasets
# Description: Splits data into train and test sets
# Author: Karan Sanjay Jadhav
#################################
def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

#################################
# Function: train_logistic_regression_pipeline
# Inputs: Training and testing datasets
# Output: y_pred - Predictions
# Description: Trains and saves Logistic Regression pipeline
# Author: Karan Sanjay Jadhav
#################################
def train_logistic_regression_pipeline(X_train, y_train, X_test, y_test, save_path=MODEL_PATH_LR):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=RANDOM_STATE))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred)*100)
    joblib.dump(pipeline, save_path)
    print(f"Logistic Regression pipeline saved to: {save_path}")
    return y_pred

#################################
# Function: train_svm_pipeline
# Inputs: Training and testing datasets
# Output: y_pred - Predictions
# Description: Trains and saves Support Vector Machine (SVM) pipeline
# Author: Karan Sanjay Jadhav
#################################
def train_svm_pipeline(X_train, y_train, X_test, y_test, save_path=MODEL_PATH_SVM):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred)*100)
    joblib.dump(pipeline, save_path)
    print(f"SVM pipeline saved to: {save_path}")
    return y_pred

#################################
# Function: evaluate_model
# Inputs: y_test (True labels), y_pred (Predicted labels)
# Output: None
# Description: Prints confusion matrix and evaluation metrics
# Author: Karan Sanjay Jadhav
#################################
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy * 100)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

#################################
# Function: BreastCancerCase
# Inputs: path (str) - Path to dataset
# Output: None
# Description: Main pipeline for breast cancer classification
# Author: Karan Sanjay Jadhav
#################################
def BreastCancerCase(path):
    df = load_dataset(path)
    df = clean_data(df)
    X, y = separate_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining Logistic Regression model...")
    y_pred_lr = train_logistic_regression_pipeline(X_train, y_train, X_test, y_test)
    print("\nLogistic Regression Model Evaluation:")
    evaluate_model(y_test, y_pred_lr)

    print("\nTraining Support Vector Machine model...")
    y_pred_svm = train_svm_pipeline(X_train, y_train, X_test, y_test)
    print("\nSVM Model Evaluation:")
    evaluate_model(y_test, y_pred_svm)

#################################
# Function: main
# Inputs: None
# Output: None
# Description: Entry point for breast cancer detection pipeline
# Author: Karan Sanjay Jadhav
#################################
def main():
    BreastCancerCase("breast-cancer-wisconsin.csv")

if __name__ == "__main__":
    main()
