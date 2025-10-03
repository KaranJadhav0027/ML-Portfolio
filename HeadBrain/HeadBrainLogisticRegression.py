####====================================================================####
#                    Head Brain Classification System                     #
#     Predicting Intelligence from Brain Size using ML Classification     #
#                         Author: [Karan Jadhav]                          #
#                      Last Updated: [30-09-2025]                         #
####====================================================================####

#==========================================================================#
#                            ****Output****                                #
# Accuracy : 66.67%                                                        #
# Confusion Matrix                                                         #
# [[21 08]                                                                 #
#  [08 11]]                                                                #
# Classification Report                                                    #
#             precision    recall  f1-score   support                      #
#                                                                          #
#            0       0.72      0.73      0.73        29                    #
#            1       0.58      0.58      0.58        19                    #
#                                                                          #
#     accuracy                           0.67       48                     #
#    macro avg       0.65      0.65      0.65       48                     #
# weighted avg       0.67      0.67      0.67       48                     #
#                                                                          #
# ROC-AUC score: 0.6515426497277677                                        #
# Precision:  0.7241379310344828                                           #
# Recall Score :  0.7241379310344828                                       #
# F1 Score :  0.7241379310344828                                           #
# Model saved to: artifacts_headbrain\RandomForest_model.joblib            #                                      #
#==========================================================================#

#==========================================================================#
#                         Import Required Libraries                        #
#==========================================================================#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score
)

#==========================================================================#
#                         Constants and Paths                              #
#==========================================================================#
ARTIFACTS = Path("artifacts_headbrain")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "RandomForest_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

#==========================================================================#
#                          Load Dataset from CSV                           #
#                                                                          #
# Function:     load_dataset                                               #
# Inputs:       path (str)                                                 #
#               - File path to the dataset CSV file.                       #
#                                                                          #
# Description:  Loads the dataset into a pandas DataFrame.                 #
#                                                                          #
# Returns:      pd.DataFrame                                               #
#               - Loaded dataset                                           #
#==========================================================================#
def load_dataset(path):
    return pd.read_csv(path)

#==========================================================================#
#                          Clean the Dataset                               #
#                                                                          #
# Function:     clean_data                                                 #
# Inputs:       df (pd.DataFrame)                                          #
#               - Raw dataset                                              #
#                                                                          #
# Description:  Cleans the dataset by removing rows with missing values.   #
#                                                                          #
# Returns:      pd.DataFrame                                               #
#               - Cleaned dataset                                          #
#==========================================================================#
def clean_data(df):
    return df.dropna()

#==========================================================================#
#                     Separate Features and Target                         #
#                                                                          #
# Function:     separate_features_target                                   #
# Inputs:       df (pd.DataFrame)                                          #
#               - Cleaned dataset                                          #
#               target_column (str)                                        #
#               - Name of the target column (default='Outcome')            #
#                                                                          #
# Description:  Splits the dataset into input features (X) and target (y).#
#                                                                          #
# Returns:      Tuple (X, y)                                               #
#==========================================================================#
def separate_features_target(df, target_column='Gender'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

#==========================================================================#
#                          Scale the Features                              #
#                                                                          #
# Function:     scale_features                                             #
# Inputs:       X (pd.DataFrame)                                           #
#               - Feature matrix                                           #
#                                                                          #
# Description:  Standardizes the feature matrix using StandardScaler.     #
#                                                                          #
# Returns:      np.ndarray                                                 #
#               - Scaled features                                          #
#==========================================================================#
def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

#==========================================================================#
#                     Split into Training and Test Sets                    #
#                                                                          #
# Function:     split_data                                                 #
# Inputs:       X (array or DataFrame)                                     #
#               y (Series or array)                                        #
#               test_size (float)                                          #
#               random_state (int)                                         #
#                                                                          #
# Description:  Splits the dataset into training and testing sets.        #
#                                                                          #
# Returns:      X_train, X_test, y_train, y_test                           #
#==========================================================================#
def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

#==========================================================================#
#                          Train Classification Model                      #
#                                                                          #
# Function:     train_model                                                #
# Inputs:       X_train (array or DataFrame)                               #
#               y_train (Series or array)                                  #
#                                                                          #
# Description:  Trains a classification model on the training dataset.    #
#               (e.g., Logistic Regression, SVM, etc.)                     #
#                                                                          #
# Returns:      model                                                      #
#               - Trained classifier                                       #
#==========================================================================#
def train_logistic_model(X_train, y_train):
    model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=None,          # Allow trees to grow fully
    class_weight='balanced', # Handle class imbalance
    random_state=42
    )
    model.fit(X_train, y_train)
    return model

#==========================================================================#
#                        Evaluate the Trained Model                        #
#                                                                          #
# Function:     evaluate_model                                             #
# Inputs:       y_test (array or Series)                                   #
#               y_pred (array)                                             #
#                                                                          #
# Description:  Evaluates the model's predictions using classification    #
#               metrics like accuracy, precision, recall, F1, AUC, etc.   #
#                                                                          #
# Returns:      None                                                       #
#==========================================================================#
def evaluate_model(y_test, y_pred):
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
    print("Classification Report\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

#==========================================================================#
#                          Save Trained Model                              #
#                                                                          #
# Function:     save_model                                                 #
# Inputs:       model (sklearn model)                                      #
#               path (str or Path)                                         #
#               - Destination to save the model                            #
#                                                                          #
# Description:  Saves the trained model pipeline to disk using joblib.    #
#                                                                          #
# Returns:      None                                                       #
#==========================================================================#
def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

#==========================================================================#
#                       Breast Cancer Classification                       #
#                                                                          #
# Function:     BreastCancerCase                                           #
# Inputs:       path (str)                                                 #
#               - Path to the dataset (CSV format)                         #
#                                                                          #
# Description:  Full machine learning pipeline for breast cancer           #
#               classification. This function handles:                     #
#                 - Data loading and cleaning                              #
#                 - Feature scaling                                        #
#                 - Train-test splitting                                   #
#                 - Model training                                         #
#                 - Evaluation with metrics                                #
#                 - Saving the trained model                               #
#                                                                          #
# Output:        None                                                      #
#==========================================================================#
def HeadBrainClassificationPipeline(path):
    df = load_dataset(path)
    df = clean_data(df)

    X, y = separate_features_target(df, target_column='Gender')  
    X_scaled = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    model = train_logistic_model(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluate_model(y_test, y_pred)
    save_model(model)

#==========================================================================#
#                               Main Function                              #
#                                                                          #
# Function:     main                                                       #
# Description:  Entry point of the program. Calls the breast cancer        #
#               classification pipeline with the given dataset path.      #
#                                                                          #
# Returns:      None                                                       #
#==========================================================================#
def main():
    HeadBrainClassificationPipeline("HeadBrain.csv")  # Replace with your dataset

if __name__ == "__main__":
    main()
