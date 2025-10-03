####====================================================================####
#                        Cancer Prediction System                         #
#              Using Random Forest Classification Algorithm              #
#                          Author: [Karan Jadhav]                         #
#                        Last Updated: [30-09-2025]                       #
####====================================================================####

#==========================================================================#
#                            ****Output****                                #
# Accuracy is 96.49%                                                       #
# Confusion Matrix                                                         #
# [[40  3]                                                                 #
#  [ 1 70]]                                                                #
# Classification Report                                                    #
#               precision    recall  f1-score   support                    #
#           0       0.98      0.93      0.95        43                     #
#           1       0.96      0.99      0.97        71                     #
#                                                                          #
#    accuracy                           0.96       114                     #
#   macro avg       0.97      0.96      0.96       114                     #
#weighted avg       0.97      0.96      0.96       114                     #
#                                                                          #
# ROC-AUC score: 0.9580740255486406                                        #
# Precision: 0.958904109589041                                             #
# Recall Score : 0.9859154929577465                                       #
# F1 Score : 0.9722222222222222                                            #
#==========================================================================#

#==========================================================================#
#                           Import Libraries                               #
#==========================================================================#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score
)

#==========================================================================#
#                      Constants and Artifacts Setup                      #
#==========================================================================#
ARTIFACTS = Path("artifacts_sample")
ARTIFACTS.mkdir(exist_ok=True)  # Create artifacts folder if it doesn't exist
MODEL_PATH = ARTIFACTS / "RF_pipeline.joblib"  # Model save path
RANDOM_STATE = 42
TEST_SIZE = 0.2

#==========================================================================#
#                           Load Breast Cancer Dataset                     #
# Loads the dataset from sklearn and returns dataframe.                    #
#==========================================================================#
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data

#==========================================================================#
#                           Check Missing Values                           #
# Prints count of missing values per column.                              #
#==========================================================================#
def check_missing_values(df):
    print(df.isnull().sum())

#==========================================================================#
#                      Split Dataset and Scale Features                    #
# Splits the dataset into training and testing sets and scales features.   #
#==========================================================================#
def preprocess_data(df):
    X = df.drop(columns='target')
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

#==========================================================================#
#                          Train Random Forest Model                       #
# Trains RandomForestClassifier on training data.                         #
# Returns trained model.                                                    #
#==========================================================================#
def train_rf_model(X_train, y_train):
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

#==========================================================================#
#                 Save Trained Model Pipeline to Disk                      #
#==========================================================================#
def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"Random Forest model pipeline saved to: {path}")

#==========================================================================#
#                    Evaluate Model and Print All Metrics                  #
# Prints accuracy, confusion matrix, classification report, and more.      #
#==========================================================================#
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy is ", accuracy * 100)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print(cm)

    report = classification_report(y_test, y_pred)
    print("Classification Report ", report)

    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC score:", roc_auc)

    precision = precision_score(y_test, y_pred)
    print("Precision:", precision)

    recall = recall_score(y_test, y_pred)
    print("Recall Score :", recall)

    f1 = f1_score(y_test, y_pred)
    print("F1 Score :", f1)

#==========================================================================#
#                         Plot Feature Importance                          #
# Plots feature importances of the trained Random Forest model.            #
#==========================================================================#
def plot_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    importances.plot(kind='barh', color='skyblue')
    plt.title("Feature Importances from Breast Cancer Data")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

#==========================================================================#
#                            Full Pipeline Call                            #
# Loads data, preprocesses, trains model, evaluates, plots, and saves model#
#==========================================================================#
def CancerPrediction():
    df, data = load_data()
    check_missing_values(df)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = train_rf_model(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluate_model(y_test, y_pred)
    plot_feature_importance(model, data.feature_names)
    save_model(model, MODEL_PATH)

#==========================================================================#
#                                Main Call                                 #
#==========================================================================#
def main():
    CancerPrediction()

if __name__ == "__main__":
    main()
