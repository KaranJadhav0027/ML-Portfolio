####====================================================================####
#                     Titanic Survival Predictor                         #
#       Built a classification model to predict passenger survival       #
#         Using Decision Trees & Logistic Regression                      #
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score
)

ARTIFACTS = Path("artifacts_sample")
ARTIFACTS.mkdir(exist_ok=True)

MODEL_PATH_DT = ARTIFACTS / "Titanic_DecisionTree_pipeline.joblib"
MODEL_PATH_LR = ARTIFACTS / "Titanic_LogisticRegression_pipeline.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

#################################
# Function: load_dataset
# Inputs: path (str) - Path to dataset CSV file
# Output: pd.DataFrame - Loaded dataset
# Description: Loads dataset from a CSV file
# Author: Karan Sanjay Jadhav
#################################
def load_dataset(path):
    return pd.read_csv(path)

#################################
# Function: clean_data
# Inputs: df (pd.DataFrame) - Input dataframe
# Output: pd.DataFrame - Dataframe after dropping missing values
# Description: Cleans the dataset by removing rows with missing values
# Author: Karan Sanjay Jadhav
#################################
def clean_data(df):
    return df.dropna()

#################################
# Function: separate_features_target
# Inputs: df (pd.DataFrame) - Cleaned dataframe
#         target_column (str) - Target variable name, default 'Survived'
# Output: tuple (X (pd.DataFrame), y (pd.Series)) - Features and target
# Description: Separates features and target variable from dataframe
# Author: Karan Sanjay Jadhav
#################################
def separate_features_target(df, target_column='Survived'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

#################################
# Function: split_data
# Inputs: X (pd.DataFrame) - Feature matrix
#         y (pd.Series) - Target vector
#         test_size (float) - Test set size ratio (default 0.2)
#         random_state (int) - Random seed for reproducibility
# Output: tuple (X_train, X_test, y_train, y_test) - Split datasets
# Description: Splits dataset into training and testing sets
# Author: Karan Sanjay Jadhav
#################################
def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

#################################
# Function: train_decision_tree_pipeline
# Inputs: X_train (pd.DataFrame) - Training features
#         y_train (pd.Series) - Training labels
#         X_test (pd.DataFrame) - Testing features
#         y_test (pd.Series) - Testing labels
#         save_path (str/Path) - Path to save the model (default MODEL_PATH_DT)
# Output: np.ndarray - Predicted labels on test set
# Description: Trains Decision Tree model pipeline and saves it to disk
# Author: Karan Sanjay Jadhav
#################################
def train_decision_tree_pipeline(X_train, y_train, X_test, y_test, save_path=MODEL_PATH_DT):
    pipeline = Pipeline([
        ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred)*100)
    joblib.dump(pipeline, save_path)
    print(f"Decision Tree pipeline saved to: {save_path}")
    return y_pred

#################################
# Function: train_logistic_regression_pipeline
# Inputs: X_train (pd.DataFrame) - Training features
#         y_train (pd.Series) - Training labels
#         X_test (pd.DataFrame) - Testing features
#         y_test (pd.Series) - Testing labels
#         save_path (str/Path) - Path to save the model (default MODEL_PATH_LR)
# Output: np.ndarray - Predicted labels on test set
# Description: Trains Logistic Regression pipeline with scaling and saves it
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
# Function: evaluate_model
# Inputs: y_test (pd.Series) - True labels
#         y_pred (np.ndarray) - Predicted labels
# Output: None
# Description: Prints accuracy, confusion matrix, classification report and other metrics
# Author: Karan Sanjay Jadhav
#################################
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy*100)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

#################################
# Function: TitanicCase
# Inputs: path (str) - Path to the Titanic CSV dataset
# Output: None
# Description: Full pipeline that loads data, cleans, trains models, evaluates them
# Author: Karan Sanjay Jadhav
#################################
def TitanicCase(path):
    df = load_dataset(path)
    df = clean_data(df)
    X, y = separate_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("\nTraining Decision Tree model...")
    y_pred_dt = train_decision_tree_pipeline(X_train, y_train, X_test, y_test)
    print("\nDecision Tree Model Evaluation:")
    evaluate_model(y_test, y_pred_dt)
    
    print("\nTraining Logistic Regression model...")
    y_pred_lr = train_logistic_regression_pipeline(X_train, y_train, X_test, y_test)
    print("\nLogistic Regression Model Evaluation:")
    evaluate_model(y_test, y_pred_lr)

#################################
# Function: main
# Inputs: None
# Output: None
# Description: Runs the Titanic survival prediction pipeline
# Author: Karan Sanjay Jadhav
#################################
def main():
    TitanicCase("titanic.csv")

if __name__ == "__main__":
    main()
