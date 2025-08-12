# Diabetes Prediction System

**Author**: Karan Jadhav  
**Last Updated**: 09‑08‑2025

---

## Overview
This project implements a K‑Nearest Neighbors-based diabetes prediction pipeline using standardized preprocessing and hyperparameter tuning. It includes an industrial-grade structure, a saved model artifact, and comprehensive evaluation metrics and visualization.

---

##  Repository Structure
<pre>
├── artifacts_sample/
│ └── KNN_pipeline.joblib # Trained pipeline artifact
├── diabetes.csv # Dataset:
└── diabetes_prediction.py # Main script with modular functions
└── README.md # This documentation
</pre>

---

##  Function Breakdown

Below is a summary of each function defined in `DiabetesPredictionKNN.py`, along with its purpose and position in the data-science workflow.

| Function | Description |
|----------|-------------|
| **`load_dataset(path)`** | Loads a CSV dataset into a pandas DataFrame. |
| **`clean_data(df)`** | Removes rows with missing values for clean input. |
| **`separate_features_target(df, target_column='Outcome')`** | Splits DataFrame into features (`X`) and target (`y`). |
| **`scale_features(X)`** | Applies `StandardScaler` to normalize features. |
| **`split_data(x, y, test_size=0.2, random_state=42)`** | Splits data into training and test sets using reproducible seeds. |
| **`train_knn_model(X_train, y_train, k)`** | Trains a standalone KNN classifier with a given number of neighbors. |
| **`find_best_k(X_train, X_test, y_train, y_test, k_range=range(1,25))`** | Runs KNN across multiple `k`, selects the one achieving highest accuracy. |
| **`train_knn_pipeline_and_save(X_train, X_test, y_train, y_test, k, save_path=MODEL_PATH)`** | Constructs a `Pipeline` combining `StandardScaler` and `KNeighborsClassifier`, evaluates it, and saves the fitted pipeline artifact to disk. |
| **`evaluate_model(y_test, y_pred)`** | Prints detailed classification metrics: accuracy, confusion matrix, precision, recall, F1, ROC‑AUC, etc. |
| **`plot_accuracy_curve(accuracies, k_range)`** | Visualizes accuracy scores across tested `k` values. |
| **`DiabetesCase(path)`** | Encapsulates the full pipeline: load, clean, split, K selection, pipeline training/saving, model evaluation, and visualization. |
| **`main()`** | Entry point executing `DiabetesCase()` using `diabetes.csv`. |

--- 

##  Execution Flow 

1. **Load and Clean Data**  
   Reads the dataset and drops incomplete entries to ensure data integrity.

2. **Feature & Target Split**  
   Separates predictors from labels to cleanly define modeling inputs.

3. **Feature Scaling**  
   Standardizes features to ensure consistent scale, boosting model stability.

4. **Train/Test Split**  
   Splits data into train/test subsets with reproducibility via `random_state=42`.

5. **Hyperparameter Tuning** (`find_best_k`)  
   Tests multiple KNN values (`k = 1 → 24`) to identify the best performing neighbor count.

6. **Pipeline Construction & Saving**  
   Wraps preprocessing + modeling into a `Pipeline`, trains it on best `k`, and persists it using `joblib`.

7. **Model Evaluation**  
   Reports classification metrics and confusion matrix for performance transparency.

8. **Visualization**  
   Plots accuracy vs. K to diagnose model behavior and potential overfitting or underfitting.

9. **Modular Design**  
   Functions are isolated by responsibility, making code maintainable and extendable.

10. **Reusable Artifact**  
    The saved pipeline can be reloaded for inference in future environments via `joblib.load()`.

---
