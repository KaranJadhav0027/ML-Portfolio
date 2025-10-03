# 🧠 Head-Size & Brain-Size Classification

## 📌 Overview
This project explores the relationship between **head size and brain size** to predict **gender-based intelligence classification** using **machine learning models**.  
The pipeline covers **data preprocessing, feature scaling, model training, evaluation, and saving the trained model** for future use.

---

## 🚀 Features
- Load and clean dataset (CSV input).  
- Preprocess and scale features.  
- Train classification models:  
  - Logistic Regression (baseline)  
  - Random Forest Classifier (final model)  
- Evaluate with metrics: Accuracy, Precision, Recall, F1, ROC-AUC.  
- Save trained models (`.joblib`) in `artifacts_headbrain/`.  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, joblib  
- **ML Models:** Logistic Regression, Random Forest  

---

## 📂 Project Structure
<pre>
  HeadBrainClassification/
                            │── HeadBrainLogisticRegression.py # Main pipeline
                            │── HeadBrain.csv # Dataset (to be provided)
                            │── artifacts_headbrain/ # Stores trained model files
</pre>

---
## 📖 Example Output

    Accuracy: 66.67%
    Confusion Matrix:
                     [[21  8]
                      [8 11]]
    ROC-AUC Score: 0.65
    Precision: 0.72
    Recall: 0.72
    F1 Score: 0.72
    Model saved to: artifacts_headbrain/RandomForest_model.joblib
---
## 👨‍💻 Author
**Karan Jadhav**  
  - Developer | Data Structures Enthusiast | System Design Learner
  [📧] (karanjadhav0027@gmail.com)
