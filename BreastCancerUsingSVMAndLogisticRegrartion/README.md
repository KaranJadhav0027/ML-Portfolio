# 🎗️ Breast Cancer Detection using ML (Logistic Regression & SVM)

## 📌 Overview
This project implements **machine learning classifiers** for **Breast Cancer Detection**, comparing the performance of **Logistic Regression** and **Support Vector Machine (SVM)**.  
The dataset is cleaned, preprocessed, and fed into ML pipelines with scaling and classification. Models are saved for reuse.

---

## 🚀 Features
- Load and preprocess breast cancer dataset.  
- Encode categorical diagnosis (`M` = Malignant, `B` = Benign).  
- Train models using:  
  - Logistic Regression  
  - Support Vector Machine (RBF kernel)  
- Evaluate with metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC.  
- Save trained models in `artifacts_sample/`.  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** pandas, scikit-learn, joblib  
- **ML Models:** Logistic Regression, SVM (RBF kernel)  

---

## 📂 Project Structure

<pre>
  BreastCancerDetection/
                        │── BreastCancerSVMAndLR.py # Main pipeline
                        │── breast_cancer.csv # Dataset (to be provided)
                        │── artifacts_sample/ # Stores trained model files
</pre>

---
## 📖 Example Output

      Training Logistic Regression model...
      Logistic Regression Accuracy: 95.61
      Confusion Matrix:
      [[107   3]
       [  2  57]]
      ROC-AUC Score: 0.97
      Precision: 0.95
      Recall: 0.97
      F1 Score: 0.96
      
      Training Support Vector Machine model...
      SVM Accuracy: 96.49
      Confusion Matrix:
                        [[108   2]
                         [2  57]]
      ROC-AUC Score: 0.98
      Precision: 0.96
      Recall: 0.97
      F1 Score: 0.97

  ---
  ## 👨‍💻 Author
**Karan Jadhav**  
  - Developer | Data Structures Enthusiast | System Design Learner
  - [📧] (karanjadhav0027@gmail.com)
