# 📊 Data Science: Classification Collection

This repository contains practical implementations of various **Classification** algorithms. The projects cover the entire machine learning pipeline: from data preprocessing and exploratory data analysis (EDA) to model evaluation and hyperparameter tuning.

---

## 🚀 Projects Overview

### 1. 🧬 Breast Cancer Diagnosis (k-NN)
* **Problem:** Classifying breast tumors as Malignant (M) or Benign (B) using the UCI Breast Cancer dataset.
* **Algorithm:** $k$-Nearest Neighbors ($k$-NN).
* **Key Techniques:**
    * Target encoding using `LabelEncoder`.
    * Feature correlation analysis via Heatmaps.
    * Data normalization with `StandardScaler`.
* **Optimization:** Hyperparameter tuning using `GridSearchCV` to find the most effective $k$-value.
* **Metrics:** Accuracy, F1-Score, and Jaccard Index.



### 2. 💊 Drug Recommendation System (Decision Tree)
* **Problem:** A multi-class classification task to predict which of five drugs is suitable for a patient based on Age, Sex, Blood Pressure, and Cholesterol.
* **Algorithm:** Decision Tree Classifier.
* **Key Techniques:**
    * Handling categorical data for medical features.
    * Model visualization using `tree.plot_tree`.
    * Evaluation via `classification_report` and `cross_val_predict`.
* **Hyperparameters:** Experimentation with `max_depth` and `min_samples_leaf` to control tree growth and prevent overfitting.



### 3. 🌲 Ensemble Learning (Random Forest)
* **Problem:** Enhancing the drug recommendation system using ensemble methods.
* **Algorithm:** Random Forest Classifier.
* **Key Results:** Achieved significantly higher stability and accuracy (approx. 97%+) by utilizing multiple decision trees (`n_estimators`).

---

## 🛠 Tech Stack

* **Language:** Python 3.x
* **Data Analysis:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `Scikit-learn`

---

## 📈 Performance Summary

| Model | Task | Metric (Avg) |
| :--- | :--- | :--- |
| **k-NN** | Cancer Diagnosis | ~95% Accuracy |
| **Decision Tree** | Drug Recommendation | ~95% Accuracy |
| **Random Forest** | Drug Recommendation | ~97% Accuracy |

---

## 📂 How to Use
1. Clone the repo: `git clone https://github.com/yourusername/data-science-classification.git`
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the Jupyter Notebooks to see the step-by-step analysis.
