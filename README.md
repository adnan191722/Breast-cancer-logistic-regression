# Breast Cancer Diagnosis Prediction using Logistic Regression

Assignment project for *Introduction to Artificial Intelligence and Data Science Engineering*.

## Research Problem
Early and accurate diagnosis of breast cancer is critical. This project builds a **Logistic Regression** model (binary classification) that predicts whether a breast tumour is **malignant** or **benign** from measurements of cell nuclei computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## Dataset
- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (also bundled with scikit-learn as `sklearn.datasets.load_breast_cancer`)
- **Size:** 569 samples, 30 numeric features (radius, texture, perimeter, area, smoothness, etc. - mean, standard error and "worst" values)
- **Target:** Malignant (212 samples) vs Benign (357 samples)

## Methodology
1. **Data loading:** loaded with scikit-learn.
2. **Cleaning / preprocessing:** checked for missing values (none) and duplicates (none); split into 80% train / 20% test (stratified, `random_state=42`); standardized features with `StandardScaler` (fit on training data only to avoid data leakage).
3. **Model:** `sklearn.linear_model.LogisticRegression` (`max_iter=1000`).
4. **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC AUC, confusion matrix on the held-out test set.

## Results (test set, 114 samples)

| Metric | Score |
|---|---|
| Accuracy | 0.9825 |
| Precision | 0.9861 |
| Recall | 0.9861 |
| F1-score | 0.9861 |
| ROC AUC | 0.9954 |

The model classifies about 98% of tumours correctly, with very few false positives or false negatives.

### Visualizations

**Confusion Matrix**

![Confusion Matrix](confusion_matrix.png)

**ROC Curve**

![ROC Curve](roc_curve.png)

**Top 10 Influential Features**

![Feature Importance](feature_importance.png)

## Repository Structure
```
.
├── breast_cancer_logistic_regression.py   # full analysis code
├── requirements.txt
├── results.txt                            # metrics output
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python breast_cancer_logistic_regression.py
```
Graphs are saved to the `images/` folder.

## Conclusion
Logistic Regression, even though it is a simple and interpretable model, performs very well on this dataset. The most influential features are worst texture, radius error, worst concave points, worst area and worst radius. Limitation: the dataset is small, so results should be validated with cross-validation and larger datasets before any real-world use.
