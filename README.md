# Car Price Prediction Project

## Overview
This project focuses on predicting car prices using various machine learning models. The dataset is preprocessed to handle missing values, class imbalance, skewness, and outliers, ensuring that the models trained are both efficient and accurate.

## Table of Contents
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Models](#models)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Installation
Make sure to install the required libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

## Data Preprocessing
The following preprocessing steps were performed on the dataset:

- **Checked for Null Values**: Identified and filled null values to maintain data integrity.
- **Checked for NaN Values**: Detected and addressed NaN values.
- **Checked for Duplicates**: Removed any duplicate entries from the dataset.
- **Checked Class Imbalance**: Handled class imbalance using undersampling and oversampling techniques.

### Before handling class imbalance:

| Class | Count |
|--------|-------|
| 0 | 283253 |
| 1 | 473 |

### After handling class imbalance:

| Class | Count |
|--------|-------|
| 0 | 165122 |
| 1 | 141627 |

- **Checked Skewness**: Analyzed the skewness of the features and applied Yeo-Johnson transformation to reduce skewness.
- **Handled Outliers**: Identified and managed outliers in the dataset.
- **Feature Scaling**: Applied feature scaling to the features, excluding the target variable.
- **Data Utilization**: Due to limited computational power, only 50% of the dataset was used for model training.

## Models
The following models were implemented in the project:

- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree
- Random Forest

## Results

### Logistic Regression
**Training Accuracy**: 93.39%  
**Testing Accuracy**: 93.33%  

#### Confusion Matrix:
```
[[16302   237]
 [ 1808 12328]]
```

#### Classification Report:
```
              precision    recall  f1-score   support

           0       0.90      0.99      0.94     16539
           1       0.98      0.87      0.92     14136
```

### Gaussian Naive Bayes
**Training Accuracy**: 91.92%  
**Testing Accuracy**: 91.93%  

#### Confusion Matrix:
```
[[16129   410]
 [ 2064 12072]]
```

#### Classification Report:
```
              precision    recall  f1-score   support

           0       0.89      0.98      0.93     16539
           1       0.97      0.85      0.91     14136
```

### K-Nearest Neighbors (KNN)
**Training Accuracy**: 99.89%  
**Testing Accuracy**: 99.85%  

#### Confusion Matrix:
```
[[16493    46]
 [    0 14136]]
```

#### Classification Report:
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     16539
           1       1.00      1.00      1.00     14136
```

### Support Vector Classifier (SVC)
**Training Accuracy**: 99.51%  
**Testing Accuracy**: 99.45%  

#### Confusion Matrix:
```
[[16430   109]
 [   61 14075]]
```

#### Classification Report:
```
              precision    recall  f1-score   support

           0       1.00      0.99      0.99     16539
           1       0.99      1.00      0.99     14136
```

### Decision Tree
**Training Accuracy**: 100.00%  
**Testing Accuracy**: 99.93%  

#### Confusion Matrix:
```
[[16516    23]
 [    0 14136]]
```

#### Classification Report:
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     16539
           1       1.00      1.00      1.00     14136
```

### Random Forest
**Training Accuracy**: 100.00%  
**Testing Accuracy**: 99.98%  

#### Confusion Matrix:
```
[[16533     6]
 [    0 14136]]
```

#### Classification Report:
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     16539
           1       1.00      1.00      1.00     14136
```

### Comparison of All Models' Accuracy

| Model Name                | Train Accuracy | Test Accuracy |
|---------------------------|---------------|--------------|
| Logistic Regression       | 93.39%        | 93.33%       |
| Gaussian Naive Bayes      | 91.92%        | 91.93%       |
| K-Nearest Neighbors (KNN) | 99.89%        | 99.85%       |
| Support Vector Classifier | 99.51%        | 99.45%       |
| Decision Tree            | 100.00%       | 99.93%       |
| Random Forest            | 100.00%       | 99.98%       |

## Conclusion
This project demonstrates the effectiveness of various machine learning models in predicting car prices. The preprocessing steps enhanced the dataset quality, leading to improved model performance. Future work could involve exploring additional models and tuning hyperparameters for better accuracy.

## Future Work
Future iterations may explore the following:
- **Hyperparameter Tuning:** Implement advanced techniques like Grid Search and Random Search for model optimization.
- **Feature Engineering:** Investigate new features or transformations to enhance model accuracy.
- **Deployment:** Develop a web application or API to enable real-time predictions using the best-performing model.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.

## Acknowledgments
The Ames Housing dataset is provided by Kaggle. Thanks to the contributors of libraries used in this project, including Pandas, NumPy, and Scikit-learn.

## Requirements
To run this project, install the required packages listed in [requirements.txt](requirements.txt) file

## Contributing
We welcome contributions to this project! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute.


You can install these packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

## How to Run
1. Clone this repository:
    ```bash
    git clone <your-github-repo-url>
    ```
2. Navigate to the project directory:
    ```bash
    cd <your-project-directory>
    ```
3. Run the Jupyter Notebook or Python script:
    ```bash
    jupyter notebook <your-notebook>.ipynb
    ```
    or
    ```bash
    python <your-script>.py
    ```

