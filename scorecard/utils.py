
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from scorecard.metrics import get_cm_metrics, gini_coefficient


def IQR_outlier_detection(var):
    """
    Detects outliers via the Interquartile rule.
            Parameters:
                    var (array) : 1-d array of numerical data points
            Returns:
                    mask (array): 1-d array of boolean elements indicating whether
                                  datapoint at given position is an outlier
    """
    q1, q3 = np.percentile(var, [25, 75])
    iqr = q3 - q1

    lb = q1 - (iqr * 1.5) 
    ub = q3 + (iqr * 1.5) 

    return (var < lb) | (var > ub)

def get_prediction(y_proba, threshold=0.5):
    """
    Predict whether the estimated sample belongs to class 1 based
    on some given custom threshold
    """
    return y_proba[:,1] >= threshold

def get_C_versus_n_features(x, y):
    """
    Fits multiple LASSO regression models with different hyperparameter C values.
    returns an dataframe with the number of non-zero feature coefficients
    and the evaluation results per C value
    """

    # Search for optimal Lasso regularization hyperparameter C
    c_values = [10, 1, .5, .25, .1, .05, 0.04, 0.03,  0.02, 0.01, 0.001, 0.0001]

    results = []
    for c in c_values:
        # Instantiate and fit LASSO penalized regression model
        model = LogisticRegression(penalty="l1", C=c, solver="liblinear")
        model.fit(x, y)
        
        # Count non-zero coefficients
        n_features = np.sum(model.coef_.ravel() != 0)
        
        # Get estimates
        y_proba = model.predict_proba(x)
        y_pred = get_prediction(y_proba, threshold=0.5)
        
        # Get evaluation metrics 
        cm = confusion_matrix(y, y_pred)
        accuracy, sensitivity, specificity = get_cm_metrics(cm)
        gini = gini_coefficient(y, y_proba[:,1])

        # Save results 
        row_data = {"C": c, "Num_features":n_features, "Accuracy": accuracy, "Sensitivity": sensitivity, "Specificity": specificity, "Gini": gini}
        results.append(row_data)
    return pd.DataFrame.from_records(results)