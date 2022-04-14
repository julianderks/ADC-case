from sklearn.metrics import roc_auc_score

def gini_coefficient(y_true, y_proba_c1):
    # Area under receiver operating charateristic cure
    au_roc = roc_auc_score(y_true, y_proba_c1)
    return (2 * au_roc) - 1


def get_cm_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn)/(tn + fp + fn + tp) # Correct label rate
    sensitivity = tp/(tp + fn) # True positive rate
    specificity = tn/(tn + fp) # True negative rate

    return accuracy, sensitivity, specificity
