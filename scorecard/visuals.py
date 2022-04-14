import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, confusion_matrix

from .information_value import calculate_IV, get_all_IVs
from .metrics import get_cm_metrics, gini_coefficient
from .utils import get_prediction

params = {
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize" : 14,
    "figure.figsize" : (16, 5)

}
plt.rcParams.update(params) 

def simpleaxis(ax):
    """ Remove top and right borders of plot """

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_evaluation(y_true, y_test_proba):
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    
    # get the values required to plot a ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_test_proba[:,1])

    # plot the ROC curve
    axs[0].plot(fpr, tpr)

    # plot a secondary diagonal line, to plot randomness of model
    axs[0].plot(fpr, fpr, linestyle = '--', color = 'k')
    axs[0].set_xlabel('False positive rate')
    axs[0].set_ylabel('True positive rate')
    axs[0].set_title('ROC curve')

    y_pred = np.argmax(y_test_proba, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    axs[1].set_title('Confusion matrix')

    cm_l = cm.reshape(-1)
    labels = [f"{nominal:,}\n{perc * 100:.2f}%" for nominal, perc in zip(cm_l, cm_l/cm_l.sum())]
    labels = np.array(labels).reshape(2,2) 

    sns.heatmap(cm, annot=labels, fmt="", linewidths=1, linecolor='white', cmap='viridis', ax=axs[1])
    axs[1].set(xlabel='Predicted label', ylabel='True label')

    plt.show()

def plot_feature_WoE_IV(df, feature, ax):
    """ Plots and barplot of combined with a lineplot """
    plot_df = df.melt(id_vars=feature, value_vars=['Distribution 0', 'Distribution 1'])
    sns.barplot(x=feature, y='value', hue='variable', data=plot_df, ax=ax)

    ax2 = ax.twinx()

    df[feature] = df[feature].astype(str)
    sns.lineplot(x=feature, y='WoE', data=df, marker='o', color='crimson', label='WoE', ax=ax2)

    ax.legend(loc=2)
    ax2.legend(loc=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)

    # Set plot title and axes labels
    ax.set(title = f"Information Value = {df['IV'].sum():.04f}",
            ylabel = "Percentage",
            xlabel = feature)

    # Clean up legend
    leg = ax.get_legend()
    for t in leg.texts:
        new_text = t.get_text().replace('Distribution', '%')
        t.set_text(new_text)

def plot_multiple_WoE_IV(df, cols, n_cols=2):
    """ Calls the plot_feature_WoE_IV method for multiple 
        columns in the given dataframe """

    n_rows = (len(cols) // n_cols) + 1

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 3*n_rows))
    axs = axs.flatten()

    for i, col in enumerate(cols):
        iv_df = calculate_IV(df[col], df['Label'])
        plot_feature_WoE_IV(iv_df, col, axs[i])
    for ax in axs[i+1:]:
        ax.set_visible(False) # To remove unused plots

    plt.tight_layout()
    plt.show()


def plot_all_IV(df, ylim_max=0.6):
    """ Plots all a barplot with feature versus information value. 
        Includes visual thresholds of information value levels"""

    iv_tresholds = [0.02, 0.1, 0.3, 0.5]
    iv_tresholds_labels = ["Weak", "Medium", "Strong", "Suspicious"]

    fig, ax = plt.subplots()
    graph = sns.barplot(x=df.index, y='IV', data=df, color='goldenrod')
    ax.set_title('Information Values')

    # Plot and annotate IV tresholds
    p = graph.patches[-1]
    x = ax.get_xlim()[1] * 1.15
    for iv_treshold, treshold_label in zip(iv_tresholds, iv_tresholds_labels):
        ax.axhline(y=iv_treshold, xmin=0, xmax=1.127, linestyle='--', clip_on=False)
        graph.annotate(f"{treshold_label} (>{iv_treshold:.2f})", 
                      (x, iv_treshold+ylim_max*0.02), weight='bold', annotation_clip=False, ha='right')

    plt.xticks(rotation=90)
    plt.ylim(0, ylim_max)

    # Annotate values above ylim_max
    for p in graph.patches:
        y = p.get_height()
        if y >= ylim_max:
            graph.annotate(format(y, '.2f'), 
                        (p.get_x() + p.get_width() / 2., min(p.get_height(), ylim_max)), 
                        ha = 'center', va = 'center', 
                        xytext = (0, -12), 
                        textcoords = 'offset points', weight='bold')
    
    simpleaxis(ax)


def plot_correlation_matrix(df):
    """ Plots the Correlation matrix without the upper diagonal """

    # Calculate (absolute) correlation matrix for the current features
    corr = df.corr().abs()
    # Get index positions of the upper diagonal of the correlation matrix
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True

    # Plot the correlation matrix (without upper diagonal)
    sns.heatmap(corr, annot=True, cmap="viridis", mask=mask, linewidths=.5, fmt=".02f")
    plt.title("Correlation matrix")
    plt.show()


def plot_cm(cm, ax):
    """ Plots the Confusion matrix """

    # Extract annotation labels from CM
    cm_l = cm.ravel()
    labels = [f"{nominal:,}\n{perc * 100:.2f}%" for nominal, perc in zip(cm_l, cm_l/cm_l.sum())]
    labels = np.array(labels).reshape(2,2) 

    # Plot CM
    sns.heatmap(cm, annot=labels, fmt="", linewidths=1, linecolor='white', cmap='viridis', ax=ax)
    ax.set(xlabel='Predicted label', ylabel='True label', title='Confusion matrix')

def plot_roc_curve(fpr, tpr, ax):
    """ Plots the ROC curve """

    # Plot the ROC curve
    ax.plot(fpr, tpr)

    # Plot a secondary diagonal line, to plot randomness of model
    ax.plot(fpr, fpr, linestyle = '--', color = 'k')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve')


def plot_evaluation(y_true, y_test_proba, threshold=0.5):
    """
    Plots the ROC curve and confusion matrix for a given set of labels and prediction probabilities. 
    The confusion matrix is calculated based on the given threshold. Additionally some evaluation metrics are printed

        Parameters:
                y_true (array)      : 1-d array of true class labels
                y_test_proba (array): 2-d array of shape (samples, classes) holding the 
                                      class probabilities
                threshold (float)   : threshold probability when to label a class as 1 
    """

    fig, axs = plt.subplots(1,2, figsize=(14,4))

    # Get and plot the ROC curve values
    fpr_list, tpr_list, _ = roc_curve(y_true, y_test_proba[:,1])
    plot_roc_curve(fpr_list, tpr_list, axs[0])


    # Get and plot the CM values
    y_pred = get_prediction(y_test_proba, threshold=threshold)
    cm = confusion_matrix(y_true, y_pred)
    plot_cm(cm, axs[1])
    accuracy, sensitivity, specificity = get_cm_metrics(cm)
    gini_c = gini_coefficient(y_true, y_test_proba[:,1])

    # Display plot
    plt.show()

    # Print metrics
    print(f"Gini Coefficient:{gini_c:.04f}\n")
    print(f"Accuracy\t:{accuracy:.04f}")
    print(f"Sensitivity\t:{sensitivity:.04f}")
    print(f"Specificity\t:{specificity:.04f}")


def plot_C_vs_num_features(df, ax):
    """ Plots the effect of LASSO hyperparameter C on the number of non-zero features 
        in the logisitic regression model """

    df["C"] = df["C"].astype(str)

    sns.barplot(x="C", y='Num_features', edgecolor='k', data=df, color='grey', ax=ax)
    ax2 = ax.twinx()

    plot_metrics = ["Gini", "Accuracy", "Sensitivity", "Specificity"]

    sns.lineplot(x='C', y='value', hue='variable', marker='o', data=pd.melt(df, id_vars=['C'], value_vars=plot_metrics), ax=ax2)

    ax2.legend(loc=1)


    # Set plot title and axes labels
    ax.set(title = f"LASSO: effect of hyperparameter C",
            xlabel = "Hyperparameter C",
            ylabel = "Number of non-zero feature coefficients")
    ax2.set(ylabel = "Value")


def plot_feature_coeffs(features, coefficients):
    """ Plots a sorted barplot of feature versus feature coefficient value """
    plot_df = pd.DataFrame({"Feature":features , "Coefficient" : coefficients})
    plot_df = plot_df.sort_values(by="Coefficient")
   
    # Plot the feature coefficients
    sns.barplot(x="Feature", y="Coefficient", data=plot_df, edgecolor="k", color="forestgreen")
    plt.xticks(rotation=45)
    plt.title("Feature coefficents")

    plt.show()