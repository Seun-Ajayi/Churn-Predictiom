"""
This Module contains ---

Author: Seun Ajayi
Date: July 2022
"""

import os
from sklearn.metrics import plot_roc_curve, classification_report
import matplotlib.pyplot as plt
import shap
import numpy as np
import seaborn as sns

sns.set()


def _plot_churn_hist(data, img_pth):
    """ plot histogram showing 'Churn' distribution and save to file.

    inputs:
            data: Dataframe to e explored
            img_pth: img destination path

    outputs:
            None
    """

    plt.figure(figsize=(20, 10))
    data['Churn'].hist()
    plt.xlabel('Churn')
    plt.ylabel('Frequency')
    plt.title('Churn Distribution')
    plt.savefig(
        img_pth,
        bbox_inches='tight',
        dpi=150)
    plt.close()


def _plot_customer_age_hist(data, img_pth):
    """ plot histogram showing 'Customer_Age' distribution and save to file.

    inputs:
            data: Dataframe to e explored
            img_pth: img destination path

    outputs:
            None
    """

    plt.figure(figsize=(20, 10))
    data['Customer_Age'].hist()
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Customer_Age Distribution')
    plt.savefig(
        img_pth,
        bbox_inches='tight',
        dpi=150)
    plt.close()


def _plot_marital_status_barchart(data, img_pth):
    """ plot bar chart showing the normalised 'Marital_Stasus' distribution and
    save to file

    inputs:
            data: Dataframe to e explored
            img_pth: img destination path

    outputs:
            None
    """

    plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Marital Status Distribution')
    plt.savefig(
        img_pth,
        bbox_inches='tight',
        dpi=150)
    plt.close()


def _plot_total_trans_ct_dist(data, img_pth):
    """ plot a displot for the 'Total_Trans_Ct' in the dataframe and save to
    file

    inputs:
            data: Dataframe to e explored
            img_pth: img destination path

    outputs:
            None
    """

    plt.figure(figsize=(20, 10))
    sns.distplot(data['Total_Trans_Ct'])
    plt.title('Total_Trans_Ct Distribution')
    plt.savefig(
        img_pth,
        bbox_inches='tight',
        dpi=150)
    plt.close()


def _plot_correlation(data, img_pth):
    """ plot an heatmap showing the corr of features in the dataframe and save
    to file

    inputs:
            data: Dataframe to e explored
            img_pth: img destination path

    outputs:
            None
    """

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        data.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(
        img_pth,
        bbox_inches='tight',
        dpi=150)
    plt.close()


def _feature_importance_plot(
        model,
        X_data,
        output_dir: str = 'images/results'):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_dir: output directory to store the figure

    output:
            NoneS
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 20))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(
        os.path.join(
            output_dir,
            'Feature-Importance-Plot.jpg'),
        bbox_inches='tight',
        dpi=150)
    plt.close()


def _roc_curve_image(model, model_name, X_test, y_test,
                     output_dir: str = 'images/results'):
    """
    Plots Receiver-Operating-Characteristic

    inputs:
        model: Fitted model to create plot for
        model_name: Used to name image file
        X_test: Test Dataframe of X values
        y_test: Test Series of y values
        output_dir: Output directory for plot

    output:
            None
    """

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        model,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)

    plt.title(
        f'{model_name} ROC-Curve')
    plt.savefig(
        os.path.join(output_dir, f'{model_name}-ROC-Curve.jpg'),
        bbox_inches='tight',
        dpi=150)
    plt.close()


def _classification_report_image(
    model_name: str,
    y_original: tuple,
    y_prediction: tuple,
    output_dir: str = 'images/results'
):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            model_name: Used to name image file
            y_original:  Tuple(y_train, y_test)
            y_predicted: Tuple(predicted_y_train, predicted_y_test)
            output_dir: Output directory for plot

    output:
            None
    '''
    y_train, y_test = y_original
    y_train_preds, y_test_preds = y_prediction

    plt.rc('figure', figsize=(7, 7))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
    # approach
    plt.text(0.01, 1.25, str(f'{model_name} Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f'{model_name}-Classification-Report.jpg'),
        bbox_inches='tight',
        dpi=150)
    plt.close()


def _shap_summary_plot(model, X_test, output_dir: str = 'images/results'):
    """ summary plot with SHAP

    input:
                model: model object containing feature_importances_
                X_test: pandas dataframe of X test values
                output_dir: output directory to store the figure

        output:
                None
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(
        os.path.join(output_dir, 'Shap-Summary-Plot.jpg'),
        bbox_inches='tight',
        dpi=150)
    plt.close()
