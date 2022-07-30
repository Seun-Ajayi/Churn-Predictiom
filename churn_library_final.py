"""
This Module contains ---

Author: Seun Ajayi
Date: July 2022
"""


from sklearn.metrics import plot_roc_curve, classification_report

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from dataclasses import dataclass
from typing import List, Dict

import argparse
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_parser() -> argparse.ArgumentParser:
    """
    parse command line arguments

    returns:
        parser - ArgumentParser object
    """

    parser = argparse.ArgumentParser(description='MLModel')
    parser.add_argument(
        '--data_pth',
        default="./data/bank_data.csv",
        type=str,
        help='Path to the data e.g: "./data/bank_data.csv"; Default: Present'
    )
    parser.add_argument(
        '--category_lst',
        type=List,
        default=[
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ],
        help='List of columns containing categorical features; Default: Present'
    )
    parser.add_argument(
        '--eda_images_pths_dict',
        type=Dict,
        default={
            "Churn": "./images/eda/churn.jpg",
            "Customer_Age": "./images/eda/customer_age.jpg",
            "Marital_Status": "./images/eda/marital_stasus.jpg",
            "Total_Trans_Ct": "./images/eda/total_trans_ct.jpg",
            "Heatmap": "./images/eda/heatmap.jpg"
        },
        help='Dict containing paths to store EDA images report; Default: Present'
    )
    parser.add_argument(
        '--feature_imp_plot_pth',
        type=str,
        default="./images/results/rfc_model_feature_importance_plot.jpg",
        help='Path to save the feature importance plot; Default: Present'
    )
    parser.add_argument(
        '--model_pths_dict',
        type=Dict,
        default={
            "RandomForestClassifier": "./models/rfc_model.pkl",
            "LogisticRegression": "./models/logistic_model.pkl"
        },
        help='Dict containing paths to save the best performing models; Defaults: Present'
    )
    parser.add_argument(
        '--classification_report_imgs_pths_dict',
        type=Dict,
        default={
            "RandomForestClassifier": "./images/results/rfc_classification_report.jpg",
            "LogisticRegression": "./images/results/lrc_classification_report.jpg"
        },
        help='Dict containing paths to save the classificatin report of the models; Defaults: Present'
    )
    parser.add_argument(
        '--model_results_pths_dict',
        type=Dict,
        default={
            "model_results": "./images/results/model_results_plot.jpg",
            "summary_plot": "images/results/rfc_summary_plot.jpg"
        },
        help='Dict containing paths to save the model reports; Defaults: Present'
    )

    return parser


@dataclass
class Modelling:
    """ This class peforms the end-end machine learning modellind pipeline
        from collecting the pre-processed data, performing EDA, encoding
        categorical features, feature engineering to training and testing the model
        and storing outputs in images files.
    """
    data_pth: str
    category_lst: List
    eda_images_pths_dict: Dict
    feature_imp_plot_pth: str
    model_pths_dict: Dict
    classification_report_imgs_pths_dict: Dict
    model_results_pths_dict: Dict

    def import_data(self):
        '''
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                data: pandas dataframe
        '''
        # read data from csv file into dataframe
        self.data = pd.read_csv(self.data_pth)

        return self.data

    def create_target_column(self):
        '''
        create target column from existing column in dataframe and encoding 

        input:
                data: pandas dataframe

        output:
                data: pandas dataframe
        '''
        # create 'Churn' column from the 'Attrition_Flag' in df and encoding to 0
        # and 1
        self.data['Churn'] = self.data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
            
        return self.data

    def perform_eda(self):
        '''
        perform eda on df and save figures to images folder
        input:
                data: pandas dataframe

        output:
                None
        '''
    
        
        # plot histogram showing 'Churn' distribution and save to file
        plt.figure(figsize=(20, 10))
        self.data['Churn'].hist()
        plt.xlabel('Churn')
        plt.ylabel('Frequency')
        plt.title('Churn Distribution')
        plt.savefig(
            self.eda_images_pths_dict['Churn'],
            bbox_inches='tight',
            dpi=150)

        # plot histogram showing 'Customer_Age' distribution and save to file
        plt.figure(figsize=(20, 10))
        self.data['Customer_Age'].hist()
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Customer_Age Distribution')
        plt.savefig(
            self.eda_images_pths_dict['Customer_Age'],
            bbox_inches='tight',
            dpi=150)

        # plot bar chart showing the normalised 'Marital_Stasus' distribution and
        # save to file
        plt.figure(figsize=(20, 10))
        self.data.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.title('Marital Status Distribution')
        plt.savefig(
            self.eda_images_pths_dict['Marital_Status'],
            bbox_inches='tight',
            dpi=150)

        # plot a displot for the 'Total_Trans_Ct' in the dataframe and save to
        # file
        plt.figure(figsize=(20, 10))
        sns.distplot(self.data['Total_Trans_Ct'])
        plt.title('Total_Trans_Ct Distribution')
        plt.savefig(
            self.eda_images_pths_dict['Total_Trans_Ct'],
            bbox_inches='tight',
            dpi=150)

        # plot an heatmap showing the corr of features in the dataframe and save
        # to file
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            self.data.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.savefig(
            self.eda_images_pths_dict['Heatmap'],
            bbox_inches='tight',
            dpi=150)

    def encoder_helper(self):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category

        input:
                data: pandas dataframe
                category_lst: list of columns that contain categorical features

        output:
                data: pandas dataframe with new columns for
        '''

        # encoded cateorical columns
        for column in self.category_lst:
            lst = []
            groups = self.data.groupby(self.data[column]).mean()['Churn']

            for val in self.data[column]:
                lst.append(groups.loc[val])

            self.data[column] = lst

        return self.data

    def perform_feature_engineering(self):
        '''
        input:
                data: pandas dataframe

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
                train_data: unsplit train data
        '''
        drop_col = ['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag', 'Churn']

        test_data = self.data['Churn']
        self.train_data = self.data.drop(columns=drop_col)

        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.train_data, test_data, test_size=0.3, random_state=42)

        return self.X_train, self.X_test, self.y_train, self.y_test, self.train_data

    def classification_report_image(self,
                                    y_train,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        '''
        # save random forest classifier classification report to image file
        plt.rc('figure', figsize=(5, 5))
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
        # approach
        plt.text(0.01, 1.25, str('Random Forest Train'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Random Forest Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig(
            self.classification_report_image['RandomForestClassifier'],
            bbox_inches='tight',
            dpi=150)

        # save loistic regression classification report to image file
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig(
            self.classification_report_image['LogisticRegression'],
            bbox_inches='tight',
            dpi=150)

    def feature_importance_plot(self, model, X_data):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''
        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        plt.savefig(self.feature_imp_plot_pth, bbox_inches='tight', dpi=150)

    def train_models(self, X_train, X_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression()

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        lrc.fit(X_train, y_train)

        # model evaluation plots
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        plot_roc_curve(
            cv_rfc.best_estimator_,
            X_test,
            y_test,
            ax=ax,
            alpha=0.8)
        plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
        plt.title(
            'ROC curve of Logistic regression and RandonForestClassifier models')
        plt.savefig(
            self.model_results_pths_dict['model_results'],
            bbox_inches='tight',
            dpi=150)

        # summary plots of the random forest model
        explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(
            self.model_results_pths_dict['summary_plot'],
            bbox_inches='tight',
            dpi=150)

        # save best model
        joblib.dump(
            cv_rfc.best_estimator_,
            self.model_pths_dict['RandomForestClassifier'])
        joblib.dump(lrc, self.model_pths_dict['LogisticRegression'])

    def start(self):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        self.data = self.import_data(self.data_pth)

        self.perform_eda(self.data)

        self.encoder_helper(self.data, self.category_lst)

        self.perform_feature_engineering(self.data)

        self.train_models(self.X_train, self.X_test, self.y_train, self.y_test)

        rfc_model = joblib.load('./models/rfc_model.pkl')
        lr_model = joblib.load('./models/logistic_model.pkl')

        y_train_preds_rf = rfc_model.predict(self.X_train)
        y_test_preds_rf = rfc_model.predict(self.X_test)

        y_train_preds_lr = lr_model.predict(self.X_train)
        y_test_preds_lr = lr_model.predict(self.X_test)

        self.classification_report_image(self.y_train,
                                         self.y_test,
                                         y_train_preds_lr,
                                         y_train_preds_rf,
                                         y_test_preds_lr,
                                         y_test_preds_rf)

        self.feature_importance_plot(
            rfc_model,
            self.train_data,
            self.feature_imp_plot_pth)
        print('O pari o, the end lopin cinema')


if __name__ == '__main__':

    parser = get_parser()
    params, _ = parser.parse_known_args()

    model = Modelling(
        data_pth=params.data_pth,
        category_lst=params.category_lst,
        eda_images_pths_dict=params.eda_images_pths_dict,
        feature_imp_plot_pth=params.feature_imp_plot_pth,
        model_pths_dict=params.model_pths_dict,
        classification_report_imgs_pths_dict=params.classification_report_imgs_pths_dic,
        model_results_pths_dict=params.model_results_pths_dict)

    model.start()
