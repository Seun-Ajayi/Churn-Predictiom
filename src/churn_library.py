"""
Base Class for Churn prediction

Author: 'Seun Ajayi
Date: July 2022
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

from pandas import DataFrame
from pandas import Series

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import joblib
import pandas as pd
import seaborn as sns

from src.plots import _plot_churn_hist
from src.plots import _plot_customer_age_hist
from src.plots import _plot_marital_status_hist
from src.plots import _plot_total_trans_ct_dist
from src.plots import _plot_correlation

from src.plots import _classification_report_image
from src.plots import _feature_importance_plot
from src.plots import _roc_curve_image
from src.plots import _shap_summary_plot


sns.set()


@dataclass
class ChurnModel:
    """ This class peforms the end-end machine learning modellind pipeline
        from collecting the pre-processed data, performing EDA, encoding
        categorical features, feature engineering to training and testing the model
        and storing outputs in images files.

        Intialization inputs:
                data_pth: dataset path
                eda_output_dir: directory to store EDA plots
                results_output_dir: directory to store model evaluation results
                model_dir: directory to trained models
                category_lst: List containing categorical variables
                drop_cols: List containing features to drop from train data
    """

    data_pth: str = None
    eda_output_dir: str = 'images/eda'
    results_output_dir: str = 'images/results'
    model_dir: str = './models'
    category_lst: List = None
    drop_cols: List = None

    def import_data(self) -> DataFrame:
        '''
        returns dataframe for the csv found at pth and create target feature

        input:
                self.data_pth: a path to the csv
        output:
                data: pandas dataframe
        '''
        # import data from csv file
        data = pd.read_csv(self.data_pth, index_col=0)

        # create target column; 'Churn'
        # from existing column; 'Attrition_Flag' in dataframe and encoding
        data['Churn'] = data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        return data

    def perform_eda(
            self,
            data: DataFrame) -> DataFrame:
        '''
        perform eda on df and save figures to images folder
        input:
                data: pandas dataframe
                self.output_dir: folder path to save EDA plots

        output:
                None
        '''
        temp = data.copy()
        os.makedirs(self.eda_output_dir, exist_ok=True)

        _plot_churn_hist(
            temp,
            os.path.join(
                self.eda_output_dir,
                'Churn-Histogram.jpg'))
        _plot_customer_age_hist(temp, os.path.join(
            self.eda_output_dir, 'Customer-Age-Histogram.jpg'))
        _plot_marital_status_hist(temp, os.path.join(
            self.eda_output_dir, 'Marital-Status-Histogram.jpg'))
        _plot_total_trans_ct_dist(temp, os.path.join(
            self.eda_output_dir, 'Total-Trans-Ct-DistPlot.jpg'))
        _plot_correlation(
            temp,
            os.path.join(
                self.eda_output_dir,
                'Heatmap.jpg'))

    def encoder_helper(self, data: DataFrame) -> DataFrame:
        '''
        helper method to turn each categorical column into a new column with
        propotion of churn for each category

        input:
                data: pandas dataframe
                self.category_lst: list of columns that contain categorical features

        output:
                data: pandas dataframe with new columns for
        '''

        if self.category_lst is None:
            self.category_lst = [
                'Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category'
            ]

        for column in self.category_lst:
            lst = []
            groups = data.groupby(data[column]).mean()['Churn']

            for val in data[column]:
                lst.append(groups.loc[val])

            data[column] = lst

        return data

    def perform_feature_engineering(
            self,
            data: DataFrame,
            test_size: float = 0.3) -> tuple:
        '''
        Feature selection and 'train_test_split' occurs here

        input:
                data: pandas dataframe
                self.drop_cols: features no valid for the train data
                test_size: percentage of test data for 'train_test_split'

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''

        if self.drop_cols is None:
            self.drop_cols = [
                'CLIENTNUM',
                'Attrition_Flag',
                'Churn']

        test_data = data['Churn']
        train_data = data.drop(columns=self.drop_cols)

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            train_data, test_data, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_models(
            self,
            X_train,
            X_test,
            y_train,
            params_grid: Dict = None):
        '''
        train, store model results: images + scores, and store models
        input:
                self.model_dir: directory to save models
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
                params_grid: RandomForestClassifier grid serach parameters
        output:
                None
        '''
        # grid search
        rfc = RandomForestClassifier(random_state=42)

        if params_grid is None:
            params_grid = {
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [4, 5, 100],
                'criterion': ['gini', 'entropy']
            }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=params_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        lrc = LogisticRegression()
        lrc.fit(X_train, y_train)

        y_train_preds_rf = cv_rfc.predict(X_train)
        y_test_preds_rf = cv_rfc.predict(X_test)

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        # save best model
        joblib.dump(
            cv_rfc.best_estimator_,
            os.path.join(self.model_dir, 'rfc.pkl'))
        joblib.dump(lrc, os.path.join(self.model_dir, 'lrc.pkl'))

        return {
            'Logistic-Regression':
            {
                'model': lrc,
                'predictions': (y_train_preds_lr, y_test_preds_lr)
            },
            'RandomForest-Classifier':
            {
                'model': cv_rfc.best_estimator_,
                'predictions': (y_train_preds_rf, y_test_preds_rf)
            }
        }

    def evaluate_model(
        self,
        model: Any,
        model_name: str,
        X_test: DataFrame,
        y_original: Tuple[Series, Series],
        y_predicted: Tuple[Series, Series],
        explain: bool = False
    ):
        """
        Evaluate fitted model. Creates and saves the ffg plots:
            - Classification Report
            - ROC Curve
            - Other plots may be saved. See `explain`

        inputs:
            model: Fitted model to be evaluated
            model_name: Used to name image file
            X_test: Test Dataframe of X values
            y_original: Tuple(y_train, y_test)
            y_predicted: Tuple(predicted_y_train, predicted_y_test)
            self.output_dir: Output directory for plots
            explain: If True, two additional plots are created"
                - Feature Importance Plot
                - SHAP Summary Plot

        outputs:
                None
        """

        _, y_test = y_original

        _classification_report_image(
            model_name,
            y_original,
            y_predicted,
            output_dir=self.results_output_dir)
        _roc_curve_image(
            model,
            model_name,
            X_test,
            y_test,
            output_dir=self.results_output_dir)
        if explain:
            _feature_importance_plot(
                model, X_test, output_dir=self.results_output_dir)
            _shap_summary_plot(
                model,
                X_test=X_test,
                output_dir=self.results_output_dir)
