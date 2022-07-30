"""
CLI for Churn Prediction

Author: 'Seun Ajayi
Date: July 2022
"""

import logging
import argparse
from typing import List
from src.churn_library import ChurnModel


logging.root.setLevel(logging.INFO)

DEFAULT_DATA_PTH = './data/bank_data.csv'

CAT_COLS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

DROP_COLS = [
    'CLIENTNUM',
    'Attrition_Flag',
    'Churn']

PARAMS_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}


def get_parser() -> argparse.ArgumentParser:
    """
    parse command line arguments

    returns:
        parser - ArgumentParser object
    """

    parser = argparse.ArgumentParser(description='ChurnModel')
    parser.add_argument(
        '--data_pth',
        default=DEFAULT_DATA_PTH,
        type=str,
        help='Path to the data e.g: "./data/bank_data.csv"'
    )

    parser.add_argument(
        '--category_cols',
        type=List,
        default=CAT_COLS,
        help='List of columns containing categorical features'
    )

    parser.add_argument(
        '--drop_cols',
        type=List,
        default=DROP_COLS,
        help='List of columns to be dropped from training dataframe'
    )

    parser.add_argument(
        '--eda_output_dir',
        default='./images/eda',
        type=str,
        help='Path to save EDA plots'
    )

    parser.add_argument(
        '--results_output_dir',
        default='./images/results',
        type=str,
        help='Path to save model evaluation plots'
    )

    parser.add_argument(
        '--model_dir',
        default='./models',
        type=str,
        help='Path to save models'
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    params, _ = parser.parse_known_args()

    model = ChurnModel(
        data_pth=params.data_pth,
        eda_output_dir=params.eda_output_dir,
        results_output_dir=params.results_output_dir,
        model_dir=params.model_dir,
        category_lst=params.category_cols,
        drop_cols=params.drop_cols
    )

    logging.info("--------------------------------------")
    logging.info("Starting modelling processes...")
    logging.info("--------------------------------------")

    logging.info("importing data...")
    logging.info("--------------------------------------")
    data = model.import_data()

    logging.info("Data successfully imported...")
    logging.info("--------------------------------------")

    logging.info("Performing Exploratory Data Anaysis...")
    logging.info("--------------------------------------")
    model.perform_eda(data)

    logging.info("Successfully saved EDA plots to file..")
    logging.info("--------------------------------------")

    logging.info("Encoding categorical features...")
    logging.info("--------------------------------------")
    data = model.encoder_helper(data)

    logging.info("Encoding Completed...")
    logging.info("--------------------------------------")

    logging.info("Performing Feature Engineering...")
    logging.info("--------------------------------------")
    X_train, X_test, y_train, y_test = model.perform_feature_engineering(
        data, test_size=0.3)

    logging.info("Feature Engineering completed...")
    logging.info("--------------------------------------")

    logging.info("Starting model training...")
    logging.info("-----This would take a while...-------")
    logging.info("--------------------------------------")
    models = model.train_models(X_train, X_test, y_train, PARAMS_GRID)

    logging.info("models successfully trained and saved...")
    logging.info("--------------------------------------")

    logging.info("Evaluating Logistic Regression model...")
    logging.info("--------------------------------------")
    model.evaluate_model(
        model=models['Logistic-Regression']['model'],
        model_name='Logistic-Regression',
        X_test=X_test,
        y_original=(y_train, y_test),
        y_predicted=models['Logistic-Regression']['predictions'],
        explain=False
    )
    logging.info("Evaluating RandomForest Classifier...")
    logging.info("----This would also take some time----")
    logging.info("--------------------------------------")
    model.evaluate_model(
        model=models['RandomForest-Classifier']['model'],
        model_name='RandomForest-Classifier',
        X_test=X_test,
        y_original=(y_train, y_test),
        y_predicted=models['RandomForest-Classifier']['predictions'],
        explain=True
    )
    logging.info("Evaluation completed...")
    logging.info("--------------------------------------")
    logging.info("-----Modelling process completed------")
    logging.info("--------------THE END-----------------")
