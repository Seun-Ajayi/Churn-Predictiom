"""
ChurnModel class Test

Author: 'Seun Ajayi
Date: July 2022
"""

import tempfile
import unittest
from unittest import TestCase
from unittest.mock import patch
import logging
from src.churn_library import ChurnModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from main import CAT_COLS, DROP_COLS

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


class ChurnModelTest(TestCase):
    """ Test the methods in the ChurnModel class."""

    def setUp(self) -> None:
        self.test_csv_file = 'testfile.csv'
        try:
            data = pd.read_csv('tests/test_data.txt')
            encoded_columns = pd.read_csv('tests/encoded_columns.txt')
        except Exception as err:
            logging.error('ERROR: Counld not create fixtures for test')
            raise err
        self.test_data = data
        self.encoded_columns = encoded_columns
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.test_dir.cleanup()
        return super().tearDown()

    def modelTrainingSetUp(self) -> None:
        test_df = self.test_data.drop(columns=CAT_COLS)
        encoded_df = pd.concat([test_df, self.encoded_columns], axis=1)
        self.dataset = ChurnModel(
            model_dir=self.test_dir.name).perform_feature_engineering(encoded_df)

    def modelEvaluationSetUp(self) -> None:
        self.modelTrainingSetUp()
        try:
            self.model = LogisticRegression().fit(
                self.dataset[0], self.dataset[2])
            check_is_fitted(self.model)
        except Exception as err:
            logging.error(
                'ERROR: Counld not create model fixtures for test; model not fitted')

    @patch('src.churn_library.pd.read_csv')
    def test_import(self, mock_read_csv):
        '''
        test data import - this example is completed for you to assist with the other test functions
        '''
        err = None
        mock_read_csv.return_value = self.test_data.drop(
            'Churn', axis=1).copy()
        result = ChurnModel(data_pth=self.test_csv_file).import_data()

        try:
            mock_read_csv.assert_called_with(self.test_csv_file, index_col=0)
            pd.testing.assert_frame_equal(result, self.test_data)
        except AssertionError as err:
            logging.error("Testing import_eda: Error in method result")
            raise err

        # try:
        #     assert result.Churn.dtypes == int64
        # except AssertionError as err:
        #     logging.error("Testing create_target_column: The created target column is not encoded")
        #     raise err

        if err is None:
            logging.info("Testing import_data: SUCCESS")

    @patch('src.churn_library._plot_correlation')
    @patch('src.churn_library._plot_total_trans_ct_dist')
    @patch('src.churn_library._plot_marital_status_hist')
    @patch('src.churn_library._plot_customer_age_hist')
    @patch('src.churn_library._plot_churn_hist')
    def test_eda(
        self,
        mock_churn_plot,
        mock_customer_age_plot,
        mock_marital_status_plot,
        mock_total_trans_ct_plot,
        mock_heat_map
    ) -> None:
        '''
        test perform eda function
        '''

        err = None

        try:
            ChurnModel(
                eda_output_dir=self.test_dir.name).perform_eda(
                self.test_data)
            mock_churn_plot.assert_called_once()
            mock_customer_age_plot.assert_called_once()
            mock_marital_status_plot.assert_called_once()
            mock_total_trans_ct_plot.assert_called_once()
            mock_heat_map.assert_called_once()
        except AssertionError as err:
            logging.error(
                "Testing perform_data: failed in plotting one or more plots")
            raise err

        if err is None:
            logging.info('Testinr perform_eda: SUCCESS')

    def test_encoder_helper(self) -> None:
        '''
        test encoder helper
        '''
        err = None

        try:
            assert all([col in list(self.test_data.columns)
                       for col in CAT_COLS])
        except AssertionError as err:
            logging.error(
                'Testing encoder_helper: One or more categorical columns is not in the fixture dataframe')

        try:
            encoded_df = ChurnModel(
                category_lst=CAT_COLS).encoder_helper(
                self.test_data)
            with pd.option_context('mode.use_inf_as_na', True):
                assert encoded_df[CAT_COLS].isna().any().sum() == 0
            pd.testing.assert_frame_equal(
                encoded_df[CAT_COLS], self.encoded_columns)
        except AssertionError as err:
            logging.error('Testing encoder_helper: Error in class method')

        if err is None:
            logging.info('Testing encoder_helper: SUCCESS')

    @patch('src.churn_library.train_test_split',
           return_value=(None, None, None, None))
    def test_perform_feature_engineering(self, mock_train_test_split) -> None:
        '''
        test perform_feature_engineering
        '''
        err = None
        test_df = self.test_data.drop(columns=CAT_COLS)
        encoded_df = pd.concat([test_df, self.encoded_columns], axis=1)

        try:
            _ = ChurnModel(
                drop_cols=DROP_COLS).perform_feature_engineering(encoded_df)
            mock_train_test_split.assert_called_once()
        except AssertionError as err:
            logging.error(
                'Test perform_feature_engineering: Ensure that train_test_split is called')
            raise err

        if err is None:
            logging.info('Testing perform_feature_engineering: SUCCESS')

    @patch('src.churn_library.joblib')
    def test_train_models(self, mock_joblib) -> None:
        '''
        test train_models
        '''
        err = None
        self.modelTrainingSetUp()

        try:
            training_output = ChurnModel(
                model_dir=self.test_dir.name).train_models(*self.dataset[:-1])
        except Exception as err:
            logging.error('Testing train_models: Error in class method')
            raise err

        try:
            check_is_fitted(training_output['Logistic-Regression']['model'])
        except Exception as err:
            logging.error(
                'Testing train_models: LogisticRegression model not fitted')
            raise err

        try:
            check_is_fitted(
                training_output['RandomForest-Classifier']['model'])
        except Exception as err:
            logging.error(
                'Testing train_models: LogisticRegression model not fitted')
            raise err

        try:
            mock_joblib.dump.assert_called()
        except AssertionError as err:
            logging.error(
                'Testing train_models:  Ensure that the models are saved to the model directory')
            raise err

        try:
            assert isinstance(training_output, dict)
            assert 'Logistic-Regression' in training_output
            assert 'RandomForest-Classifier' in training_output

            assert isinstance(training_output['Logistic-Regression'], dict)
            assert isinstance(
                training_output['Logistic-Regression']['predictions'], tuple)

            assert isinstance(training_output['RandomForest-Classifier'], dict)
            assert isinstance(
                training_output['RandomForest-Classifier']['predictions'], tuple)
        except AssertionError as err:
            logging.error('Testing train_models: Error in model outputs')
            raise err

        if err is None:
            logging.info('Testing train_models: SUCCESS')

    @patch('src.churn_library._shap_summary_plot')
    @patch('src.churn_library._feature_importance_plot')
    @patch('src.churn_library._roc_curve_image')
    @patch('src.churn_library._classification_report_image')
    def test_evaluate_model(
        self,
        mock_classification_report_image,
        mock_roc_curve_image,
        mock_feature_importance_plot,
        mock_shap_summary_plot
    ) -> None:
        '''
        test evaluate_models
        '''
        err = None
        self.modelEvaluationSetUp()

        try:
            ChurnModel(results_output_dir=self.test_dir).evaluate_model(
                self.model,
                'test_model',
                self.dataset[1],
                (None, None),
                (None, None),
                explain=True
            )

            mock_classification_report_image.assert_called_once()
            mock_roc_curve_image.assert_called_once()
            mock_feature_importance_plot.assert_called_once()
            mock_shap_summary_plot.assert_called_once()
        except AssertionError as err:
            logging.error(
                'Testing evaluate_model: Unexpected classmethod error while "explain=True"')
            raise err

        mock_classification_report_image.reset_mock()
        mock_roc_curve_image.reset_mock()
        mock_feature_importance_plot.reset_mock()
        mock_shap_summary_plot.reset_mock()

        try:
            ChurnModel(results_output_dir=self.test_dir).evaluate_model(
                self.model,
                'test_model',
                self.dataset[1],
                (None, None),
                (None, None),
                explain=False
            )

            mock_classification_report_image.assert_called_once()
            mock_roc_curve_image.assert_called_once()
            mock_feature_importance_plot.assert_not_called()
            mock_shap_summary_plot.assert_not_called()
        except AssertionError as err:
            logging.error(
                'Testing evaluate_model: Unexpected classmethod error while "explain=False"')
            raise err

        if err is None:
            logging.info('Testing evaluate_model: SUCCESS')


if __name__ == "__main__":
    with open('./logs/churn_library.log', 'a') as file:
        unittest.main()
