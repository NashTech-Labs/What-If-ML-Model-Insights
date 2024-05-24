import logging
from src.utils.helpers import (drop_columns, load_dataset, replace_nan_with_median, scale_features,
                               modify_dataset, evaluate_all_correlations,
                               get_drop_columns_list_based_on_correlation)


class DataPreprocessor:
    def __init__(self, train_data, scaled_dataset_path, unscaled_dataset_path, data_for_wit_path, target_col,
                 feature_range, insert_position, threshold_for_nan):
        self.file_path = train_data
        self.scaled_dataset_path = scaled_dataset_path
        self.unscaled_data = unscaled_dataset_path
        self.wit_data_path = data_for_wit_path
        self.target_col = target_col
        self.feature_range = feature_range
        self.insert_position = insert_position
        self.threshold_for_nan = threshold_for_nan

    def preprocess_data(self):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        try:
            logger.info("Loading dataset...")
            df = load_dataset(self.file_path)

            logger.info("Replacing NaN values with median...")
            df = replace_nan_with_median(df=df, threshold=self.threshold_for_nan)

            logger.info("Evaluating correlations...")
            correlation_results = evaluate_all_correlations(df)
            dropping_columns = get_drop_columns_list_based_on_correlation(correlation_results=correlation_results)

            logger.info("Dropping columns...")
            df = drop_columns(data_frame=df, dropping_columns=dropping_columns)
            logger.info("Saving Unscaled dataset for describe function...")
            df.to_csv(self.unscaled_data, index=False)
            logger.info("Scaling numerical variables...")
            scaled_df = df.copy()
            scaled_df = scale_features(scaled_df, target_col=self.target_col, feature_range=self.feature_range)

            logger.info("Modifying dataset to be compatible with WIT...")
            modified_dataset = modify_dataset(df=scaled_df, target_col=self.target_col,
                                              insert_position=self.insert_position)

            logger.info("Creating dataset for WIT...")
            train_size = int(len(modified_dataset) * 0.8)
            test_data_with_labels = modified_dataset[train_size:]
            test_data_with_labels.to_csv(self.wit_data_path,
                                         index=False)
            logger.info("Dataset for WIT Saved...")

            logger.info("Saving preprocessed dataset...")
            modified_dataset.to_csv(self.scaled_dataset_path, index=False)
            logger.info("Preprocessed dataset saved successfully.")

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise e
