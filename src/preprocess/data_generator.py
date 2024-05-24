import pandas as pd
from src.utils.constant import scaled_dataset_path, unscaled_dataset_path, dataset_path
import mlrun
import os


@mlrun.handler(outputs=["dataset"])
def data_generator(context):
    """a function which generates the dataset"""
    logger = context.logger
    logger.info("Starting data generation")

    scaled_dataset = pd.read_csv(scaled_dataset_path)
    logger.info("Scaled dataset loaded")

    unscaled_dataset = pd.read_csv(unscaled_dataset_path)
    logger.info("Unscaled dataset loaded")

    try:
        os.mkdir('artifacts')
        logger.info("Artifacts directory created")
    except:
        logger.warning("Failed to create artifacts directory")

    unscaled_dataset.to_parquet(dataset_path)
    logger.info("Saving mobile dataset")

    logger.info("Data generation complete")
    return scaled_dataset
