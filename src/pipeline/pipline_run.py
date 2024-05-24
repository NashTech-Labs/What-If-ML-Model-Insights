from src.preprocess.data_prep import DataPreprocessor
import logging
from src.utils.constant import (water_dataset, scaled_dataset_path, unscaled_dataset_path, data_for_wit_path, label_col,
                                feature_range, insert_position, threshold_for_nan, project_name,
                                data_generation_filename, model_trainer_filename, data_gen_function_name,
                                describe_run_function_name, dataset_path, dataset_name, model_trainer_function_name,
                                base_dir)
from src.utils.helpers import load_project
import mlrun
import os

# Set up logging
logging.basicConfig(level=logging.INFO)


class MlrunPipeline:
    def __init__(self):
        """
        Initializes the MLflow Pipeline with DataPreprocessor and NeuralNetwork objects.
        """
        self.prep_data = DataPreprocessor(water_dataset, scaled_dataset_path, unscaled_dataset_path, data_for_wit_path, label_col,
                                          feature_range, insert_position, threshold_for_nan)
        self.data_gen_func_path = os.path.join(base_dir, data_generation_filename)
        self.model_train_func_path = os.path.join(base_dir, model_trainer_filename)

    def runner_for_mlrun_and_pipeline(self):
        try:
            logging.info("Executing pipeline run")

            logging.info("Step: Data preprocessing.")
            project = load_project(project_name)
            self.prep_data.preprocess_data()

            logging.info("Step: Data Generation and saving it as parquet. ")
            project.set_function(
                self.data_gen_func_path,
                name=data_gen_function_name,
                kind="job",
                image="mlrun/mlrun",
                handler="data_generator",
            )
            project.save()
            gen_data_run = project.run_function(data_gen_function_name, local=True)
            logging.info("Step: Data Generation completed. ")

            logging.info("Step: Data Visualization")
            describe_func = mlrun.import_function("hub://describe")
            describe_func.run(
                name=describe_run_function_name,
                handler='analyze',
                inputs={"table": f"{dataset_path}"},
                params={"name": dataset_name, "label_column": label_col},
                local=True
            )

            logging.info("Step: Model Training")
            project.set_function(
                self.model_train_func_path, name=model_trainer_function_name, kind="job",
                image="mlrun/mlrun", handler="train"
            )
            project.run_function(
                model_trainer_function_name,
                inputs={
                    "dataset": gen_data_run.outputs["dataset"]
                },
                local=True,
            )
            logging.info("Step: Model Training Completed.")

        except Exception as e:
            logging.error(f"An error occurred during pipeline execution: {str(e)}")
            raise
