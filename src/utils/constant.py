from pathlib import Path

path = Path(__file__).parents[1]
path1 = Path(__file__).parents[2]

# Paths
data = path / 'data'
What_If_Folder = path1 / 'WHAT_IF_TOOL'
scaled_dataset_path = data / 'scaled_dataset_for_training.csv'
unscaled_dataset_path = data / 'unscaled_dataset_for_describe_function.csv'
dataset_path = data / "water_dataset.parquet"
water_dataset = data / "water_potability.csv"
model_local_path = What_If_Folder / 'model/trained_model.h5'
data_for_wit_path = What_If_Folder / "data/test_data_with_labels.csv"
project_name = "water-potability-classifier"

base_dir = path
data_generation_filename = 'preprocess/data_generator.py'
model_trainer_filename = 'model/trainer.py'

feature_range = (0, 1)
insert_position = 0
threshold_for_nan = 0.02
label_col = 'Potability'
train_test_split_size = 0.2
random_state = 42
Model_Name = "water_classifier"
data_gen_function_name = "Water-Data"
describe_run_function_name = "Water-dataset-describe"
model_trainer_function_name = "trainer"
dataset_name = "Water dataset"
