# End-to-End Machine Learning Execution Pipeline with WHAT IF TOOL and MLrun

## Introduction
In this project, we develop an end-to-end machine learning execution pipeline that integrates the What-If Tool (WIT) and MLrun. The goal is to leverage both tools to gain insights into model behavior, track experiments, and manage the machine learning lifecycle effectively. We focus on training a TensorFlow model on the COMPAS dataset, a dataset used in the criminal justice system to predict the likelihood of recidivism.

### WHAT IF TOOL
The What-If Tool, developed by Google's PAIR team, is an open-source tool designed to help machine learning practitioners understand and interpret model behavior. It offers features such as data exploration, model understanding, counterfactual reasoning, fairness analysis, and model comparison.

### MLrun
MLrun, developed by Iguazio, is an open-source platform for managing the end-to-end machine learning lifecycle. It provides capabilities for experiment tracking, model management, model serving, and orchestration.

## Steps in the end-to-end pipeline - 
- Data Preprocessing
- Data Visualization on the MLRun UI using describe function from MlRun function hub.
- Model Building and Training
- Use MLrun to track the experiments and also to visualize the model performance on the MLRun UI.
- Log the model parameters and metrics on MLrun along with the model itself in the artifact.
- Visualize the model behaviour using the What If Tool.
- Make predictions from the model using Flask app.

## Use Case: Water Dataset
This dataset is crucial for understanding the quality of water sources and determining whether they are safe for consumption. The use case for this dataset is primarily in environmental science, public health, and water management. By analyzing the data, researchers and policymakers can identify patterns and correlations between various water quality parameters and their potability. This information is vital for making informed decisions about water treatment, policy, and public health interventions. The dataset likely includes a range of parameters such as pH levels, presence of contaminants, and other chemical indicators that can influence water safety.
## To run the repository and execute the pipeline, follow these instructions:

### Prerequisites
1. **Ensure you have Python installed on your system(version 3.9).**
2. **Clone this gitHub repo:**
    ```bash
   git clone https://github.com/username/repository-name.git
3. **Install the required dependencies listed in `requirements.txt` using the following command:**


### Running the Pipeline
1. Navigate to the root directory of the repository in your terminal.
2. Execute the `main.py` script to run the pipeline:
   ```
   python main.py
   ```

### Pipeline Execution Steps
1. **Data Preprocessing:** The pipeline will preprocess the data using the `preprocess/data_prep.py` module.
2. **Data Generation:** Generates a dataset in parquet format and also returns the dataset as output which will be fed as input in the next function using `preprocess/data.py` module.
3. **Data Visualization:** Visualizes the dataset on the MLRun UI using the `describe` function from the function hub, it will show various plots about the data like Scatter plot, Violin Plot and many more.
4. **Model Training:** The pipeline will train a TensorFlow model using the `model/trainer.py` module.
5. **Logging with MLRun:** During model training, MLRun will log experiment parameters, model configurations,evaluation metrics and model. You can view these logs in the MLRun UI.
6. **Saving Model and Data:** After training, the pipeline will save the trained model in the `What_If_Tool/model` directory (`trained_model.h5`) and the preprocessed data in the `What_If_Tool/data` directory (`test_data_with_labels.csv`).
7. **Notebook Integration:** You can use the saved model (`trained_model.h5`) and data (`test_data_with_labels.csv`) in the provided `WIT_Notebook.ipynb` Jupyter notebook for further analysis and visualization using the What-If Tool.
   
   **Visualization with What-If Tool:**
   1. Open the `WIT_Notebook.ipynb` notebook in Google Colab or any other Jupyter notebook environment.
   2. Load the saved model (`trained_model.h5`) and data (`test_data_with_labels.csv`) into the notebook.
   3. Use the What-If Tool within the notebook to visualize the model's behavior, analyze predictions, and explore fairness and bias considerations.

8. **Making predictions:** Navigate to the root directory and run the file :
   ```
   python flask_app.py
   ```
   Then the flask app will be start running on the localhost port 8000, you can make predictions by sending POST requests to the port using below commands or by using Postman -
   ```shell
   curl -X POST -H "Content-Type: application/json" -d '[[0.5026251502738249,0.5711390082822001,0.3360964634654397,0.5438913403667128,0.6803852067559715,0.6694394775245862,0.3134016505013297,0.6997531312286739,0.2860910154342297]]' http://localhost:8000/predict
   ```
