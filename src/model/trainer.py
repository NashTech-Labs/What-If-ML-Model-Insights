from sklearn.model_selection import train_test_split
import tensorflow as tf
import mlrun
import mlrun.frameworks.tf_keras as mlrun_tf
from src.utils.constant import model_local_path, label_col, Model_Name
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@mlrun.handler()
def train(
        dataset: mlrun.DataItem,  # data inputs are of type DataItem (abstract the data source)
        label_column: str = label_col,
        model_name: str = Model_Name
):
    # Get the input dataframe (Use DataItem.as_df() to access any data source)
    df = dataset.as_df()
    logger.info("Dataset is successfully loaded.")

    # Initialize the x & y data
    X = df.drop(label_column, axis=1)
    y = df[label_column]

    # Train/Test split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Log the train and test dataset sizes
    logger.info(f"Train dataset size: {X_train.shape[0]}")
    logger.info(f"Test dataset size: {X_test.shape[0]}")

    # Create a sequential model
    model = tf.keras.Sequential([
        # First dense layer with 200 units and ReLU activation function
        tf.keras.layers.Dense(200, activation='relu'),
        # Dropout layer with a dropout rate of 20%
        tf.keras.layers.Dropout(0.2),
        # Second dense layer with 100 units and ReLU activation function
        tf.keras.layers.Dense(100, activation='relu'),
        # Batch normalization layer
        tf.keras.layers.BatchNormalization(),
        # Third dense layer with 50 units and ReLU activation function
        tf.keras.layers.Dense(50, activation='relu'),
        # Fourth dense layer with 10 units and linear activation function
        tf.keras.layers.Dense(10, activation='linear'),
        # Output layer with 1 unit and sigmoid activation function
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    logger.info("Model Compiled")

    # -------------------- The only line you need to add for MLOps -------------------------
    # Wraps the model with MLOps (test set is provided for analysis & accuracy measurements)
    mlrun_tf.apply_mlrun(model=model, model_name=model_name, x_test=X_test, y_test=y_test)
    # --------------------------------------------------------------------------------------

    # Train the model
    model.fit(X_train, y_train, epochs=50, validation_split=0.2)
    model.save(model_local_path)

    logger.info("Model Trained and Saved")
