import pandas as pd
import os
import mlrun
import logging
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from itertools import combinations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path):
    """
        Loads the dataset from the specified file path using pandas.

        Args:
            file_path (str): Path to the dataset file.

        Returns:
            DataFrame: DataFrame containing the loaded dataset.
        """
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File not found at the specified path.")
    try:
        dataset = pd.read_csv(file_path)
        if dataset.empty:
            raise ValueError("File is empty.")
        return pd.DataFrame(dataset)
    except Exception as e:
        raise f"Error loading dataset: {str(e)}"


def drop_columns(data_frame, dropping_columns):
    """
        Drops specified columns from the given dataframe.

        Args:
            data_frame (DataFrame): DataFrame from which columns are to be dropped.
            dropping_columns (list): List of column names to be dropped.

        Returns:
            DataFrame: DataFrame after dropping specified columns.
        """
    try:
        validate_columns_exist(data_frame, dropping_columns)
        return data_frame.drop(dropping_columns, axis=1)
    except Exception as e:
        raise KeyError(f"Exception raised at drop_columns:str{e}")


def validate_columns_exist(df, columns):
    """
        Validate if the specified columns exist in the DataFrame.

        Args:
            df (DataFrame): DataFrame to validate columns against.
            columns (list): List of column names to validate.

        Raises:
            KeyError: If any of the specified columns do not exist in the DataFrame.
        """
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise KeyError(f"Columns {', '.join(missing_columns)} do not exist in the DataFrame.")


def load_project(project_name):
    """
        Loading the project

        raise HttpStatusError for Mlrun if found
        :raise Exception if unknown exception occurs
        """
    try:
        return mlrun.get_or_create_project(f"{project_name}", context="./", user_project=True)
    except mlrun.errors.MLRunHTTPStatusError as error1:
        error1.strerror = 'Connection to port not found:'
        raise error1


def replace_nan_with_median(df, threshold=None):
    """
        Calculates the number of NaN values in a dataset and replaces NaN with the median
        of the column if the percentage of NaN values in that column is higher than or
        equal to the specified threshold.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            threshold (float): The threshold percentage for replacing NaN with median.
                               Default is 0.5 (50%).

        Returns:
            pandas.DataFrame: The DataFrame with NaN values replaced by the column median
                              if the percentage of NaN values in that column exceeds the
                              threshold.
    """
    try:
        # Calculate the total number of values in the DataFrame
        total_values = df.size

        # Calculate the number of NaN values in each column
        nan_counts = df.isna().sum()

        # Calculate the percentage of NaN values in each column
        nan_percentages = nan_counts / len(df) * 100

        # Replace NaN with median for columns exceeding the threshold
        for col, perc in nan_percentages.items():
            if perc >= threshold * 100:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        logger.info(
            f"Number of NaN values in the dataset: {df.isna().sum().sum()}/{total_values} \
            ({(df.isna().sum().sum() / total_values) * 100:.2f}%)")

        return df

    except Exception as e:
        raise f"An error occurred in the replace_nan_with_median function: {str(e)}"


def scale_features(df, target_col, feature_range=None):
    """
    Scales the feature columns of a DataFrame using MinMaxScaler and adds the
    target column back to the scaled DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing features and target.
        target_col (str): The name of the target column.
        feature_range (tuple, optional): The range to which the feature values
                                         will be scaled. If not provided, the
                                         default range (0, 1) will be used.

    Returns:
        pandas.DataFrame: A new DataFrame with scaled feature columns and the
                          original target column.
    """
    try:
        # Separate the feature columns (X) and the target column (y)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Initialize the MinMaxScaler
        if feature_range is None:
            scaler = MinMaxScaler()
        else:
            scaler = MinMaxScaler(feature_range=feature_range)

        # Fit the scaler to the feature columns and transform them
        X_scaled = scaler.fit_transform(X)

        # Create a new DataFrame with the scaled feature columns
        scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Add the target column back to the scaled DataFrame
        scaled_df[target_col] = y

        return scaled_df

    except Exception as e:
        raise f"An error occurred in the scale_features function: {str(e)}"


def modify_dataset(df, target_col, insert_position):
    """
    Modifies a DataFrame by extracting the target column, scaling the feature
    columns, and inserting the target column back at the desired position.
    The modified DataFrame is then saved to a specified output file.

    Args:
        df (pandas.DataFrame): The input DataFrame containing features and target.
        target_col (str): The name of the target column.
        insert_position (int, optional): The position at which to insert the
                                         target column in the modified DataFrame.

    Returns: df
        None
    """
    try:
        # Extract the target column
        target_col_data = df.pop(target_col)

        # Insert the target column back into the scaled DataFrame
        df.insert(insert_position, target_col, target_col_data)

        return df

    except Exception as e:
        raise f"An error occurred in the modify_dataset function: {str(e)}"


def evaluate_all_correlations(data, alpha=0.05):
    """
    Evaluate the significance of correlation coefficients for all pairs of columns in the DataFrame.

    Parameters:
    - data: DataFrame containing the dataset.
    - alpha: Significance level for hypothesis testing (default is 0.05).

    Returns:
    - correlations: Dictionary containing correlation coefficients, p-values, and significance for each pair of columns.
    """

    try:
        correlations = {}

        # Generate all combinations of column pairs
        column_pairs = combinations(data.columns, 2)

        # Iterate over each pair of columns
        for pair in column_pairs:
            column1, column2 = pair

            # Extract the columns from the DataFrame
            x = data[column1]
            y = data[column2]

            # Calculate the Pearson correlation coefficient and its p-value
            correlation_coefficient, p_value = pearsonr(x, y)

            # Determine if the correlation is statistically significant
            significant = p_value < alpha

            # Store the results in the correlations dictionary
            correlations[pair] = {
                'correlation_coefficient': correlation_coefficient,
                'p_value': p_value,
                'significant': significant
            }

        return correlations

    except Exception as e:
        raise f"An error occurred in the evaluate_all_correlations function: {str(e)}"


def get_drop_columns_list_based_on_correlation(correlation_results, threshold=0.5):
    """
    Drop columns based on correlation analysis.

    Parameters:
    - correlation_results (dict): A dictionary containing correlation results
                                  for each pair of columns.
    - threshold (float): The correlation coefficient threshold to consider
                         columns as highly correlated. Default is 0.5.

    Returns:
    - list: A list of column names to be dropped.
    """
    try:
        columns_to_drop = []

        for pair, results in correlation_results.items():
            col1, col2 = pair
            corr_coefficient = abs(results['correlation_coefficient'])
            is_significant = results['significant']

            if corr_coefficient >= threshold and is_significant:
                # If both columns are highly correlated and significant,
                # drop the column with lower relevance or relevance to the task.
                # Here, we're assuming relevance needs to be manually determined.
                columns_to_drop.append(col1)

        return columns_to_drop

    except Exception as e:
        raise f"An error occurred in the get_drop_columns_list_based_on_correlation function: {str(e)}"
