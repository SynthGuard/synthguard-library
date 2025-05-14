import os
import pandas as pd 
import datetime
import numpy as np
import time
import json
from sdv.metadata import Metadata
from faker import Faker
import random

def load_data(file_path):
    """Loads data from a given file path."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        return data
    else:
        raise FileNotFoundError(f"File not found: {file_path}")
    
def load_metadata(file_path):
    """Loads metadata from a json file path."""
    if os.path.exists(file_path):
        return Metadata.load_from_json(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def load_data_csv(file_path):
    # List of common delimiters
    delimiters = [',', '\t', ';']
    # Dictionary to store the number of columns for each delimiter
    columns_count = {}
    
    # Try to read the file with each delimiter
    for delimiter in delimiters:
        try:
            # Read the first row to determine the number of columns
            df = pd.read_csv(file_path, delimiter=delimiter, header=None, nrows=1)
            columns_count[delimiter] = len(df.columns)
        except pd.errors.EmptyDataError:
            # Skip the delimiter if the file is empty
            continue
        except Exception as e:
            # Handle other exceptions
            print(f"Error occurred while reading with delimiter '{delimiter}': {e}")
            continue
    
    # Choose the delimiter with the maximum number of columns (usually a good heuristic)
    if columns_count:
        best_delimiter = max(columns_count, key=columns_count.get)
    else:
        raise ValueError("Could not detect a valid delimiter for the file.")
    
    # Read the file using the detected delimiter
    try:
        df = pd.read_csv(file_path, delimiter=best_delimiter)
    except Exception as e:
        print(f"Error occurred while reading the file with the best delimiter '{best_delimiter}': {e}")
        df = pd.DataFrame()  # Return an empty DataFrame if there's an error
    
    return df

def save_to_csv(df: pd.DataFrame, file_path: str):
    """Saves given content to a file."""
    try: 
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise e(f'DataFrame failed to save: {e}')
    print(f"DataFrame saved to {file_path}")

def save_metadata(metadata, file_path):
    """Saves given metadata to a json"""
    try:
        metadata.save_to_json(file_path)
    except Exception as e:
        raise e(f'Error with saving metadata: {e}')
    print(f"Metadata saved to {file_path}")


def save_to_file(content, file_path):
    """Saves given content to a file."""
    with open(file_path, 'w') as file:
        file.write(content)
    print(f"Data saved to {file_path}")


def log_event(event_message, log_file="tool_log.txt"):
    """Logs events to a file with timestamps."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {event_message}\n"
    
    with open(log_file, 'a') as file:
        file.write(log_message)
    
    print(f"Logged: {event_message}")


def validate_data_format(data, expected_format="csv"):
    """Validates that the data is in the expected format."""
    if not data.endswith(f".{expected_format}"):
        raise ValueError(f"Invalid data format. Expected {expected_format} format.")
    return True

def validate_not_empty(data):
    """Checks if the data is not empty."""
    if not data:
        raise ValueError("The data is empty.")
    return True


def convert_dict_to_json(data_dict):
    """Converts a Python dictionary to a JSON string."""
    return json.dumps(data_dict, indent=4)

def convert_json_to_dict(json_string):
    """Converts a JSON string to a Python dictionary."""
    return json.loads(json_string)


def calculate_mean(data):
    """Calculates the mean of a dataset."""
    return np.mean(data)

def calculate_standard_deviation(data):
    """Calculates the standard deviation of a dataset."""
    return np.std(data)


def show_progress(current_step, total_steps):
    """Displays a simple progress indicator."""
    percentage = (current_step / total_steps) * 100
    print(f"Progress: {percentage:.2f}% ({current_step}/{total_steps})")
    time.sleep(0.5)


def handle_error(error_message, terminate=False):
    """Handles errors by logging and optionally terminating the process."""
    log_event(f"Error: {error_message}")
    print(f"Error: {error_message}")
    
    if terminate:
        exit(1)

def start_timer():
    """Starts a timer to measure process duration."""
    return time.time()

def end_timer(start_time):
    """Ends the timer and returns the elapsed time."""
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    return elapsed_time



# read json files and parse them into df
import pandas as pd
import json
from pandas import json_normalize

def flatten_json(json_obj, parent_key='', sep='_'):
    """Recursively flatten JSON."""
    items = []
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            items.extend(flatten_json(value, new_key, sep=sep).items())
    elif isinstance(json_obj, list):
        # If the value is a list, expand each item into a row
        for i, elem in enumerate(json_obj):
            items.extend(flatten_json(elem, f"{parent_key}{sep}{i}", sep=sep).items())
    else:
        items.append((parent_key, json_obj))
    return dict(items)

def handle_nested_data_json(df):
    """Automatically handle nested lists and dictionaries in a DataFrame."""
    for column in df.columns:
        # If the column contains dictionaries, flatten them
        if df[column].apply(lambda x: isinstance(x, dict)).any():
            df[column] = df[column].apply(lambda x: flatten_json(x) if isinstance(x, dict) else x)
            df = pd.concat([df.drop(columns=[column]), pd.json_normalize(df[column], sep='_')], axis=1)

        # If the column contains lists, explode them into multiple rows
        elif df[column].apply(lambda x: isinstance(x, list)).any():
            # Explode the lists into multiple rows and normalize if necessary
            df_exploded = df.explode(column).reset_index(drop=True)
            if df_exploded[column].apply(lambda x: isinstance(x, dict)).any():
                df_exploded[column] = df_exploded[column].apply(lambda x: flatten_json(x) if isinstance(x, dict) else x)
                df_exploded = pd.concat([df_exploded.drop(columns=[column]), pd.json_normalize(df_exploded[column], sep='_')], axis=1)
            df = df_exploded

    return df


# Function to reverse the flattening process (convert back to original nested structure)
def reverse_flatten(df, original_json_structure):
    """Revert the DataFrame back to the original nested structure."""
    
    # Convert any timestamp columns to ISO 8601 strings
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    
    
    # if any of the columns are not objects, convert them to objects
    # for col in df.columns:
    #     if df[col].dtype != 'object':
    #         df[col] = df[col].astype('object')
    
    def get_column_path(col, parent=None):
        """Build the path to the nested column."""
        path = col.split('_')
        return parent + path if parent else path

    # Create a copy of the original structure for rebuilding
    rebuilt_json = original_json_structure.copy()

    # For each column in the flattened DataFrame
    for col in df.columns:
        # Get the corresponding nested path for the column
        path = get_column_path(col)
        value = df[col]

        
        # Set value in the correct place in the nested dictionary
        try:            
            current_dict = rebuilt_json
            for part in path[:-1]:
                current_dict = current_dict.get(part, {})
            current_dict[path[-1]] = value.tolist() if len(value) > 1 else value.iloc[0]
        except Exception as e:
            print(f"Error setting value for column {col}: {e}")

    return current_dict


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
    
    

def save_json(output_folder, file_name, rebuilt_synthetic_data):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
        
    output_file_path = os.path.join(output_folder, file_name)
    
    with open(output_file_path, 'w') as output_file:
        json.dump(rebuilt_synthetic_data, output_file, indent=4)
        print(f"Saved rebuilt synthetic data to: {output_file_path}")
        
              
def combine_json_files(input_files, output_file):
    """
    Merge multiple JSON files into one.
    
    Parameters:
        input_files (list): List of JSON file paths to merge.
        output_file (str): Path to save the merged JSON file.
    """
    merged_data = {}
    
    for file in input_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                # Merge current file data into the main dictionary
                merged_data.update(data)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Save the merged JSON to the output file
    try:
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=4)
        print(f"Merged JSON saved to {output_file}")
    except Exception as e:
        print(f"Error saving merged JSON: {e}")
        


import pandas as pd
import os

def read_txt_and_convert_to_df(input_path: str, input_file: str) -> pd.DataFrame:
    """
    Reads a text file from the given path, splits the text into individual street names,
    and returns a DataFrame with the column renamed to the file name (without extension).
    
    Parameters:
    input_path (str): The path where the input file is located.
    input_file (str): The name of the text file to read.
    
    Returns:
    pd.DataFrame: DataFrame with street names as rows and the column named after the file.
    """
    # Full file path
    file_path = os.path.join(input_path, input_file)
    
    # Read the text file and extract the first line
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    # Split the text into individual street names (assuming street names are space-separated in the first line)
    street_names = data[0].split()  # Split the first line of the file into a list of street names
    
    # Create DataFrame and rename column based on the file name (without extension)
    df_street_names = pd.DataFrame(street_names, columns=[os.path.splitext(input_file)[0]])
    
    return df_street_names


def combine_and_save(data_df: pd.DataFrame, output_path: str, output_file: str, delimiter: str = ' '):
    """
    Saves the content of a DataFrame to a text file, with each row as a space-separated string.
    
    Parameters:
    data_df (pd.DataFrame): The DataFrame to be saved.
    output_path (str): The path where the output file should be saved.
    output_file (str): The name of the output file.
    delimiter (str): The delimiter used to separate values in each row (default is space).
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save the DataFrame content to a text file
    with open(os.path.join(output_path, output_file), 'w') as file:
        for _, row in data_df.iterrows():
            # Convert each row to a string, join values with the specified delimiter, and write to file
            file.write(delimiter.join(row.astype(str).values) + '\n')
            
            


def generate_data_addresses(street_names, municipality_codes, n_addresses=10):
    # Initialize Faker
    fake = Faker()
    Faker.seed(0)  # Set seed for reproducibility

    # Ensure both DataFrames have the same number of rows
    min_rows = min(len(street_names), len(municipality_codes))

    # Truncate the longer DataFrame to the same number of rows as the smaller one
    street_names = street_names.iloc[:min_rows]
    municipality_codes = municipality_codes.iloc[:min_rows]

    # Convert columns to appropriate types
    street_names = street_names.astype(str).squeeze()  # Convert to Series if single column
    municipality_codes = municipality_codes.astype(int).squeeze()

    # Listify for random sampling
    street_names = street_names.tolist()
    municipality_codes = municipality_codes.tolist()

    # Generate synthetic addresses
    # n_addresses = 10  # Number of addresses to generate
    addresses = []

    for _ in range(n_addresses):
        street_name = random.choice(street_names)
        municipality_code = random.choice(municipality_codes)
        house_number = fake.building_number()
        city = fake.city()  # Correct method for generating city names
        postal_code = fake.postcode()
        
        # Format the address
        address = {
            "id": len(addresses) + 1,
            "address": f"{street_name}, {house_number}, {city}, {postal_code}",
            #"municipality_code": municipality_code
        }
        addresses.append(address)

    # Convert to DataFrame for better handling
    addresses_df = pd.DataFrame(addresses)
    
    return addresses_df

    # Display generated addresses
    # print(addresses_df)           
    
    
# zip list of files
import zipfile
def zip_files(files, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))

# unzip files into  output folder
def unzip_files(zip_name, output_folder):
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(output_folder)