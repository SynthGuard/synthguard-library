import os
import pandas as pd

class InputHandler:
    def __init__(self):
        self.data = None
        

    def load_data_csv(self, file_path:str, n_rows:int = None, delimiter:str = None):
        # List of common delimiters
        delimiters = [',', '\t', ';']
        # Dictionary to store the number of columns for each delimiter
        columns_count = {}
        
        if not delimiter:
            # Try to read the file with each delimiter
            for delimiter in delimiters:
                try:
                    # Read the first row to determine the number of columns
                    df = pd.read_csv(file_path, delimiter=delimiter, header=None, nrows=1)
                    columns_count[delimiter] = len(df.columns)  
                    print(f"Detected {columns_count[delimiter]} columns with delimiter '{delimiter}'")
                except pd.errors.EmptyDataError:
                    # Skip the delimiter if the file is empty
                    print(f"Skipping delimiter '{delimiter}' due to EmptyDataError")
                    continue
                except Exception as e:
                    # Handle other exceptions
                    print(f"Error occurred while reading with delimiter '{delimiter}': {e}")
                    continue
            
            # Choose the delimiter with the maximum number of columns (usually a good heuristic)
            if columns_count:
                best_delimiter = max(columns_count, key=columns_count.get)
                print(f"Choosing best delimiter '{best_delimiter}' with {columns_count[best_delimiter]} columns")
            else:
                raise ValueError("Could not detect a valid delimiter for the file.")
        else:
            best_delimiter = delimiter
        
        try:
            if n_rows:
                df = pd.read_csv(file_path, delimiter=best_delimiter, nrows=n_rows)
            else:
                df = pd.read_csv(file_path, delimiter=best_delimiter)
        except Exception as e:
            print(f"Error occurred while reading the file with the best delimiter '{best_delimiter}': {e}")
            df = pd.DataFrame()  # Return an empty DataFrame if there's an error
        
        self.data = df


    def load_data_from_txt(self, file_path):
        """Loads data from a TXT file."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                self.data = file.readlines()
            print(f"Data loaded successfully from {file_path}")
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    def load_data_from_parquet(self, file_path):
        """Loads data from a Parquet file."""
        if os.path.exists(file_path):
            self.data = pd.read_parquet(file_path)
            print(f"Data loaded successfully from {file_path}")            
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    def load_data_from_url(self, url):
        """Loads data from a URL."""
        try:
            self.data = pd.read_csv(url, sep='\t')  # Assuming CSV data with tab delimiter
            print(f"Data loaded successfully from {url}")
        except Exception as e:
            raise ValueError(f"Error loading data from URL: {e}")
        
    def load_data_from_zip(self, zip_file, file_name):
        """Loads data from a ZIP file."""
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as z:
            with z.open(file_name) as f:
                self.data = pd.read_csv(f)
        print(f"Data loaded successfully from {file_name} in {zip_file}")


# # Example usage:
# # csv_data = load_data_from_csv('path_to_your_file.csv')
# # txt_data = load_data_from_txt('path_to_your_file.txt')
# # parquet_data = load_data_from_parquet('path_to_your_file.parquet')
# # url_data = load_data_from_url('http://example.com/data.csv')