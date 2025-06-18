import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sdv.metadata import Metadata


class DataPreprocessor:
    def __init__(self, data=None):
        self.data = data
        self.metadata = {}

    def check_real_data_availability(self):
        """Check if real data (DataFrame) is available and not empty."""
        if self.data is not None and not self.data.empty:
            return True
        print("No real data available or the data is empty.")
        return False

    def handle_missing_values(self, strategy='mean'):
        """Handle missing values in the data using a specified strategy."""
        if self.check_real_data_availability():
            for column in self.data.columns:
                if self.data[column].isnull().any():
                    if self.data[column].dtype in [int, float]:  # Numeric columns
                        if strategy == 'mean':
                            self.data[column].fillna(self.data[column].mean(), inplace=True)
                        elif strategy == 'median':
                            self.data[column].fillna(self.data[column].median(), inplace=True)
                        elif strategy == 'mode':
                            self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                        elif strategy == 'drop':
                            self.data.dropna(subset=[column], inplace=True)
                        else:
                            raise ValueError("Invalid strategy for handling missing values.")
                    else:  # Non-numeric columns
                        if strategy == 'mode':
                            self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                        elif strategy == 'drop':
                            self.data.dropna(subset=[column], inplace=True)
                            
    def replace_invalid_values(self):
        """Replace invalid values ('-', 'n.d.') with pd.NA."""
        if self.check_real_data_availability():
            self.data.replace(['-', 'n.d.', 'None'], pd.NA, inplace=True)
            
    def drop_na_columns(self, columns_to_drop):
        """Drop rows with missing values in specific columns."""
        if columns_to_drop is None:
            pass
        else:            
            for column in columns_to_drop:
                if column in self.data.columns:
                    self.data.dropna(subset=[column], inplace=True)
                    
        # if self.check_real_data_availability():
        #     self.data.dropna(subset=[columns_to_drop], inplace=True)
            
    def assign_column_dtypes(self, columns_dict):
        """Assign specified column data types."""
        if self.check_real_data_availability():
            self.data = self.data.astype(columns_dict)

    def encode_categorical_data(self):
        """Encode categorical variables using LabelEncoder."""
        if self.check_real_data_availability():
            for column in self.data.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column].astype(str))

    def scale_numerical_data(self):
        """Scale numerical data using StandardScaler."""
        if self.check_real_data_availability():
            scaler = StandardScaler()
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

    def convert_timestamps(self):
        """Convert any timestamp columns to a standard datetime format."""
        if self.check_real_data_availability():
            for column in self.data.columns:
                if pd.api.types.is_datetime64_any_dtype(self.data[column]):
                    self.data[column] = pd.to_datetime(self.data[column], errors='coerce')

    def extract_metadata(self):
        """Extract metadata from the data using SDV."""
        if self.check_real_data_availability():
            metadata = Metadata.detect_from_dataframe(self.data, )
            self.metadata = metadata
            return self.metadata
        raise ValueError("No real data to extract metadata from.")

    def extract_dtypes_from_metadata(self, metadata):
        """Map SDV metadata sdtypes to pandas dtypes and extract datetime formats."""
        dtypes = {}
        datetime_formats = {}
        columns = metadata['tables']['table']['columns']
        sdtype_to_dtype = {
            'datetime': 'datetime64[ns]',
            'categorical': 'object',
            'integer': 'Int64',
            'float': 'float64',
            'boolean': 'boolean',
            'string': 'string'
        }

        for column, properties in columns.items():
            sdtype = properties['sdtype']
            if sdtype == 'datetime':
                datetime_formats[column] = properties.get('datetime_format')
            dtypes[column] = sdtype_to_dtype.get(sdtype, 'object')
        
        return dtypes

    def preprocess_data(self, handle_missing='mean', columns_dict=None, columns_to_drop=None):
        """Perform preprocessing including handling missing values, encoding, scaling, and metadata extraction."""
        if self.check_real_data_availability():
            print(f"Preprocessing data. Data shape: {self.data.shape}")
            self.handle_missing_values(strategy=handle_missing)
            # Perform the operations based on the methods
            self.replace_invalid_values()  # Replace invalid values with pd.NA
            self.drop_na_columns(columns_to_drop)         # Drop rows with missing PM10/PM2dot5 values
            if columns_dict:
                self.assign_column_dtypes(columns_dict)  # Assign specified column dtypes
                
            
            # self.convert_timestamps()
            # self.encode_categorical_data()
            # self.scale_numerical_data()
            metadata = self.extract_metadata()
            dict_dtypes = self.extract_dtypes_from_metadata(metadata.to_dict())
            # for column, dtype in dict_dtypes.items():
            #     self.data[column] = self.data[column].astype(dtype)
                
                
            for column, dtype in dict_dtypes.items():
                if 'datetime' in str(dtype):  # Check if dtype is datetime-related
                    if 'tz' in str(dtype):  # If target dtype is timezone-aware
                        self.data[column] = pd.to_datetime(self.data[column]).dt.tz_localize('UTC')  # Adjust timezone as needed
                    else:  # If target dtype is timezone-naive
                        self.data[column] = pd.to_datetime(self.data[column]).dt.tz_localize(None)
                else:
                    self.data[column] = self.data[column].astype(dtype)

            print("Preprocessing complete.")
            return self.data, metadata
        raise ValueError("No real data to preprocess.")

