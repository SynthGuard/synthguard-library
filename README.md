# synthguard

`synthguard` is a Python library designed to streamline the process of synthetic data generation, quality evaluation, and privacy risk assessment. It provides a comprehensive suite of tools for preprocessing data, generating synthetic datasets, evaluating their quality, and assessing privacy risks. The library leverages the Synthetic Data Vault (SDV) framework for many of its core functionalities.

---

## Features

- **Data Preprocessing**: Simplifies data preparation for machine learning or synthetic data generation.
- **Synthetic Data Generation**: Supports multiple methods for generating synthetic datasets.
- **Quality Evaluation**: Assesses the quality of synthetic data using statistical metrics and visualizations.
- **Privacy Risk Assessment**: Evaluates the privacy risks associated with synthetic datasets.
- **Utility Functions**: Includes helper functions for data handling, validation, and statistical calculations.

---

## Installation

To install the library, clone the repository and install the dependencies:

```bash
git clone https://github.com/your-repo/synthguard-library.git
cd synthguard-library
pip install -r requirements.txt
```

---

## Modules and Classes

### 1. DataPreprocessor Class

The `DataPreprocessor` class provides tools for preparing raw data for machine learning or synthetic data generation. It handles tasks such as:

- Missing value imputation (mean, median, mode, or drop).
- Categorical encoding for machine learning models.
- Numerical scaling to zero mean and unit variance.
- Timestamp conversion to a standard datetime format.

**Key Features**:
- SDV metadata extraction for efficient synthetic data generation.
- A comprehensive preprocessing pipeline via the `preprocess_data()` method.

**Example Usage**:
```python
import pandas as pd
from synthguard.data_preprocessor import DataPreprocessor

df = pd.read_csv('your_data.csv')
preprocessor = DataPreprocessor(data=df)
processed_data, metadata = preprocessor.preprocess_data()
```

---

### 2. SyntheticDataGenerator Class

The `SyntheticDataGenerator` class generates synthetic datasets based on preprocessed real data. It supports multiple generation methods and integrates with SDV.

**Key Features**:
- Supports methods like 'hybrid,' 'causal,' 'knowledge-based,' and 'realistic.'
- Uses SDV's `GaussianCopulaSynthesizer` for realistic data generation.
- Allows saving generated data to a CSV file.

**Example Usage**:
```python
from synthguard.synthetic_data_generator import SyntheticDataGenerator

synthesizer = SyntheticDataGenerator(n_rows=5000, output_csv='synthetic_data.csv', method='realistic')
synthetic_df = synthesizer.generate_synthetic_data(processed_data, metadata)
print(synthetic_df.head())
```

---

### 3. DataQualityEvaluator Class

The `DataQualityEvaluator` class evaluates the quality of synthetic data using SDV's metrics and provides visualizations.

**Key Features**:
- Calculates scores for statistical properties like mean, variance, and correlations.
- Generates visualizations for quality reports and column comparisons.
- Provides detailed property scores.

**Example Usage**:
```python
from synthguard.quality_report_generator import DataQualityEvaluator

evaluator = DataQualityEvaluator(real_data=df, synthetic_data=synthetic_df, metadata=metadata)
evaluator.evaluate_quality()
print("Overall Score:", evaluator.get_score())
evaluator.visualize_quality_report()
```

---

### 4. PrivacyRiskEvaluator Class

The `PrivacyRiskEvaluator` class assesses privacy risks associated with synthetic data, including identity and attribute disclosure risks.

**Key Features**:
- Calculates identity and attribute disclosure risk scores.
- Uses SDV's `CategoricalCAP` metric for attribute disclosure risk.
- Identifies categorical columns for privacy analysis.

**Example Usage**:
```python
from synthguard.privacy_report_generator import PrivacyRiskEvaluator

evaluator = PrivacyRiskEvaluator(real_data=df, synthetic_data=synthetic_df, metadata=metadata)
privacy_report = evaluator.evaluate_privacy()
print("Identity Disclosure Risk:", privacy_report['identity_disclosure_risk'])
print("Attribute Disclosure Risk:", privacy_report['attribute_disclosure_risk'])
```

---

### 5. InputHandler Class

The `InputHandler` class simplifies loading data from various sources into pandas DataFrames.

**Key Features**:
- Supports CSV, TXT, Parquet files, and URLs.
- Automatically infers delimiters for CSV files.
- Includes error handling for file existence and parsing issues.

**Example Usage**:
```python
from synthguard.input_handler import InputHandler

handler = InputHandler()
data_csv = handler.load_data_from_csv('path/to/your/data.csv')
print(data_csv)
```

---

### 6. helper_functions Module

The `helper_functions` module provides reusable utility functions for common tasks.

**Functions**:
- **Data Loading & Saving**:
  - `load_data(file_path)`: Loads file content into a string.
  - `load_data_csv(file_path)`: Loads CSV data into a pandas DataFrame.
  - `save_to_file(content, file_path)`: Saves content to a file.

- **Logging & Error Handling**:
  - `log_event(event_message, log_file="tool_log.txt")`: Logs events with timestamps.
  - `handle_error(error_message, terminate=False)`: Logs errors and optionally terminates the process.

- **Data Validation**:
  - `validate_data_format(data, expected_format="csv")`: Validates file format.
  - `validate_not_empty(data)`: Ensures data is not empty.

- **Data Conversion**:
  - `convert_dict_to_json(data_dict)`: Converts a dictionary to JSON.
  - `convert_json_to_dict(json_string)`: Converts JSON to a dictionary.

- **Statistical Calculations**:
  - `calculate_mean(data)`: Calculates the mean using NumPy.
  - `calculate_standard_deviation(data)`: Calculates the standard deviation.

- **Progress & Timing**:
  - `show_progress(current_step, total_steps)`: Displays a progress indicator.
  - `start_timer()`: Starts a timer.
  - `end_timer(start_time)`: Ends the timer and prints elapsed time.
