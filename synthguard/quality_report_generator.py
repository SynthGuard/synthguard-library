import os
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from synthgauge.metrics.propensity import propensity_metrics, specks
from sdv.evaluation.single_table import evaluate_quality as evaluate_quality_sdv
from io import StringIO 
import re



class DataQualityEvaluator:
    def __init__(self, real_data, synthetic_data, metadata, method='realistic'):
        """
        Initializes the DataQualityEvaluator with real and synthetic data, metadata, and evaluation method.

        Args:
            real_data (pd.DataFrame): The real data.
            synthetic_data (pd.DataFrame): The synthetic data.
            metadata (Metadata): Metadata associated with the data.
            method (str): Evaluation method ('realistic' by default).
        """
        self.synthetic_data = synthetic_data
        self.real_data = real_data
        self.metadata = metadata
        self.method = method
        self.quality_report = None
        self.fig = None

        # Mapping of methods to their corresponding functions
        self.methods = {
            'realistic': lambda data: self.realistic_method(data, metadata),
            'causal': 'TO ADD',
            'knowledge-based': 'TO ADD',
            'hybrid': 'TO ADD'
        }

        # Check if the provided method is valid
        if self.method not in self.methods:
            raise ValueError(f"Unknown method: {self.method}")

    def evaluate_quality(self):
        """
        Evaluates the quality of the synthetic data compared to real data using the specified method.

        Returns:
            dict: The quality report.
        """
        # Call the selected method from the methods dictionary
        method_function = self.methods[self.method]
        self.quality_report = method_function(self.synthetic_data)
        return self.quality_report

    def plot_quality_report_realistic(self, output_path=None):
        """
        Plots a quality report comparing real and synthetic data quality.

        Args:
            output_path (str, optional): Path to save the quality report as PDF and SVG.
        """
        observed, standard, ratio = self._calculate_propensity_metrics()
        print('Ks_distance...')
        ks_distance = self._calculate_ks_distance()

        # Create the figure and axes for the plots
        fig = plt.figure(figsize=(14, 12))  # Increased height for better spacing
        grid = plt.GridSpec(3, 2, hspace=0.5, wspace=0.3)  # Adjusted grid for better layout

        # Create subplots for statistical similarity and SPECKS
        plot_axes = fig.add_subplot(grid[0:2, 0])  # Left plot (similarity metrics)
        specks_axes = fig.add_subplot(grid[0:2, 1])  # Right plot (SPECKS)

        # Plot Quality Report (Statistical Similarity Metrics)
        self._plot_statistical_similarity(plot_axes)

        # Plot SPECKS (KS Distance)
        self._plot_specks(specks_axes, ks_distance)

        # Add a text box below the plots for propensity metrics
        text_ax = fig.add_subplot(grid[2, :])  # Full-width text box below the plots
        text_ax.axis('off')  # Hide the axis

        # Add the propensity metrics as text
        text = (
            f"Propensity Metrics:\n"
            f"Observed pMSE: {observed:.2f} - Measures the difference between real and synthetic data.\n"
            f"Standardized pMSE: {standard:.2f} - Normalized version of observed pMSE.\n"
            f"Observed-null pMSE ratio: {ratio:.2f} - Ratio of observed pMSE to null pMSE."
        )
        text_ax.text(
            0, 0.5, text, ha='left', va='center', fontsize=14, wrap=True, transform=text_ax.transAxes
        )

        # Save the figure to the output path or locally
        self._save_quality_report(output_path)
        self.fig = fig

        # Display the plot
        plt.show()

    def _calculate_propensity_metrics(self):
        """
        Calculates the propensity metrics for the real and synthetic data using the minimum number of rows from either dataset.
    
        Returns:
            tuple: Observed pMSE, Standardized pMSE, and Observed-null pMSE ratio.
        """
        print('_calculate_propensity_metrics')
    
        # Determine the sample size (e.g., 5000 rows or less)
        sample_size = min(5000, len(self.real_data), len(self.synthetic_data))

        # Randomly sample rows from both datasets
        real_data_sample = self.real_data.sample(n=sample_size, random_state=42)
        synth_data_sample = self.synthetic_data.sample(n=sample_size, random_state=42)

        # Convert datetime columns to object type for propensity metrics
        real_data_sample = self._convert_datetime_to_object(real_data_sample)
        synth_data_sample = self._convert_datetime_to_object(synth_data_sample)
    
        # Try-catch block for handling different data types
        try:
            print('Try: propensity_metrics')
            observed, standard, ratio = propensity_metrics(
                real=real_data_sample,
                synth=synth_data_sample,
                method='cart',
                num_perms= 10,
                estimator='boot'
            )
        except Exception as e:
            # If an exception occurs, attempt with object data type
            print(f'Exception {e} occurred, retrying with object type.')
            observed, standard, ratio = propensity_metrics(
                real=real_data_sample.astype('object'),
                synth=synth_data_sample.astype('object'),
                method='cart',
                num_perms= 10,
                estimator='boot'
            )
    
        print('Finished calculating propensity metrics.')
        
        return observed, standard, ratio


    def _convert_datetime_to_object(self, df):
        """
        Automatically detect datetime columns and convert them into object type (or timestamp).
        """
        for column in df.select_dtypes(include=['datetime64[ns]']).columns:
            # Convert datetime to string (object type)
            df[column] = df[column].astype(str)
            
            # If you prefer to convert datetime to numerical (timestamp in seconds), use:
            # df[column] = df[column].astype('int64') // 10**9  # Uncomment for timestamp
            
        return df

    def _calculate_ks_distance(self):
        """
        Calculates the KS Distance (SPECKS) for the real and synthetic data.

        Returns:
            float: The KS Distance.
        """
        #print('_calculate_ks_distance')
        # categorical_columns = self.real_data.select_dtypes(include=['object', 'category']).columns.tolist()
        # categorical_column_indices = [self.real_data.columns.get_loc(col) for col in categorical_columns]

        # Identify categorical columns (object or category types)
        categorical_columns = self.real_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Identify float columns and treat them as categorical
        float_columns = self.real_data.select_dtypes(include=['float']).columns.tolist()
        
        # Combine categorical and float columns
        columns_to_convert = list(set(categorical_columns + float_columns))
        
        # Get indices for categorical and float columns
        categorical_column_indices = [self.real_data.columns.get_loc(col) for col in columns_to_convert]

        # Convert these columns in both real and synthetic data
        #print('For...')
        for col in columns_to_convert:
            self.real_data[col] = self.real_data[col].fillna("missing").astype(str)
            self.synthetic_data[col] = self.synthetic_data[col].fillna("missing").astype(str)

        try:
            #print('Try:')
            ks_distance = specks(
            real=self.real_data,
            synth=self.synthetic_data,
            classifier=CatBoostClassifier,
            iterations=100,
            learning_rate=0.1,
            depth=6,
            cat_features=categorical_column_indices,
            random_state=42,
            verbose=0
        )
        except:
            #print('Except')
            self.real_data = self._convert_datetime_to_object(self.real_data)
            self.synthetic_data = self._convert_datetime_to_object(self.synthetic_data)
            ks_distance = specks(
            real=self.real_data,
            synth=self.synthetic_data,
            classifier=CatBoostClassifier,
            iterations=100,
            learning_rate=0.1,
            depth=6,
            cat_features=categorical_column_indices,
            random_state=42,
            verbose=0
        )
            
        return ks_distance

    def _plot_statistical_similarity(self, ax):
        """
        Plots the statistical similarity metrics on the given axis.

        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
        """
        scores = self.quality_report.get_properties()
        labels = scores['Property'].tolist()
        sizes = scores['Score'].tolist()
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        import numpy as np
        # if any value in the scores is nan, do not plot it
        if any(np.isnan(sizes)):
            return 

        ax.bar(labels, sizes, color=colors[:len(labels)])
        ax.set_title('Statistical Similarity Metrics')
        ax.set_ylim(0, 1)  # Set y-axis limits to 0-1

        # Add values above each bar
        for i, v in enumerate(sizes):
            ax.text(i, v, f"{v:.2f}", ha='center', fontweight='bold')

    def _plot_propensity_metrics(self, ax, observed, standard, ratio):
        """
        Plots the propensity metrics on the given axis.

        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            observed (float): Observed pMSE value.
            standard (float): Standardized pMSE value.
            ratio (float): Observed-null pMSE ratio.
        """
        prop_labels = ['Observed pMSE', 'Standardized pMSE', 'Observed-null pMSE ratio']
        prop_values = [observed, standard, ratio]

        ax.bar(prop_labels, prop_values, color='skyblue')
        ax.set_title('Propensity Metrics')
        ax.set_ylabel('Value')

        # Add values above each bar
        for i, v in enumerate(prop_values):
            ax.text(i, v, f"{v:.2f}", ha='center', fontweight='bold')

    def _plot_specks(self, ax, ks_distance):
        """
        Plots the SPECKS (KS Distance) on the given axis.

        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            ks_distance (float): The KS distance (SPECKS) value.
        """
        ax.bar(['SPECKS (KS distance)'], [ks_distance], color='purple')
        ax.set_title('SPECKS Metric (KS Distance)')
        ax.set_ylim(0, 1)  # Set y-axis limits to 0-1

        # Add value above the bar
        ax.text(0, ks_distance, f"{ks_distance:.2f}", ha='center', fontweight='bold')

    def _save_quality_report(self, output_path):
        """
        Saves the quality report as PDF and SVG files.

        Args:
            output_path (str): The path to save the report.
        """
        if output_path:
            plt.savefig(os.path.join(output_path, 'utility_report.pdf'), bbox_inches='tight')
            plt.savefig(os.path.join(output_path, 'utility_report.svg'), bbox_inches='tight')
        else:
            plt.savefig('utility_report.pdf', bbox_inches='tight')
            plt.savefig('utility_report.svg', bbox_inches='tight')

    def realistic_method(self, data, metadata):
        """
        Method for the 'realistic' evaluation.

        Args:
            data (pd.DataFrame): The synthetic data.
            metadata (Metadata): Metadata for the evaluation.
        """
        # You can implement any specific logic for the 'realistic' evaluation here.
        # For now, we'll use existing quality metrics for this method.
        self.quality_report = evaluate_quality_sdv(data, self.real_data, metadata)
        return self.quality_report

    def save_plot_to_html(self, html_file_path):
        """
        Saves the current figure to an SVG and writes it to an HTML file.
        Ensures the SVG is responsive for Kubeflow Pipelines Visualization tab.
        """
        if self.fig is None:
            raise ValueError("No figure available. Generate a plot first.")

        # Save the figure as SVG into a string buffer
        svg_buffer = StringIO()
        self.fig.savefig(svg_buffer, format='svg')
        svg_content = svg_buffer.getvalue()
        svg_buffer.close()

        # Remove width and height attributes from the <svg> tag
        svg_content = re.sub(r'(<svg[^>]*)(\swidth="[^"]*")', r'\1', svg_content)
        svg_content = re.sub(r'(<svg[^>]*)(\sheight="[^"]*")', r'\1', svg_content)

        # Ensure viewBox is present; if not, add a default one (adjust as needed)
        if 'viewBox' not in svg_content:
            width_match = re.search(r'width="([\d\.]+)(\w*)"', svg_content)
            height_match = re.search(r'height="([\d\.]+)(\w*)"', svg_content)
            if width_match and height_match:
                width = width_match.group(1)
                height = height_match.group(1)
                viewbox_str = f' viewBox="0 0 {width} {height}"'
            else:
                viewbox_str = ' viewBox="0 0 800 600"'
            svg_content = re.sub(r'(<svg[^>]*)', r'\1' + viewbox_str, svg_content, count=1)

        # Write the SVG content to an HTML file with responsive style
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Utility Report</title>
            <style>
                html, body {{
                    width: 100%;
                    height: 100%;
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                }}
                #container {{
                    width: 100vw;
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: stretch;
                    justify-content: start;
                }}
                svg {{
                    width: 100%;
                    height: 100%;
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div id="container">
                {svg_content}
            </div>
        </body>
        </html>
        """

        with open(html_file_path, 'w') as f:
            f.write(html_content)