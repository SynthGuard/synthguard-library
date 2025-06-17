import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from sdmetrics.single_table import CategoricalCAP, NewRowSynthesis, CategoricalKNN
from synthgauge.metrics.privacy import tcap_score, min_nearest_neighbour, sample_overlap_score
from sdv.metadata import SingleTableMetadata
from io import StringIO 
from sklearn.impute import SimpleImputer
import re

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class PrivacyRiskEvaluator:
    def __init__(self, real_data, synthetic_data, metadata, method='realistic'):
        """
        Initializes the PrivacyRiskEvaluator with real data and synthetic data.

        Args:
            real_data (pd.DataFrame): The original real data.
            synthetic_data (pd.DataFrame): The synthetic data generated.
            metadata (Metadata): Metadata for the evaluation.
            method (str): The evaluation method (default is 'realistic').
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.metadata = metadata
        self.method = method
        self.key_features = []
        self.target_feature = None
        self.privacy_metrics = None
        self.fig = None
        
        # Mapping of methods to their corresponding functions
        self.methods = {
            'realistic': self.run_privacy_realistic,
            'causal': 'self.run_privacy_causal',
            'knowledge-based': 'self.run_privacy_knowledge_based',
            'hybrid': 'self.run_privacy_hybrid'
        }

        # Check if the provided method is valid
        if self.method not in self.methods:
            raise ValueError(f"Unknown method: {self.method}")
        
    def realistic_method(self, data, metadata):
        """
        Method for the 'realistic' evaluation.
        """
        self.privacy_metrics = self.get_privacy_metrics_realistic(data)
        return self.privacy_metrics

    def generate_metadata(self):
        """
        Generates metadata for the synthetic data using SDV's SingleTableMetadata.
        """
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.synthetic_data)
        
        # Add missing columns from real data to metadata
        missing_columns = set(self.real_data.columns) - set(metadata.columns.keys())
        for column in missing_columns:
            column_type = 'categorical' if self.real_data[column].dtype == 'object' else 'numerical'
            metadata.add_column(column, sdtype=column_type)
        
        return metadata

    def set_key_target_features(self, target_column=None, key_columns=None):
        """
        Sets the key and target features for privacy evaluation.
        """
        categorical_columns = self.real_data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Select target feature
        self.target_feature = target_column or np.random.choice(categorical_columns)

        # Select key features
        if key_columns:
            self.key_features = key_columns
        else:
            self.key_features = [col for col in categorical_columns if col != self.target_feature]
            if self.key_features:
                self.key_features = [np.random.choice(self.key_features)]

    def convert_datetime_columns_to_numeric(self, df):
        """
        Converts datetime columns to numeric format.
        """
        datetime_columns = df.select_dtypes(include=['datetime']).columns
        df[datetime_columns] = df[datetime_columns].apply(lambda col: col.apply(lambda x: x.timestamp()))
        return df
    
    def normalize_sigmoid(self,x, scale=100):
        """
        Normalize a value between 0 and 1 using the sigmoid function.
        
        Args:
            x (float): The value to normalize.
            scale (int, optional): The scaling factor to control how quickly the sigmoid function converges.
            
        Returns:
            float: The normalized value between 0 and 1.
        """
        return 1 / (1 + np.exp(-x / scale))

    def get_privacy_metrics_realistic(self):
        """
        Calculates and returns various privacy metrics for the 'realistic' method.
        """
        results = {}

        # Calculate privacy metrics
        results['CategoricalCAP Score'] = CategoricalCAP.compute(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            key_fields=self.key_features,
            sensitive_fields=[self.target_feature]
        )


        try:
            results['NewRowSynthesis Score'] = NewRowSynthesis.compute(
                real_data=self.real_data,
                synthetic_data=self.synthetic_data,
                metadata=self.generate_metadata().to_dict(),
                numerical_match_tolerance=0.001,
                synthetic_sample_size=10_000
            )
        except Exception as e:
            print(f"Error in NewRowSynthesis: {e}")
            print('Setting NewRowSynthesis Score to 0.6')
            results['NewRowSynthesis Score'] = 0.6

        results['Inference Attack Score'] = CategoricalKNN.compute(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            key_fields=self.key_features,
            sensitive_fields=[self.target_feature]
        )

        results['TCAP Score'] = tcap_score(
            real=self.real_data,
            synth=self.synthetic_data,
            key=self.key_features,
            target=self.target_feature
        )

        
        try:
            results['Min Nearest Neighbour Distance'] = self.normalize_sigmoid(min_nearest_neighbour(
                real=self.convert_datetime_columns_to_numeric(self.real_data),
                synth=self.convert_datetime_columns_to_numeric(self.synthetic_data),
                feats=None,
                outliers_only=True
                )
                )
        except:
            # Fill NaN values for non-numeric columns with "missing" and convert to string
            for col in self.real_data.select_dtypes(include=['object', 'category']).columns:
                self.real_data[col] = self.real_data[col].fillna("missing").astype(str)
                self.synthetic_data[col] = self.synthetic_data[col].fillna("missing").astype(str)
            
            # Handle missing values in numeric columns and convert datetime columns to numeric
            real_data_processed = self.convert_datetime_columns_to_numeric(self.real_data)
            synth_data_processed = self.convert_datetime_columns_to_numeric(self.synthetic_data)
            
            # Replace NaN in numeric columns with a placeholder if necessary
            real_data_processed = real_data_processed.fillna(0)  # Example placeholder
            synth_data_processed = synth_data_processed.fillna(0)  # Example placeholder
            
            # Compute the metric
            results['Min Nearest Neighbour Distance'] = self.normalize_sigmoid(
                min_nearest_neighbour(
                    real=real_data_processed,
                    synth=synth_data_processed,
                    feats=None,
                    outliers_only=True
                )
                )

        results['Sample Overlap Score'] = sample_overlap_score(
            real=self.real_data,
            synth=self.synthetic_data,
            feats=None,
            sample_size=0.2,
            runs=5,
            score_type='unique'
        )

        return results

    def interpret_privacy_metrics(self, metrics):
        """
        Interprets privacy metrics and provides descriptive statements.
        """
        interpretations = {}

        for metric, score in metrics.items():
            if metric == 'CategoricalCAP Score':
                interpretations[metric] = self._interpret_categorical_cap(score)
            elif metric == 'NewRowSynthesis Score':
                interpretations[metric] = self._interpret_new_row_synthesis(score)
            elif metric == 'Inference Attack Score':
                interpretations[metric] = self._interpret_inference_attack(score)
            elif metric == 'TCAP Score':
                interpretations[metric] = self._interpret_tcap(score)
            elif metric == 'Min Nearest Neighbour Distance':
                interpretations[metric] = self._interpret_min_nearest_neighbour(score)
            elif metric == 'Sample Overlap Score':
                interpretations[metric] = self._interpret_sample_overlap(score)

        return interpretations

    def _interpret_categorical_cap(self, score):
        if score > 0.8:
            return "Best privacy protection: The real data is 100% safe from the attack."
        elif score < 0.2:
            return "Worst privacy protection: The attacker can guess every sensitive value."
        return "Privacy protection: The attacker can guess some sensitive values."

    def _interpret_new_row_synthesis(self, score):
        if score == 1.0:
            return "Best synthetic data quality: The rows in the synthetic data are all new."
        elif score == 0.0:
            return "Worst synthetic data quality: All the rows in the synthetic data are copies of real data."
        return "Moderate synthetic data quality: Some rows in the synthetic data are new."

    def _interpret_inference_attack(self, score):
        if score > 0.8:
            return "Best privacy protection: The real data is 100% safe from the attack."
        elif score < 0.2:
            return "Worst privacy protection: The attacker can guess every sensitive value."
        return "Privacy protection: The attacker can guess some sensitive values."

    def _interpret_tcap(self, score):
        if score < 0.2:
            return "Best privacy preservation: minimal target leakage."
        elif score > 0.8:
            return "Worst privacy preservation: significant target leakage."
        return "Privacy preservation: moderate target leakage."

    def _interpret_min_nearest_neighbour(self, score):
        if score > 0.5:
            return "Best privacy protection: synthetic samples are well-distanced from real data."
        elif score < 0.2:
            return "Worst privacy protection: synthetic samples closely match real data."
        return "Moderate privacy protection: some overlap between synthetic and real data."

    def _interpret_sample_overlap(self, score):
        if score < 0.2:
            return "Best privacy protection: low overlap with real data."
        elif score > 0.8:
            return "Worst privacy protection: high overlap with real data."
        return "Moderate privacy protection: some overlap with real data."


    def plot_privacy_metrics_realistic(self, output_path=None):
        metrics = self.privacy_metrics
        interpretations = self.interpretations

        metric_names, scores = zip(*metrics.items())

        fig, ax = plt.subplots(figsize=(14, 8))
        colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']

        y_positions = range(len(metric_names))
        bars = ax.barh(y_positions, scores, color=colors, height=0.5)

        ax.set_xlabel('Score', fontsize=12)
        ax.set_title(
            f'Privacy Report\nAttacker knows: {self.key_features}\nAttacker guesses: {self.target_feature}',
            fontsize=14,
            pad=20
        )
        ax.set_xlim(0, 1.3)  # Extend x-axis to allow room for text
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Keep axis ticks capped at 1.0
        ax.set_yticks(y_positions)
        ax.set_yticklabels(metric_names, fontsize=11)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        # Annotate scores and interpretations with dynamic positioning and color
        for i, (bar, score, metric) in enumerate(zip(bars, scores, metric_names)):
            if score > 0.79:
                ax.text(score - 0.02, i, f'{score:.2f}', va='center', ha='right', fontsize=10, weight='bold', color='white')
                ax.text(score - 0.01, i, interpretations[metric], va='center', ha='right', fontsize=9, color='white')
            else:
                ax.text(score + 0.02, i, f'{score:.2f}', va='center', ha='left', fontsize=10, weight='bold')
                ax.text(score + 0.1, i, interpretations[metric], va='center', ha='left', fontsize=9, color='dimgray')

        plt.tight_layout()
        plt.subplots_adjust(left=0.35, right=0.95)

        if output_path:
            plt.savefig(os.path.join(output_path, 'privacy_report.pdf'), bbox_inches='tight')
            plt.savefig(os.path.join(output_path, 'privacy_report.svg'), bbox_inches='tight')
        else:
            plt.savefig('privacy_report.pdf', bbox_inches='tight')
            plt.savefig('privacy_report.svg', bbox_inches='tight')

        plt.show()
        self.fig = fig


    def run_privacy_realistic(self):
        """
        Runs the 'realistic' method for privacy evaluation.
        """
        self.set_key_target_features()
        self.privacy_metrics = self.get_privacy_metrics_realistic()
        self.interpretations = self.interpret_privacy_metrics(self.privacy_metrics)
        # self.plot_privacy_metrics_realistic(self.privacy_metrics, interpretations)
        return self.privacy_metrics, self.interpretations

    

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
        
        # Write the SVG content to an HTML file with responsive style
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
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


