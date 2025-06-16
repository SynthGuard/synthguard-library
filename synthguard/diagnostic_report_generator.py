import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import os
from sdv.evaluation.single_table import run_diagnostic as run_diagnostic_sdv
from io import StringIO
import re

class DiagnosticEvaluator:
    def __init__(self, real_data, synthetic_data, metadata, method='realistic'):
        """
        Initializes the DataQualityEvaluator with the given parameters.

        Args:
            real_data (pd.DataFrame): The real data as a pandas DataFrame.
            synthetic_data (pd.DataFrame): The synthetic data as a pandas DataFrame.
            metadata (Metadata): Metadata object describing the dataset.
            method (str): The method used for evaluation, default is 'realistic'.
        """
        self.synthetic_data = synthetic_data
        self.real_data = real_data
        self.metadata = metadata
        self.method = method
        self.quality_report = None  # Initialize quality_report as None
        self.diagnostic_report = None  # Initialize diagnostic_report as None
        self.fig = None
        
        # Mapping evaluation methods to their corresponding functions
        self.methods = {
            'realistic': self.run_diagnostic_realistic,      # 'realistic' shares the same method as 'sdv'
            'hybrid': self.run_diagnostic_hybrid,      # Maps 'hybrid' to the 'run_diagnostic_hybrid' method
            'causal': self.run_diagnostic_causal,      # Maps 'causal' to the 'run_diagnostic_causal' method
            'knowledge-based': self.run_diagnostic_knowledge_based  # Maps 'knowledge-based' to the 'run_diagnostic_knowledge_based' method
        }

    def run_diagnostic(self):
        """Runs diagnostic tests on the synthetic data."""
        if self.method not in self.methods:
            raise ValueError(f"Unknown evaluation method: {self.method}")

        # Call the corresponding diagnostic method based on the selected method
        self.diagnostic_report = self.methods[self.method]()
        return self.diagnostic_report

    def run_diagnostic_realistic(self):
        """Runs diagnostic tests using the SDV/realistic method."""
        self.diagnostic_report = run_diagnostic_sdv(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            metadata=self.metadata
        )
        return self.diagnostic_report

    def run_diagnostic_hybrid(self):
        """Runs diagnostic tests using the Hybrid method."""
        # Implement the hybrid diagnostic logic here
        print("Running Hybrid diagnostic...")
        # Return a placeholder report
        return {"Hybrid diagnostic": "Results"}

    def run_diagnostic_causal(self):
        """Runs diagnostic tests using the Causal method."""
        # Implement the causal diagnostic logic here
        print("Running Causal diagnostic...")
        # Return a placeholder report
        return {"Causal diagnostic": "Results"}

    def run_diagnostic_knowledge_based(self):
        """Runs diagnostic tests using the Knowledge-based method."""
        # Implement the knowledge-based diagnostic logic here
        print("Running Knowledge-based diagnostic...")
        # Return a placeholder report
        return {"Knowledge-based diagnostic": "Results"}

    def plot_diagnostic_report_realistic(self, output_path=None):
        """
        Plots the diagnostic report with a radar chart for Data Validity,
        a horizontal bar chart for Data Structure, and displays the Overall Score.
        Optionally saves the plot to a specified output path.

        Args:
            output_path (str, optional): Path to save the diagnostic report plots. Defaults to None.
        """
        if self.diagnostic_report is None:
            raise ValueError("No diagnostic report generated. Run 'run_diagnostic' first.")

        # Extract details from the diagnostic report
        if self.method == "realistic" or self.method == "hybrid":
            data_validity = self.diagnostic_report.get_details(property_name='Data Validity')
            data_structure = self.diagnostic_report.get_details(property_name='Data Structure')
            overall_score = self.diagnostic_report.get_score()
        else:
            # For methods without a standard report format (hybrid, causal, knowledge-based)
            data_validity = {'Column': ['N/A'], 'Score': [0.5]}
            data_structure = {'Metric': ['N/A'], 'Score': [0.5]}
            overall_score = 0.5

        # Set up the figure
        fig = plt.figure(figsize=(14, 8))        
        grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])

        # Radar chart for Data Validity
        ax1 = fig.add_subplot(grid[0, 0], polar=True)
        angles = np.linspace(0, 2 * np.pi, len(data_validity), endpoint=False).tolist()
        scores = data_validity['Score'].tolist()
        # Repeat the first value to close the radar chart
        scores += scores[:1]
        angles += angles[:1]

        ax1.fill(angles, scores, color='skyblue', alpha=0.4)
        ax1.plot(angles, scores, color='blue', linewidth=2)
        ax1.set_yticks([0.5, 1.0])
        ax1.set_yticklabels(['0.5', '1.0'])
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(data_validity['Column'], fontsize=10)
        ax1.set_title("Data Validity Radar Chart", fontsize=16)

        # Horizontal bar for Data Structure
        ax2 = fig.add_subplot(grid[0, 1])
        ax2.barh(data_structure['Metric'], data_structure['Score'], color='salmon')
        ax2.set_xlim(0, 1.1)
        ax2.set_xticks([0, 0.5, 1.0])
        ax2.set_title("Data Structure Score", fontsize=16)
        for index, score in enumerate(data_structure['Score']):
            ax2.text(score + 0.01, index, str(np.round(score, 2)), va='center', color='black', fontsize=12)

        # Add legend for color codes
        legend_elements = [
            Patch(facecolor='skyblue', edgecolor='b', label='Data Validity'),
            Patch(facecolor='salmon', edgecolor='r', label='Data Structure')
        ]
        fig.legend(handles=legend_elements, loc='center', fontsize=12)

        # Display the Overall Score at the bottom in green
        fig.text(0.5, 1, f"Overall Diagnostic Score: {np.round(overall_score, 2)}", ha='center', color='green', fontsize=14, fontweight='bold')
        plt.tight_layout()  # Adjust layout to make space for text at the bottom 

        # Save the plot if output_path is provided
        if output_path:
            plt.savefig(os.path.join(output_path, 'diagnostic_report.pdf'), bbox_inches='tight')
            plt.savefig(os.path.join(output_path, 'diagnostic_report.svg'), bbox_inches='tight')
        else:
            plt.savefig('diagnostic_report.pdf', bbox_inches='tight')
            plt.savefig('diagnostic_report.svg', bbox_inches='tight')

        self.fig = fig
        plt.show()

    

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
            <title>Diagnostic Report</title>
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
                <h1>Diagnostic Report</h1>
                {svg_content}
            </div>
        </body>
        </html>
        """

        with open(html_file_path, 'w') as f:
            f.write(html_content)
