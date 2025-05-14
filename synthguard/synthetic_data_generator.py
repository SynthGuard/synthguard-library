import math 
from sdv.single_table import GaussianCopulaSynthesizer, CopulaGANSynthesizer
from sdv.multi_table import HMASynthesizer 
from synthguard.privacy_report_generator import PrivacyRiskEvaluator
from synthguard.quality_report_generator import DataQualityEvaluator
import synthguard.helper_functions as hf


class SyntheticDataGenerator:
    def __init__(self, locales='ee_ET', n_rows=1000, output_csv=None, method='hybrid'):
        """
        Initializes the synthetic data generator.

        Args:
            locales (str): Locale for synthetic data generation.
            n_rows (int): Number of rows to generate.
            output_csv (str): Path to save the generated synthetic data.
            method (str): The method to use for generating synthetic data.
        """
        self.locales = locales
        self.n_rows = n_rows
        self.output_csv = output_csv
        self.method = method

    def generate_synthetic_data(self, processed_data, metadata, Nepochs=1):
        """
        Generates synthetic data based on the selected generation method.

        Args:
            processed_data (pd.DataFrame): The preprocessed data.
            metadata (Metadata): The metadata extracted from the real data.

        Returns:
            pd.DataFrame: The generated synthetic data.
        """
        methods = {
            'hybrid': self.hybrid_method,
            'causal': self.causal_method,
            'knowledge-based': self.knowledge_based_method,
            'realistic': lambda data: self.realistic_method(data, metadata, Nepochs),
            'realistic-multi': lambda data: self.realistic_multi_method(data, metadata)
        }
        
        if self.method in methods:
            return methods[self.method](processed_data)
        else:
            raise ValueError(f"Unknown synthetic data generation method: {self.method}")

    def hybrid_method(self, data):
        """Generates synthetic data using the hybrid method."""
        print("Generating synthetic data using the hybrid method...")
        return "synthetic_data_hybrid"  # Replace with actual generation logic

    def causal_method(self, data):
        """Generates synthetic data using the causal method."""
        print("Generating synthetic data using the causal method...")
        return "synthetic_data_causal"  # Replace with actual generation logic

    def knowledge_based_method(self, data):
        """Generates synthetic data using the knowledge-based method."""
        print("Generating synthetic data using the knowledge-based method...")
        return "synthetic_data_knowledge_based"  # Replace with actual generation logic

    def realistic_method(self, data, metadata, Nepochs):
        """
        Generates synthetic data using a realistic method based on Gaussian Copula.

        Args:
            data (pd.DataFrame): The preprocessed data.
            metadata (Metadata): The metadata for the data.

        Returns:
            pd.DataFrame: The generated synthetic data.
        """
        try:
            synthesizer = CopulaGANSynthesizer(
                metadata=metadata,
                enforce_min_max_values=True,
                enforce_rounding=True,
                locales=self.locales,
                epochs=Nepochs,
                verbose=True
            )
            synthesizer.fit(data)
            synthetic_data = synthesizer.sample(num_rows=self.n_rows)
            
            if self.output_csv:
                synthetic_data.to_csv(self.output_csv, index=False)
                print(f"Synthetic data saved to {self.output_csv}")
                
            return synthetic_data

        except Exception as e:
            try:
                synthesizer = GaussianCopulaSynthesizer(
                    metadata=metadata,
                    enforce_min_max_values=True,
                    enforce_rounding=True,
                    # epochs=Nepochs,
                    # verbose=True
                )
                synthesizer.fit(data)
                synthesizer.reset_sampling()
                synthetic_data = synthesizer.sample(num_rows=self.n_rows)
                
                if self.output_csv:
                    synthetic_data.to_csv(self.output_csv, index=False)
                    print(f"Synthetic data saved to {self.output_csv}")
                
                return synthetic_data
            except:
                print(f"An error occurred during synthetic data generation: {e}")
                raise

    def realistic_multi_method(self, data, metadata):
        """
        Generates synthetic data using the HMASynthesizer for multi-table data.

        Args:
            data (pd.DataFrame): The preprocessed data.
            metadata (Metadata): The metadata for the data.

        Returns:
            pd.DataFrame: The generated synthetic data.
        """
        try:
            synthesizer = HMASynthesizer(
                metadata,
                locales=self.locales)
            
            synthesizer.fit(data)
            
            synthetic_data = synthesizer.sample()

            if self.output_csv:
                synthetic_data.to_csv(self.output_csv, index=False)
                print(f"Synthetic data saved to {self.output_csv}")
                
            return synthetic_data
        except Exception as e:
            print(f"An error occurred during synthetic data generation: {e}")
            raise

    def find_closest_dataset(self, real_data, metadata, n_epochs, n_datasets, target_utility, target_privacy):
        privacy_dict = {}
        distance_dict = {}
        
        for i in range(n_datasets):
            dataset = self.generate_synthetic_data(real_data, metadata, n_epochs)
            
            privacyRiskEval = PrivacyRiskEvaluator(real_data, dataset, metadata)
            privacyRiskEval.set_key_target_features()
            privacy_metrics = privacyRiskEval.get_privacy_metrics_realistic()
            privacy_score = privacy_metrics['NewRowSynthesis Score']
            privacy_dict[i] = dataset
            
            utilityEval = DataQualityEvaluator(real_data, dataset, metadata)
            utility_score = utilityEval._calculate_ks_distance()
            print(f'utility: {utility_score}, privacy: {privacy_score}')

            # Cauculate Euclidean distance from target scores
            distance = math.sqrt((privacy_score - target_privacy) ** 2 + (utility_score - target_utility) ** 2)
            distance_dict[i] = distance
            
            print(f'{i+1}/{n_datasets} datasets generated')

        closest_dataset_idx = min(distance_dict, key=distance_dict.get)
        closest_dataset = privacy_dict[closest_dataset_idx]
        print(f'Closest dataset to privacy target {target_privacy} and utility target {target_utility} was found!')

        if self.output_csv:
            closest_dataset.to_csv(self.output_csv, index=False)
            print(f"Synthetic data saved to {self.output_csv}")

        return closest_dataset


