from faker import Faker
from typing import Callable, Any
import pandas as pd
import random
import string
import os
import xmlschema
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
import random
import uuid
import json
import os
from typing import List, Tuple


# pakages to install:
# pip install faker

class PersonalFaker:
    """A cluster for generating Estonian-specific fake data."""

    def __init__(self, locale: str = "et_EE"):
        self.locale = locale
        # Initialize Faker with the Estonian locale
        self.fake = Faker(locale=self.locale)
        # Ensure reproducibility by seeding
        Faker.seed(0)
        
        
    def parse_xml_section(self, root, section):
        """
        Parses a specific section of the XML and extracts it as a list of dictionaries.
        
        Args:
            root: The root element of the XML tree.
            section: The tag name of the section to parse.
        
        Returns:
            List of dictionaries representing the parsed section.
        """
        records = []
        for element in root.findall(f".//{section}"):
            # Extract child elements as key-value pairs
            record = {child.tag: child.text.strip() if child.text else None for child in element}
            records.append(record)
        return records

    def xml_sections_to_dataframes(sel, xml_file, sections):
        """
        Parses specific sections of an XML file and converts them into DataFrames.
        
        Args:
            xml_file: Path to the XML file.
            sections: List of section tag names to parse.
        
        Returns:
            A dictionary of DataFrames for each specified section.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        dataframes = {}
        for section in sections:
            parsed_section = parse_xml_section(root, section)
            dataframes[section] = pd.DataFrame(parsed_section)
        return dataframes

    import zipfile
    import os

    def generate_xml_files_from_template(self, input_path, output_dir, xsd_file_path, xml_file_path,
                                         addresses, municipality_codes, num_certificates, ratings, count=0, rt_ape_energy_ratings=None):
        """
        Generates a zip file containing XML files with modified information based on an existing XML schema and template.
        Args:
            input_path (str): Directory path where the XML schema and template file are located.
            output_dir (str): Directory to save the zip file.
            xsd_file_path (str): Path to the XML schema file.
            xml_file_path (str): Path to the XML template file.
            addresses (list): List of addresses.
            municipality_codes (list): List of municipality codes.
            num_certificates (int): Number of certificates to generate.
            ratings (list): List of energy ratings.
            count (int, optional): Counter for the number of generated files. Defaults to 0.
            rt_ape_energy_ratings (list of tuples, optional): List containing (address, municipality_code, energy_rating).
        """
        # Read the XML schema
        rt_ape_schema = xmlschema.XMLSchema(xsd_file_path)
        # Read the example XML and convert it to a dictionary
        rt_ape_xml = rt_ape_schema.to_dict(xml_file_path, decimal_type=str)
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        valid_energy_ratings = ['A4', 'A3', 'A2', 'A1', 'B', 'C', 'D', 'E', 'F', 'G']
        # Generate energy ratings if not provided
        if not rt_ape_energy_ratings:
            print("Generating energy ratings...")
            rt_ape_energy_ratings = self.generate_energy_ratings(addresses, municipality_codes, num_certificates, ratings)

        zip_file = os.path.join(output_dir, "rt-ape-energy-ratings.zip")
        # Create a zip file to store the XML files
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for address, municipality_code, energy_rating in rt_ape_energy_ratings:
                # Validate energy_rating
                if energy_rating not in valid_energy_ratings:
                    raise ValueError(f"Invalid energy rating '{energy_rating}' for address '{address}'. Must be one of {valid_energy_ratings}")

                # Convert municipality_code to string and pad to 6 digits
                municipality_code = str(municipality_code).zfill(6)

                # Make a deep copy of the template dictionary to avoid modifying the original
                new_xml = rt_ape_xml.copy()
                # Replace target values based on parsed sections
                general_data_id = new_xml["datiGenerali"]["datiIdentificativi"]
                energetic_classification = new_xml["prestazioneGlobale"]["prestazioneEnergeticaGlobale"]["classificazione"]
                # Ensure that 'indirizzo' is a string and 'codiceISTAT' is padded to 6 digits
                general_data_id["indirizzo"] = str(address).replace("\"", "")
                general_data_id["codiceISTAT"] = municipality_code  # Ensure it's always a 6-digit string
                # Ensure that the energy classification is valid and numeric
                energetic_classification["classeEnergetica"] = energy_rating
                # Convert the modified dictionary back to an XML ElementTree
                etree = rt_ape_schema.to_etree(new_xml)
                # Wrap the Element in an ElementTree to be able to use the 'write' method
                tree = ET.ElementTree(etree)
                # Define the output file name for the XML inside the zip archive
                zip_file_name = f"rt-ape-{abs(hash(address))}-{municipality_code}.xml"
                # Add the XML file directly to the zip file
                with zipf.open(zip_file_name, 'w') as f:
                    tree.write(f, encoding="utf-8", xml_declaration=True)
                count += 1
                print(f"Counts of Added to zip: {count}")

        print(f"Zipped: {zip_file}")
        return zip_file
        
        
    # Generate energy rating certificates
    def generate_energy_ratings(self, addresses, municipality_codes, num_certificates, ratings):
        """
        Generate random energy ratings for a set of addresses.

        :param addresses: List of addresses.
        :param municipality_codes: List of municipality codes corresponding to the addresses.
        :param num_certificates: Number of certificates to generate.
        :param ratings: List of possible energy ratings.
        :return: List of tuples (address, municipality_code, energy_rating).
        """
        if len(addresses) != len(municipality_codes):
            raise ValueError("Addresses and municipality codes must have the same length.")

        if num_certificates > len(addresses):
            raise ValueError("Number of certificates cannot exceed the number of addresses.")

        selected_samples = random.sample(list(zip(addresses, municipality_codes)), num_certificates)
        return [
            (address, municipality_code, random.choice(ratings))
            for address, municipality_code in selected_samples
        ]
        
        
    def generate_code(self, length: int, char_set: str = string.ascii_uppercase + string.digits) -> str:
        """
        Generate a random code of a given length using a specified character set.

        :param length: Length of the generated code
        :param char_set: Set of characters to choose from (default: uppercase letters and digits)
        :return: Randomly generated code
        """
        return ''.join(random.choices(char_set, k=length))
    
    def generate_cadastre_and_thermal_units(self, n_addresses: int, n_reports_per_address: int) -> list:
        """
        Generate unique cadastre codes and thermal unit codes for each address.

        :param n_addresses: Number of addresses
        :param n_reports_per_address: Number of reports per address
        :return: List of tuples containing cadastre_code and thermal_unit
        """
        
        # Helper functions to generate cadastre code and thermal unit code
        def generate_cadastre_code(length: int = 8) -> str:
            """Generate a random cadastre code."""
            return ''.join(random.choices(string.digits, k=length))

        def generate_thermal_unit_code(length: int = 6) -> str:
            """Generate a random thermal unit code."""
            return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

        # Initialize sets and list for cadastre codes and thermal unit mappings
        cadastre_codes = set()
        cadastre_codes_thermal_units = []

        # Generate unique cadastre codes for each address
        while len(cadastre_codes) < n_addresses:
            cadastre_codes.add(generate_cadastre_code())

        # Generate unique thermal unit codes for each cadastre code
        for cadastre_code in cadastre_codes:
            thermal_units = set()
            while len(thermal_units) < n_reports_per_address:
                thermal_units.add(generate_thermal_unit_code())
            
            for thermal_unit in thermal_units:
                cadastre_codes_thermal_units.append((cadastre_code, thermal_unit))

        return cadastre_codes_thermal_units
    

    def generate_rt_cit_thermal_group(self, synthetic_addresses: pd.DataFrame, cadastre_codes_thermal_units: list, n_reports_per_address: int) -> list:
        """
        Process synthetic addresses and generate a list of thermal group data.

        :param synthetic_addresses: DataFrame containing address data
        :param cadastre_codes_thermal_units: List of tuples containing cadastre codes and thermal units
        :param n_reports_per_address: Number of reports per address
        :return: List of tuples containing cadastre_code, thermal_unit, address, municipality_code, and combustion_efficiency
        """
        rt_cit_thermal_group = []
        
        # Iterate over addresses and assign thermal units
        for i, address in enumerate(synthetic_addresses['address']):
            # Slice cadastre codes and thermal units
            slice_cctu = cadastre_codes_thermal_units[i*n_reports_per_address:(i+1)*n_reports_per_address]
            
            # Split address and extract municipality code
            parts = address.split(",")
            if len(parts) >= 4:
                municipality_code = parts[-2].strip()  # Extract municipality code
                full_address = ", ".join(parts[:-2])  # Address excluding municipality and postal code
            else:
                municipality_code = "Unknown"  # Fallback if no municipality code
                full_address = ", ".join(parts)  # Use full address if municipality is not available

            # Append thermal group data for each cadastre code and thermal unit
            for cadastre_code, thermal_unit in slice_cctu:
                rt_cit_thermal_group.append(
                    (
                        cadastre_code, 
                        thermal_unit, 
                        full_address, 
                        municipality_code, 
                        random.uniform(0, 1)  # Combustion efficiency between 0 and 1
                    )
                )
        
        return pd.DataFrame(rt_cit_thermal_group, columns=["cadastre_code", "thermal_unit", "address", "municipality_code", "combustion_efficiency"])
        
    def generate_data_addresses(self, street_names: pd.Series, municipality_codes: pd.Series, n_addresses: int = 10) -> pd.DataFrame:
        """Generate synthetic addresses based on street names and municipality codes."""
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
        addresses = []
        for _ in range(n_addresses):
            street_name = random.choice(street_names)
            municipality_code = random.choice(municipality_codes)
            house_number = self.fake.building_number()
            city = self.fake.city()  # Correct method for generating city names
            postal_code = self.fake.postcode()

            # Format the address
            address = {
                "id": len(addresses) + 1,
                "address": f"{street_name}, {house_number}, {city}, {municipality_code}, {postal_code}",
            }
            addresses.append(address)

        # Convert to DataFrame for better handling
        addresses_df = pd.DataFrame(addresses)
        return addresses_df

    def license_plate(self) -> str:
        """Generate an Estonian license plate."""
        return self.fake.license_plate()

    def vin(self) -> str:
        """Generate a VIN number."""
        return self.fake.vin()

    def first_name(self) -> str:
        """Generate a random first name."""
        return self.fake.first_name()
    
    
    def generate_local_addresses(self, n_addresses: int) -> List[Tuple[str, str]]:
        """Generate local addresses."""
        addresses = []
        for _ in range(n_addresses):
            street_address = self.fake.street_address()
            city = self.fake.city()
            addresses.append((street_address, city))
        return addresses
    
    
    def parse_address(self, address: Tuple[str, str]) -> dict:
        """Parse an address into structured information."""
        street, municipality = address
        return {
            "street": street,
            "municipality": municipality,
        }
        
    def generate_location_key(self, address: Tuple[str, str], date: str) -> str:
        """Generate a unique location key."""
        parsed = self.parse_address(address)
        return f"{parsed['street'].replace(' ', '_')}-{parsed['municipality'].replace(' ', '_')}-{date.replace('-', '')}"

        
    def generate_base_record(self, address: Tuple[str, str], date: str, time_offset: int) -> dict:
        """Generate a base record with static and dynamic information."""
        timestamp = f"{date}T{time_offset:02}:00:00.000Z"
        return {
            "transactionId": str(uuid.uuid4()),
            "timestamp": timestamp,
            "timestampLocal": timestamp[:-1],  # Remove trailing 'Z' for local timestamp
            "customerKey": "xf1iyhmkay",
            "customerName": "Teadal",
            "locationKey": self.generate_location_key(address, date),
            "locationName": f"{address[0]}, {address[1]}",
            "deviceKey": "ltwnaupx72",
            "deviceName": "teadal-1",
            "deviceVendor": "BOX2M-M0",
            "deviceModel": None,
            "circuitKey": "otibdvn2ri",
            "circuitName": "circuit-1",
            "sensorVendor": "Lovato",
            "sensorModel": "DMG110",
            "cost": None,
            "costCurrency": None,
            "carbon": None,
        }
        
    def generate_data_for_address(self, address: Tuple[str, str], n_records_per_day: int, n_box2m_records: int, box2m_units: List[dict]) -> List[dict]:
        """Generate measurement records for a single address."""
        records = []
        start_date = datetime.now()
        hours_per_record = 24 // n_records_per_day

        for day in range(n_box2m_records):
            date = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")

            for time_index in range(n_records_per_day):
                base_record = self.generate_base_record(address, date, time_index * hours_per_record)

                for unit in box2m_units:
                    unit_record = base_record.copy()
                    unit_record.update(unit)
                    unit_record["value"] = round(random.uniform(0, 1000), 5)
                    unit_record["raw"] = str(unit_record["value"]).replace(".", "").ljust(7, "0")
                    records.append(unit_record)

        return records
    
    def generate_box2m_data(self, n_addresses: int, output_dir: str, n_records_per_day: int, n_box2m_records: int, box2m_units: List[dict]):
        """Main function to generate and save data for multiple addresses."""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        addresses = self.generate_local_addresses(n_addresses)

        for address in addresses:
            records = self.generate_data_for_address(address, n_records_per_day, n_box2m_records, box2m_units)
            location_key = self.generate_location_key(address, datetime.now().strftime("%Y-%m-%d"))
            file_path = os.path.join(output_dir, f"{location_key}.json")
            self.save_records_to_file(records, file_path)
        

        print(f"Data generation completed. Files saved in {output_dir}.")
        return file_path
    
    def save_records_to_file(self, records: List[dict], file_path: str):
        """Save records to a JSON file."""
        with open(file_path, "w") as file:
            json.dump(records, file, indent=4)
    
    def generate_local_addresses(self, n_addresses: int) -> list:
        """
        Generate and parse a specified number of Italian synthetic addresses
        to extract street and municipality information.
        
        Args:
            n_addresses (int): Number of synthetic addresses to generate.

        Returns:
            list: A list of tuples containing street and municipality information.
        """
        # Create an Italian Faker instance
        italian_fake = Faker(locale=self.locale)
        
        # Generate synthetic Italian addresses
        synthetic_addresses = [italian_fake.address() for _ in range(n_addresses)]
        
        # Parse each address to extract street and municipality
        parsed_addresses = []
        for address in synthetic_addresses:
            lines = address.split("\n")  # Split the address into lines
            street = lines[0]  # First line is the street
            if len(lines) > 1:  # Second line contains postal code and municipality
                postal_and_city = lines[1]
                # Extract municipality by splitting on comma or space
                if "," in postal_and_city:
                    municipality = postal_and_city.split(",")[1].strip()
                else:
                    municipality = postal_and_city.split(" ")[-1].strip()
            else:
                municipality = "Unknown"  # Handle cases with missing municipality info
            parsed_addresses.append((street, municipality))
        
        return parsed_addresses

    

    def generate_random_date(self, start_date: str, end_date: str) -> date:
        """
        Generate a random date between two given dates.

        Parameters:
        - start_date (str): Start date in the format 'YYYY-MM-DD'.
        - end_date (str): End date in the format 'YYYY-MM-DD'.

        Returns:
        - date: Random date between start_date and end_date.
        """
        # Convert start_date and end_date to datetime.date
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()

        # Generate a random date between start_date and end_date
        return self.fake.date_between(start_date=start_date_obj, end_date=end_date_obj)

    def generate_random_dates(self, n_dates: int, start_date: str, end_date: str) -> list:
        """
        Generate a list of random dates between two given dates.

        Parameters:
        - n_dates (int): Number of random dates to generate.
        - start_date (str): Start date in the format 'YYYY-MM-DD'.
        - end_date (str): End date in the format 'YYYY-MM-DD'.

        Returns:
        - list: List of random dates.
        """
        return [self.generate_random_date(start_date, end_date) for _ in range(n_dates)]


    # write a function to generate a random administrative unit up to certain number of administrative units
    def generate_administrative_units(self, n_administrative_units: int) -> list:
        """Generate a list of random administrative units."""
        return [self.generate_administrative_unit() for _ in range(n_administrative_units)]
    
    def generate_administrative_unit(self) -> str:
        """Generate a random Estonian administrative unit."""
        return self.fake.administrative_unit()

    def first_name_est(self) -> str:
        """Generate a random Estonian first name."""
        return self.fake.first_name_est()

    def first_name_female(self) -> str:
        """Generate a random female first name."""
        return self.fake.first_name_female()

    def first_name_female_est(self) -> str:
        """Generate a random Estonian female first name."""
        return self.fake.first_name_female_est()

    def first_name_female_rus(self) -> str:
        """Generate a random Russian female first name."""
        return self.fake.first_name_female_rus()

    def first_name_male(self) -> str:
        """Generate a random male first name."""
        return self.fake.first_name_male()

    def first_name_male_est(self) -> str:
        """Generate a random Estonian male first name."""
        return self.fake.first_name_male_est()

    def first_name_male_rus(self) -> str:
        """Generate a random Russian male first name."""
        return self.fake.first_name_male_rus()

    def first_name_nonbinary(self) -> str:
        """Generate a random non-binary first name."""
        return self.fake.first_name_nonbinary()

    def first_name_rus(self) -> str:
        """Generate a random Russian first name."""
        return self.fake.first_name_rus()

    def language_name(self) -> str:
        """Generate a random language name."""
        return self.fake.language_name()

    def last_name(self) -> str:
        """Generate a random last name."""
        return self.fake.last_name()

    def last_name_est(self) -> str:
        """Generate a random Estonian last name."""
        return self.fake.last_name_est()

    def last_name_female(self) -> str:
        """Generate a random female last name."""
        return self.fake.last_name_female()

    def last_name_male(self) -> str:
        """Generate a random male last name."""
        return self.fake.last_name_male()

    def last_name_nonbinary(self) -> str:
        """Generate a random non-binary last name."""
        return self.fake.last_name_nonbinary()

    def last_name_rus(self) -> str:
        """Generate a random Russian last name."""
        return self.fake.last_name_rus()

    def name(self) -> str:
        """Generate a random full name."""
        return self.fake.name()

    def name_female(self) -> str:
        """Generate a random female full name."""
        return self.fake.name_female()

    def name_male(self) -> str:
        """Generate a random male full name."""
        return self.fake.name_male()

    def name_nonbinary(self) -> str:
        """Generate a random non-binary full name."""
        return self.fake.name_nonbinary()

    def prefix(self) -> str:
        """Generate a random prefix."""
        return self.fake.prefix()

    def prefix_female(self) -> str:
        """Generate a random female prefix."""
        return self.fake.prefix_female()

    def prefix_male(self) -> str:
        """Generate a random male prefix."""
        return self.fake.prefix_male()

    def prefix_nonbinary(self) -> str:
        """Generate a random non-binary prefix."""
        return self.fake.prefix_nonbinary()

    def suffix(self) -> str:
        """Generate a random suffix."""
        return self.fake.suffix()

    def suffix_female(self) -> str:
        """Generate a random female suffix."""
        return self.fake.suffix_female()

    def suffix_male(self) -> str:
        """Generate a random male suffix."""
        return self.fake.suffix_male()

    def suffix_nonbinary(self) -> str:
        """Generate a random non-binary suffix."""
        return self.fake.suffix_nonbinary()

    def ssn(self, min_age: int = 16, max_age: int = 90) -> str:
        """Generate a random Estonian personal identity code (IK)."""
        return self.fake.ssn(min_age=min_age, max_age=max_age)

    def vat_id(self) -> str:
        """Generate a random Estonian VAT ID."""
        return self.fake.vat_id()

    def all_generators(self) -> dict[str, Callable[[], Any]]:
        """Get a dictionary of all generator functions in this class."""
        return {
            method_name: method
            for method_name, method in self.__class__.__dict__.items()
            if callable(method) and not method_name.startswith("_") and method_name != "all_generators"
        }
        
        
        


# # Map user-friendly input to generator methods
# GENERATOR_MAP = {
#     "license plate": "license_plate",
#     "vin": "vin",
#     "first name": "first_name",
#     "estonian first name": "first_name_est",
#     "last name": "last_name",
#     "full name": "name",
#     "ssn": "ssn",
#     "vat id": "vat_id",
# }


# def generate_user_data(requests: list[str], faker_instance: PersonalFaker, rows: int = 1) -> pd.DataFrame:
#     """Generate user-requested fake data as a DataFrame.

#     Args:
#         requests: List of user-friendly data requests.
#         faker_instance: An instance of PersonalFaker.
#         rows: Number of rows of data to generate.

#     Returns:
#         A DataFrame with the requested fake data.
#     """
#     generators = faker_instance.all_generators()
#     data = {request: [] for request in requests}

#     for _ in range(rows):
#         for request in requests:
#             method_name = GENERATOR_MAP.get(request.lower())
#             if method_name and method_name in generators:
#                 try:
#                     data[request].append(generators[method_name](faker_instance))
#                 except Exception as e:
#                     data[request].append(f"Error: {e}")
#             else:
#                 data[request].append("Unknown request")

#     return pd.DataFrame(data)

# # Example usage
# if __name__ == "__main__":
#     # Create an instance of PersonalFaker
#     estonian_fake = PersonalFaker("et_EE")

#     # Example user requests
#     user_requests = ["license plate", "vin", "estonian first name", "ssn", "vat id"]

#     # Generate 100 rows of data
#     num_rows = 100
#     user_data_df = generate_user_data(user_requests, estonian_fake, rows=num_rows)

#     # Print the resulting DataFrame
#     print(user_data_df)

#     # Optionally, save the DataFrame to a CSV file
#     # user_data_df.to_csv("user_requests_data.csv", index=False)
