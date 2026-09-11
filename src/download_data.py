"""
Download and Data Ingestion Module for CNC Machining Power Prediction.
This script acquires, downloads from remote URLs/mirrors, verifies, and validates the machining dataset.
Every single line of code is documented with inline comments explaining its engineering rationale.
"""

# Import operating system interfaces for directory and file path management
import os

# Import system-specific parameters and functions to modify module search paths
import sys

# Import urllib.request to download datasets directly from remote web links / URLs
import urllib.request

# Import argparse to parse command-line arguments such as download links
import argparse

# Import pathlib for object-oriented filesystem paths
from pathlib import Path

# Import pandas library for tabular data structures and CSV reading/writing operations
import pandas as pd

# Import numpy for numerical operations and array sanity checks
import numpy as np

# Determine the absolute directory path of the current Python script
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent

# Determine the project root directory by navigating one level up from src
PROJECT_ROOT_DIR = CURRENT_SCRIPT_DIR.parent

# Check if the project root directory is already present in the Python system path
if str(PROJECT_ROOT_DIR) not in sys.path:
    # Prepend the project root directory to sys.path to enable absolute imports from src
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Import the physics-based dataset generator function from src/dataset_generator.py
from src.dataset_generator import generate_machining_dataset

# Define the standard relative destination path for the raw CSV dataset
RAW_DATASET_REL_PATH = os.path.join("data", "raw", "machining_power_consumption_12k.csv")

# Resolve the full absolute path for the raw CSV dataset file
RAW_DATASET_FULL_PATH = os.path.join(PROJECT_ROOT_DIR, RAW_DATASET_REL_PATH)

# Public repository dataset download mirror URLs
PUBLIC_DATASET_URLS = [
    # Primary GitHub Raw repository dataset link
    "https://raw.githubusercontent.com/nagarajuboddu2004-web/prediction-of-power-consumption-during-machining-using-machine-learning-model-technic/main/data/raw/machining_power_consumption_12k.csv",
    # Alternative open-access mirror
    "https://raw.githubusercontent.com/datasets/cnc-machining-power/main/machining_power_consumption_12k.csv"
]

# Define expected essential columns that must exist in the dataset for machine learning
REQUIRED_DATASET_COLUMNS = [
    "Operation_Type",                  # Machining operation (e.g., CNC Milling, CNC Turning)
    "Workpiece_Material",              # Alloy designation (e.g., Ti-6Al-4V, AISI 1045)
    "Material_Hardness_HB",            # Workpiece Brinell hardness number
    "Tool_Diameter_mm",                # Cutting tool diameter in millimeters
    "Number_of_Flutes",                # Number of cutting teeth on the tool
    "Tool_Wear_VB_mm",                 # Flank wear land width in millimeters
    "Rake_Angle_deg",                  # Tool rake angle in degrees
    "Cutting_Speed_vc_mpm",            # Peripheral cutting speed in meters per minute
    "Spindle_Speed_RPM",               # Rotational spindle frequency in revolutions per minute
    "Feed_Rate_mm_rev",                # Feed per revolution (or feed per tooth in milling)
    "Axial_Depth_ap_mm",               # Axial depth of cut in millimeters
    "Radial_Depth_ae_mm",              # Radial width of cut in millimeters
    "Feed_Speed_vf_mmpm",              # Linear table feed rate in millimeters per minute
    "Coolant_Condition",               # Cooling mode (Dry, Flood, MQL, Cryogenic)
    "Coolant_Flow_Rate_Lpm",           # Lubricant volumetric flow rate in liters/min
    "Tool_Coating",                    # Thin film coating type (TiAlN, AlCrN, etc.)
    "Material_Removal_Rate_cm3_min",   # Volumetric chip removal rate in cm^3/min
    "Cutting_Temperature_C",           # Cutting zone thermo-mechanical temperature in degrees Celsius
    "Power_Consumption_kW",            # Target active electrical power draw in kilowatts
    "Carbon_Emission_Rate_kgCO2e_hr"   # Total carbon emission rate in kg CO2e per hour
]


def verify_dataset_integrity(df: pd.DataFrame) -> bool:
    """
    Validates that the provided DataFrame adheres to required physics and quality constraints.
    Returns True if valid, raises ValueError or AssertionError otherwise.
    """
    # Verify that the DataFrame is not completely empty
    if df.empty:
        # Raise an exception if the dataset contains zero records
        raise ValueError("Integrity Error: The provided dataset DataFrame is empty.")
    
    # Iterate through all mandatory column names defined in REQUIRED_DATASET_COLUMNS
    for col_name in REQUIRED_DATASET_COLUMNS:
        # Check if the mandatory column is present in the DataFrame columns
        if col_name not in df.columns:
            # Raise an exception indicating which expected column is missing
            raise ValueError(f"Integrity Error: Missing required column '{col_name}' in dataset.")
    
    # Calculate the total count of null or NaN values across the entire DataFrame
    null_value_count = df.isnull().sum().sum()
    
    # Verify that there are zero missing or null entries in the dataset
    if null_value_count > 0:
        # Raise an exception if missing data is detected
        raise ValueError(f"Integrity Error: Dataset contains {null_value_count} null/NaN values.")
    
    # Check that power consumption values are strictly positive physical values
    if (df["Power_Consumption_kW"] <= 0).any():
        # Raise an exception if zero or negative power values exist
        raise ValueError("Integrity Error: Non-positive power consumption values detected.")
    
    # Check that cutting speed is strictly positive
    if (df["Cutting_Speed_vc_mpm"] <= 0).any():
        # Raise an exception if non-positive cutting speed is detected
        raise ValueError("Integrity Error: Non-positive cutting speed values detected.")

    # Check that carbon emissions are non-negative physical values
    if (df["Carbon_Emission_Rate_kgCO2e_hr"] < 0).any():
        # Raise an exception if negative carbon emission rate is detected
        raise ValueError("Integrity Error: Negative carbon emission values detected.")
    
    # Print confirmation that all integrity checks passed
    print(f"[OK] Integrity checks passed: {len(df):,} samples, {len(df.columns)} columns, zero nulls.")
    
    # Return True indicating successful validation
    return True


def download_from_url(url: str, target_filepath: str) -> bool:
    """
    Downloads a dataset directly from an external HTTP/HTTPS link to the local destination path.
    Includes custom User-Agent headers and network exception handling.
    """
    # Print download initiation message showing source URL
    print(f"[DOWNLOAD] Initiating download from URL link:\n  {url}")
    
    # Configure custom HTTP request headers to simulate a modern browser agent
    request_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # Create urllib Request object with target URL and headers
    req = urllib.request.Request(url, headers=request_headers)
    
    # Open remote connection within a safe try-except block
    try:
        # Open URL and read response stream
        with urllib.request.urlopen(req, timeout=30) as response:
            # Check HTTP response status code
            if response.status == 200:
                # Read binary payload content
                content_bytes = response.read()
                
                # Check if downloaded content has non-trivial size (>1000 bytes)
                if len(content_bytes) > 1000:
                    # Write downloaded bytes directly to destination file
                    with open(target_filepath, "wb") as f_out:
                        f_out.write(content_bytes)
                    # Print download success confirmation
                    print(f"[DOWNLOAD SUCCESS] Downloaded {len(content_bytes):,} bytes from link.")
                    return True
                else:
                    # Warn if payload size is unexpectedly small
                    print(f"[DOWNLOAD WARNING] Downloaded payload too small ({len(content_bytes)} bytes).")
                    return False
            else:
                # Print status error if HTTP response is not 200 OK
                print(f"[DOWNLOAD FAILED] Remote server responded with HTTP status: {response.status}")
                return False
    except Exception as err:
        # Catch and report any network, timeout, or DNS exceptions
        print(f"[DOWNLOAD EXCEPTION] Could not download from link: {err}")
        return False


def download_or_generate_dataset(
    target_filepath: str = RAW_DATASET_FULL_PATH,
    download_url: str = None,
    sample_count: int = 12500,
    random_seed: int = 42,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Downloads or synthesizes the CNC machining power consumption dataset.
    1. If download_url is provided, attempts to download directly from the link.
    2. If target_filepath already exists and force_refresh is False, loads existing data.
    3. If file missing and download fails or is omitted, synthesizes 12,500 physics observations.
    """
    # Extract the directory portion of the target file path
    target_directory = os.path.dirname(target_filepath)
    
    # Create the destination directory if it does not already exist on disk
    os.makedirs(target_directory, exist_ok=True)
    
    # Flag to track whether download succeeded
    download_successful = False
    
    # If an explicit download URL was provided by the user
    if download_url:
        # Attempt download from user-specified URL
        download_successful = download_from_url(download_url, target_filepath)
        
    # Check whether the dataset file already exists and if a forced refresh is not requested
    if os.path.exists(target_filepath) and not force_refresh:
        # Log that an existing dataset was found on the local filesystem
        print(f"[INFO] Using verified dataset at: {target_filepath}")
        
        # Read the existing CSV file into a pandas DataFrame
        df_loaded = pd.read_csv(target_filepath)
        
        # Validate the integrity and schema of the existing dataset
        verify_dataset_integrity(df_loaded)
        
        # Return the verified existing DataFrame
        return df_loaded
    
    # If download was not executed or failed, attempt primary public mirror links
    if not download_successful:
        for mirror_link in PUBLIC_DATASET_URLS:
            print(f"[INFO] Attempting download from public repository mirror link...")
            if download_from_url(mirror_link, target_filepath):
                try:
                    df_mirror = pd.read_csv(target_filepath)
                    verify_dataset_integrity(df_mirror)
                    return df_mirror
                except Exception:
                    pass
                    
    # Print status message indicating that physics-based dataset synthesis is beginning
    print(f"[INFO] Generating {sample_count:,} physics-grounded machining observations...")
    
    # Call the multi-physics generator function parameterized by sample count and random seed
    df_generated = generate_machining_dataset(num_samples=sample_count, random_seed=random_seed)
    
    # Verify the integrity and schema of the newly generated dataset
    verify_dataset_integrity(df_generated)
    
    # Save the generated DataFrame to the target CSV file path without row indices
    df_generated.to_csv(target_filepath, index=False)
    
    # Print status message showing the successful write operation to the file system
    print(f"[SUCCESS] Dataset successfully saved to: {target_filepath}")
    
    # Return the newly generated DataFrame
    return df_generated


def main():
    """
    Main entry point for command-line execution of the dataset downloader/generator.
    Supports --url for direct remote download link.
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Download or Ingest CNC Machining Power Dataset")
    
    # Add optional argument for remote download link
    parser.add_argument("--url", type=str, default=None, help="Remote HTTP/HTTPS download link for raw CSV dataset")
    
    # Add optional argument for sample count if generating
    parser.add_argument("--samples", type=int, default=12500, help="Number of observations to generate if not downloading")
    
    # Add optional flag to force re-download / re-generation
    parser.add_argument("--force", action="store_true", help="Force refresh even if dataset file already exists")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Print banner separator line
    print("=" * 70)
    
    # Print script title header
    print("      CNC MACHINING DATA INGESTION & DOWNLOAD MODULE (src/download_data.py)")
    
    # Print banner separator line
    print("=" * 70)
    
    # Display download link information
    if args.url:
        print(f"[LINK] User-Specified Download Link: {args.url}")
    else:
        print(f"[LINK] Default Public Mirror Link: {PUBLIC_DATASET_URLS[0]}")
        
    # Execute the download/generation procedure with parsed parameters
    df_data = download_or_generate_dataset(
        target_filepath=RAW_DATASET_FULL_PATH,
        download_url=args.url,
        sample_count=args.samples,
        random_seed=42,
        force_refresh=args.force
    )
    
    # Print dataset summary statistics header
    print("\n--- Dataset Summary Overview ---")
    
    # Print total number of rows (samples) in the dataset
    print(f"Total Machining Observations : {df_data.shape[0]:,}")
    
    # Print total number of feature columns
    print(f"Total Parameter Columns      : {df_data.shape[1]}")
    
    # Print the distribution of workpiece materials in the dataset
    print("\nWorkpiece Material Distribution:")
    
    # Print value counts of workpiece materials with indentation
    print(df_data["Workpiece_Material"].value_counts().to_string())
    
    # Print the distribution of machining operations
    print("\nOperation Types Distribution:")
    
    # Print value counts of machining operations
    print(df_data["Operation_Type"].value_counts().to_string())
    
    # Print numerical power consumption descriptive statistics
    print("\nTarget Electrical Power Statistics (kW):")
    
    # Print mean, std, min, 25%, 50%, 75%, and max of Power_Consumption_kW
    print(df_data["Power_Consumption_kW"].describe().to_string())
    
    # Print completion confirmation message
    print("\n" + "=" * 70)
    print("      DATA INGESTION AND VALIDATION COMPLETE SUCCESSFULLY")
    print("=" * 70)


# Standard Python construct to execute main() only when script is run directly
if __name__ == "__main__":
    # Call the main entry point function
    main()
