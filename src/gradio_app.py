import gradio as gr # type: ignore
import pandas as pd # type: ignore
import json
import os
import torch # type: ignore
import logging
from dotenv import load_dotenv # type: ignore
from weather import retrieve_weather_and_humidity
from e3_calculations import calculate_e3_dry_times, load_model
from cost_calculations import calculate_costs
from custom_exceptions import WeatherAPIError, ModelLoadError # type: ignore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load configuration
try:
    with open('config/config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    logger.error("Config file not found. Please ensure config/config.json exists.")
    raise
except json.JSONDecodeError:
    logger.error("Error decoding config file. Please check the JSON format.")
    raise

# Use environment variables
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
LLM_URL = os.getenv('LLM_URL', 'http://localhost:1234/v1/chat/completions')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'mathstral-7b-v0.1-q4k_m')

# Update config with environment variables
config['weather_api']['api_key'] = WEATHER_API_KEY
config['llm_service']['url'] = LLM_URL
config['llm_service']['model_name'] = LLM_MODEL_NAME

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def validate_and_process_files(csv_file):
    """
    Validates and processes the uploaded CSV file for E3 drying time calculations.
    """
    try:
        csv_data = pd.read_csv(csv_file.name)
    except Exception as e:
        logger.error(f"Failed to load CSV file: {str(e)}")
        return None, f"Failed to load CSV file: {e}"

    # Required columns for the calculations
    required_columns = [
        "County", "State", "Room_ID", "Room_Name", "Size of Room",
        "ambient_temp", "dew_point_temp", "wet_bulb_temp", "gpp_ambient",
        "gpp_target", "e3_days"
    ]

    # Check if required columns are present
    missing_columns = [col for col in required_columns if col not in csv_data.columns]
    if missing_columns:
        logger.error(f"Missing columns in CSV file: {missing_columns}")
        return None, f"Missing columns in CSV file: {', '.join(missing_columns)}"

    # Additional validation can be added here if needed

    return csv_data, None

# File: src/gradio_app.py (Continuation)

def validate_config(config):
    """
    Validates the loaded configuration to ensure all necessary keys and values are present.
    """
    required_keys = ['weather_api', 'llm_service']
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration section: {key}")
            raise ValueError(f"Missing required configuration section: {key}")

    if not config['weather_api'].get('api_key'):
        logger.error("Missing Weather API key in configuration.")
        raise ValueError("Missing Weather API key in configuration.")
    
    if not config['llm_service'].get('url'):
        logger.error("Missing LLM service URL in configuration.")
        raise ValueError("Missing LLM service URL in configuration.")

    # Add further validation as needed

# Validate the configuration after loading
validate_config(config)
