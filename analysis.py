import re
import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import fitz
from config import DAMAGE_TYPES

# Initialize models
image_feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
image_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

def parse_pdf_content(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    claim_number = re.search(r'GC\d+', text)
    claim_number = claim_number.group(0) if claim_number else "Unknown"
    
    location = re.search(r'Location:\s*(.+)', text)
    location = location.group(1) if location else "Unknown"
    
    date_of_loss = re.search(r'Created:\s*(.+)', text)
    date_of_loss = date_of_loss.group(1) if date_of_loss else "Unknown"
    
    type_of_loss = re.search(r'(.+)\s*-\s*Water\s*Damage', text)
    type_of_loss = type_of_loss.group(0) if type_of_loss else "Unknown"

    return {
        "claim_number": claim_number,
        "location": location,
        "date_of_loss": date_of_loss,
        "type_of_loss": type_of_loss,
        "full_text": text
    }

def analyze_image(image_path):
    image = Image.open(image_path)
    inputs = image_feature_extractor(images=image, return_tensors="pt")
    outputs = image_model(**inputs)
    predictions = outputs.logits.softmax(1)

    detected_damages = [DAMAGE_TYPES[i] for i, prob in enumerate(predictions[0]) if prob > 0.5]
    return ", ".join(detected_damages) if detected_damages else "No significant damage detected"

def calculate_e3(weather_data, building_data):
    temp_f = weather_data['current']['temp_f']
    humidity = weather_data['current']['humidity']
    room_volume = building_data['room_length'] * building_data['room_width'] * building_data['room_height']

    temp_c = (temp_f - 32) * 5/9
    mass_of_air = room_volume * 0.075
    specific_heat_air = 0.24
    delta_t = temp_f - 70
    sensible_energy = mass_of_air * specific_heat_air * delta_t

    water_vapor_pressure = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    actual_vapor_pressure = water_vapor_pressure * (humidity / 100)
    humidity_ratio = 0.622 * (actual_vapor_pressure / (101.325 - actual_vapor_pressure))
    latent_heat_vaporization = 970
    latent_energy = mass_of_air * humidity_ratio * latent_heat_vaporization

    total_enthalpy = sensible_energy + latent_energy
    e3_value = total_enthalpy / room_volume
    drying_equipment_power = 5000
    dry_time = total_enthalpy / drying_equipment_power

    return {
        "e3_value": e3_value,
        "dry_time": dry_time,
        "sensible_energy": sensible_energy,
        "latent_energy": latent_energy,
        "total_enthalpy": total_enthalpy
    }

def generate_scope_of_work(parsed_data, e3_results, image_analyses):
    sow = f"""
    Scope of Work for {parsed_data['location']}
    Claim Number: {parsed_data['claim_number']}
    Date of Loss: {parsed_data['date_of_loss']}
    Type of Loss: {parsed_data['type_of_loss']}

    E3 Value: {e3_results['e3_value']:.2f}
    Estimated Dry Time: {e3_results['dry_time']:.2f} hours

    Recommended Actions:
    1. Remove and replace water damaged ceiling tiles
    2. Remove lower one foot of drywall in affected areas
    3. Set up drying equipment to achieve E3 target
    4. Clean and disinfect all water-affected surfaces

    Image Analysis Results:
    """
    for i, analysis in enumerate(image_analyses):
        sow += f"\nImage {i+1}: {analysis}"

    return sow

def calculate_costs(sow, rate_sheet):
    # Extract key information from SOW
    ceiling_tile_area = 100  # Placeholder, should be extracted from SOW
    drywall_area = 200  # Placeholder, should be extracted from SOW
    estimated_dry_time = 72  # Placeholder, should be extracted from SOW

    labor_rate = rate_sheet['Flat Rate Restoration'].iloc[0]
    labor_hours = ceiling_tile_area * 0.1 + drywall_area * 0.2 + estimated_dry_time
    labor_cost = labor_hours * labor_rate

    equipment_rate = rate_sheet['Flat Rate Equipment Rental'].iloc[0]
    equipment_days = np.ceil(estimated_dry_time / 24)
    equipment_cost = equipment_days * equipment_rate

    consumables_rate = rate_sheet['Flat Rate Consumables'].iloc[0]
    consumables_units = ceiling_tile_area * 0.05 + drywall_area * 0.1
    consumables_cost = consumables_units * consumables_rate

    total_cost = labor_cost + equipment_cost + consumables_cost

    return {
        "best_case": total_cost * 0.8,
        "typical_case": total_cost,
        "worst_case": total_cost * 1.2
    }
