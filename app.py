import os
import pandas as pd
import numpy as np
import logging
from PIL import Image
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, ViTForImageClassification, ViTImageProcessor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage
from reportlab.lib.units import inch
from io import BytesIO
import base64
from datetime import datetime
import json
import fitz  # PyMuPDF
from pytesseract import image_to_string
from openai import OpenAI
import torch
from dotenv import load_dotenv

load_dotenv() 

# Load environment variables from config.json
try:
    with open('config.json') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    raise FileNotFoundError("config.json file not found. Please ensure it exists in the current directory.")
except json.JSONDecodeError:
    raise ValueError("config.json is not a valid JSON file. Please check its contents.")

hf_api_key = config['huggingface_api_key']
openai_api_key = config['openai_api_key']

# Replace [CURRENT_DATE] in the report output path
current_date = datetime.now().strftime('%Y-%m-%d')
report_output_path = config['report_output_path'].replace('[CURRENT_DATE]', current_date)

# Ensure output directory exists
output_dir = os.path.dirname(report_output_path)
os.makedirs(output_dir, exist_ok=True)

# Configure logging
logging.basicConfig(filename='app.log', level=config['logging_level'], format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize models
try:
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=hf_api_key)
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=hf_api_key)
    
    # Initialize vision model for image analysis
    vision_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    vision_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
except Exception as e:
    logging.error(f"Error initializing models: {e}")
    raise

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# E3 Calculation Logic
def calculate_e3(gpp_ambient, gpp_dew_point, gpp_wet_bulb, outdoor_temp, outdoor_humidity):
    # Constants
    A = 0.0012
    B = 0.1540
    C = 0.0650
    D = 0.0309
    
    # Calculate E3
    e3 = (A * gpp_ambient + B * gpp_dew_point + C * gpp_wet_bulb) / (D * outdoor_temp * outdoor_humidity)
    
    return e3

def process_extracted_data_with_llama(extracted_text):
    try:
        inputs = llama_tokenizer(extracted_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = llama_model.generate(**inputs, max_length=1000)
        processed_data = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logging.info("Processed extracted data with Llama-2 successfully.")
        return processed_data
    except Exception as e:
        logging.error(f"Error processing extracted data with Llama-2: {e}")
        raise

def perform_image_analysis(image_path):
    try:
        image = Image.open(image_path)
        inputs = vision_processor(images=image, return_tensors="pt")
        outputs = vision_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
        # Get the label for the predicted class
        predicted_label = vision_model.config.id2label[predicted_class_idx]
        
        # Use GPT-4 to generate a detailed analysis based on the image classification
        prompt = f"Analyze water damage in a room based on the image classification '{predicted_label}'. Provide potential remedies and considerations for restoration."
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in water damage assessment and remediation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        analysis = response.choices[0].message.content.strip()
        
        logging.info(f"Image analysis completed for {image_path}")
        return analysis
    except Exception as e:
        logging.error(f"Error performing image analysis: {e}")
        raise

def calculate_room_specific_e3(csv_file_path, image_folder):
    try:
        df = pd.read_csv(csv_file_path)
        
        df['e3_dry_time'] = df.apply(lambda row: calculate_e3(
            row['gpp_ambient'],
            row['gpp_dew_point'],
            row['gpp_wet_bulb'],
            row['outdoor_temp'],
            row['outdoor_humidity']
        ), axis=1)
        
        # Perform image analysis for each room
        df['image_analysis'] = df.apply(lambda row: perform_image_analysis(os.path.join(image_folder, row['Image_File_Name1'])) if pd.notna(row['Image_File_Name1']) else "No image available", axis=1)
        
        site_wide_e3 = df['e3_dry_time'].mean()
        
        logging.info(f"Room-specific E3 calculations and image analysis completed. Site-wide E3: {site_wide_e3}.")
        return df, site_wide_e3
    except Exception as e:
        logging.error(f"Error processing room-specific E3 data and images: {e}")
        raise

def generate_report(report_data, output_path):
    try:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Water Damage Assessment Report", styles['Title']))
        elements.append(Spacer(1, 12))

        if report_data.get("include_text_analysis"):
            elements.append(Paragraph("Text Analysis Results:", styles['Heading2']))
            elements.append(Paragraph(str(report_data["text_analysis_output"]), styles['BodyText']))
            elements.append(Spacer(1, 12))

        if report_data.get("include_room_analysis"):
            elements.append(Paragraph("Room-Specific Analysis:", styles['Heading2']))
            for _, row in report_data["room_data"].iterrows():
                elements.append(Paragraph(f"Room: {row['Room_Name']} (ID: {row['Room_ID']})", styles['Heading3']))
                elements.append(Paragraph(f"E3 Dry Time: {row['e3_dry_time']:.2f}", styles['BodyText']))
                elements.append(Paragraph(f"Image Analysis: {row['image_analysis']}", styles['BodyText']))
                elements.append(Spacer(1, 12))

        if report_data.get("include_e3_calculations"):
            elements.append(Paragraph("E3 Calculations:", styles['Heading2']))
            elements.append(Paragraph(f"Site-wide E3 Value: {report_data['e3_value']:.2f}", styles['BodyText']))
            elements.append(Spacer(1, 12))

        doc.build(elements)

        with open(output_path, 'rb') as f:
            pdf = f.read()

        encoded_pdf = base64.b64encode(pdf).decode('utf-8')
        pdf_data_uri = f"data:application/pdf;base64,{encoded_pdf}"

        logging.info(f"Report generated successfully and saved to {output_path}.")
        return pdf_data_uri
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        raise

def integrate_workflow(csv_path, image_folder):
    try:
        room_data, site_e3 = calculate_room_specific_e3(csv_path, image_folder)
        
        report_data = {
            "include_room_analysis": True,
            "room_data": room_data,
            "include_e3_calculations": True,
            "e3_value": site_e3
        }
        report_uri = generate_report(report_data, report_output_path)
        
        return report_uri
    except Exception as e:
        logging.error(f"Error in workflow integration: {e}")
        return f"An error occurred: {e}"

# Gradio Interface
def gradio_interface(csv_file, image_folder):
    try:
        report_uri = integrate_workflow(csv_file.name, image_folder)
        return report_uri
    except Exception as e:
        logging.error(f"Error in Gradio interface: {e}")
        return f"An error occurred: {e}"

# Gradio UI
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload CSV File"),
        gr.Folder(label="Upload Image Folder")
    ],
    outputs=gr.HTML(label="Generated Report"),
    title="Water Damage Remediation Application",
    description="Upload your CSV and images to generate a detailed damage assessment report."
)

# Run the Gradio app
if __name__ == "__main__":
    interface.launch(share=True)
import os
import pandas as pd
import numpy as np
import logging
from PIL import Image
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage
from io import BytesIO
import base64
from datetime import datetime
import json

# Load environment variable (assuming this has been set up in Hugging Face's UI)
hf_api_key = os.getenv('HUGGINGFACE_API_KEY')

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Replace [CURRENT_DATE] in the report output path
current_date = datetime.now().strftime('%Y-%m-%d')
report_output_path = config['report_output_path'].replace('[CURRENT_DATE]', current_date)

# Ensure output directory exists
output_dir = os.path.dirname(report_output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure logging
logging.basicConfig(filename='app.log', level=config['logging_level'], format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize models once
mistral_tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", use_auth_token=hf_api_key)
mistral_model = AutoModelForSeq2SeqLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", use_auth_token=hf_api_key)

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_auth_token=hf_api_key)
bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", use_auth_token=hf_api_key)

llama_tokenizer = AutoTokenizer.from_pretrained("llama-2-70b-chat", use_auth_token=hf_api_key)
llama_model = AutoModelForSeq2SeqLM.from_pretrained("llama-2-70b-chat", use_auth_token=hf_api_key)

# Data Ingestion and Validation
def validate_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = ['Room_ID', 'Room_Name', 'ambient_temp', 'dew_point_temp', 'wet_bulb_temp', 
                            'gpp_ambient', 'gpp_dew_point', 'gpp_wet_bulb', 'Damage_Description', 'Image_File_Name']

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")

        if df.isnull().values.any():
            raise ValueError("CSV file contains missing values.")

        logging.info(f"CSV {file_path} validated successfully.")
        return df
    except FileNotFoundError as e:
        logging.error(f"CSV file not found: {e}")
        raise
    except ValueError as e:
        logging.error(f"Value error in CSV: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during CSV validation: {e}")
        raise

def validate_images(image_dir, csv_df):
    try:
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        for _, row in csv_df.iterrows():
            image_file = row['Image_File_Name']
            image_path = os.path.join(image_dir, image_file)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file '{image_file}' not found.")
            ext = os.path.splitext(image_file)[1].lower()
            if ext not in valid_extensions:
                raise ValueError(f"Invalid file format '{ext}' for file '{image_file}'")
        
        logging.info(f"All images in {image_dir} validated successfully.")
        return True
    except FileNotFoundError as e:
        logging.error(f"File not found during image validation: {e}")
        raise
    except ValueError as e:
        logging.error(f"Value error during image validation: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during image validation: {e}")
        raise

# Image Processing
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        image = Image.open(image_path)
        image = image.resize(target_size)
        image = image.convert('RGB')
        image_array = np.array(image) / 255.0
        logging.info(f"Image {image_path} preprocessed successfully.")
        return image_array
    except FileNotFoundError as e:
        logging.error(f"Image file not found during preprocessing: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during image preprocessing: {e}")
        raise

# Model Inference
def mistral_text_analysis(text):
    try:
        inputs = mistral_tokenizer(text, return_tensors="pt")
        outputs = mistral_model.generate(**inputs)
        return mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during Mistral text analysis: {e}")
        raise

def bert_ner_analysis(text):
    try:
        inputs = bert_tokenizer(text, return_tensors="pt")
        outputs = bert_model(**inputs)
        return outputs  # Process as needed for NER
    except Exception as e:
        logging.error(f"Error during BERT NER analysis: {e}")
        raise

def llava_image_analysis(image_array):
    try:
        result = "image analysis result using llava-7b"  # Placeholder for real implementation
        logging.info(f"Image analysis result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error during image analysis with LLaVA: {e}")
        raise

def llama_report_generation(text_inputs):
    try:
        inputs = llama_tokenizer(text_inputs, return_tensors="pt")
        outputs = llama_model.generate(**inputs)
        return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during LLaMA report generation: {e}")
        raise

def t5_text_summarization(long_text):
    try:
        t5_tokenizer = AutoTokenizer.from_pretrained("t5-base", use_auth_token=hf_api_key)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", use_auth_token=hf_api_key)
        inputs = t5_tokenizer(long_text, return_tensors="pt")
        outputs = t5_model.generate(**inputs)
        return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during T5 text summarization: {e}")
        raise

# Implement E3 Calculations using the Dewald white paper
def e3_calculations(weather_data):
    try:
        # Implement the logic for E3 calculations using weather data and the white paper
        # This is a placeholder function where actual calculation logic will be added
        e3_value = "calculated E3 value"  # Placeholder for real implementation
        logging.info(f"E3 calculation result: {e3_value}")
        return e3_value
    except Exception as e:
        logging.error(f"Error during E3 calculations: {e}")
        raise

# Report Generation
def generate_report(report_data, output_path):
    try:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Damage Assessment Report", styles['Title']))
        elements.append(Spacer(1, 12))

        if report_data.get("include_text_analysis"):
            elements.append(Paragraph("Text Analysis Results:", styles['Heading2']))
            elements.append(Paragraph(str(report_data["text_analysis_output"]), styles['BodyText']))
            elements.append(Spacer(1, 12))

        if report_data.get("include_image_analysis"):
            elements.append(Paragraph("Image Analysis Results:", styles['Heading2']))
            for image_path in report_data["image_paths"]:
                elements.append(ReportImage(image_path, width=2*inch, height=2*inch))
                elements.append(Spacer(1, 12))

        if report_data.get("include_e3_calculations"):
            elements.append(Paragraph("E3 Calculations:", styles['Heading2']))
            elements.append(Paragraph(f"E3 Value: {report_data['e3_value']}", styles['BodyText']))
            elements.append(Spacer(1, 12))

        buffer = BytesIO()
        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()

        with open(output_path, 'wb') as f:
            f.write(pdf)

        encoded_pdf = base64.b64encode(pdf).decode('utf-8')
        pdf_data_uri = f"data:application/pdf;base64,{encoded_pdf}"

        logging.info(f"Report generated successfully and saved to {output_path}.")
        return pdf_data_uri
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        raise

# Gradio Interface
def gradio_interface(csv_file, image_folder):
    try:
        df = validate_csv(csv_file.name)
        validate_images(image_folder, df)
        
        text_results = []
        for _, row in df.iterrows():
            text = row['Damage_Description']
            text_analysis_result = mistral_text_analysis(text)
            ner_result = bert_ner_analysis(text)
            text_results.append({"analysis": text_analysis_result, "ner": ner_result})
        
        image_results = []
        processed_images = {}
        for _, row in df.iterrows():
            image_file = row['Image_File_Name']
            image_path = os.path.join(image_folder.name, image_file)
            processed_image = preprocess_image(image_path)
            image_results.append(llava_image_analysis(processed_image))
            processed_images[image_file] = image_path

        e3_value = e3_calculations(weather_data="weather_data_placeholder")  # Replace with actual weather data

        report_data = {
            "include_text_analysis": True,
            "text_analysis_output": [llama_report_generation(text["analysis"]) for text in text_results],
            "include_image_analysis": True,
            "image_paths": list(processed_images.keys()),
            "include_e3_calculations": True,
            "e3_value": e3_value
        }
        report_uri = generate_report(report_data, report_output_path)
        
        return report_uri
    
    except Exception as e:
        logging.error(f"Error in Gradio interface: {e}")
        return f"An error occurred: {e}"

# Gradio UI
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload CSV File"),
        gr.Folder(label="Upload Image Folder")
    ],
    outputs=gr.HTML(label="Generated Report"),
    title="Water Damage Remediation Application",
    description="Upload your CSV and images to generate a detailed damage assessment report."
)

# Run the Gradio app
if __name__ == "__main__":
    interface.launch()
