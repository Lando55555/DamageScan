import os
import pandas as pd
import fitz
import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from config import RATE_SHEET_PATH, WEATHER_API_KEY

def load_rate_sheet(path=RATE_SHEET_PATH):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Rate sheet not found at {path}")
        new_path = input("Please enter the correct path to the rate sheet CSV file: ")
        return pd.read_csv(new_path)

def get_gdrive_service():
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = Flow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def extract_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        image_list = page.get_images()
        for j, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = f"{output_folder}/page{i+1}_image{j+1}.{image_ext}"
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            image_paths.append(image_path)
    doc.close()
    return image_paths

def get_weather_data(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    response = requests.get(url)
    return response.json()
