# Water Damage Remediation Application

## Overview

This Gradio-based application is designed to assist in water damage assessment and remediation planning. It processes input data from a CSV file and associated images to generate a comprehensive report on water damage in different rooms, including E3 (Evaporation, Evacuation, and Exchange) calculations and AI-powered image analysis.

## Features

- E3 (Evaporation, Evacuation, and Exchange) calculation for each room
- AI-powered image analysis of water damage using Vision Transformer (ViT) and GPT-4
- Generation of a detailed PDF report including room-specific analyses and site-wide metrics
- User-friendly Gradio interface for easy data input and report retrieval

## Setup

1. Clone this repository or create a new Hugging Face Space with these files.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - `HUGGINGFACE_API_KEY`: Your Hugging Face API key
   - `OPENAI_API_KEY`: Your OpenAI API key

   You can set these in the Hugging Face Space settings under the "Secrets" tab.

4. Ensure you have the necessary permissions and quota to use the specified models (Llama-2, ViT, GPT-4).

## Usage

1. Prepare your input data:
   - A CSV file containing room-specific data (see `Damage_Report_Sample.csv` for the required format)
   - A folder containing images of the water damage, with filenames matching those specified in the CSV

2. Open the Gradio interface by running the application or visiting your Hugging Face Space.

3. Upload your CSV file and select the folder containing your images.

4. Click "Submit" to process the data and generate the report.

5. Once processing is complete, you'll receive a link to download the generated PDF report.

## Input Data Format

Your CSV file should include the following columns:

- County, State
- Room_ID, Room_Name, Size of Room
- ambient_temp, dew_point_temp, wet_bulb_temp
- gpp_ambient, gpp_dew_point, gpp_wet_bulb
- outdoor_temp, outdoor_humidity
- Damage Class, Damage_Description
- Natural Airflow, Affected Materials
- Image_File_Name1, Image_File_Name2, Image_File_Name3, Image_File_Name4
- Inspection_Date, Inspector_Name
- Initial_Moisture_Content, Desired_Final_Moisture_Content
- Moisture Measurement Method
- Comments

Ensure that the image filenames in your CSV match the actual filenames in your image folder.

## Technical Details

- The application uses a Llama-2 model for text processing and a ViT (Vision Transformer) model for initial image classification.
- GPT-4 is used to generate detailed analyses based on the image classifications.
- E3 calculations are performed using a custom formula based on the input environmental data.
- The final report is generated as a PDF using the ReportLab library.

## Limitations and Future Improvements

- The current image analysis model is a general-purpose classifier. Future versions could use a model specifically trained on water damage images.
- The application currently processes one image per room. Future versions could incorporate analysis from multiple images per room.
- Error handling and input validation could be improved for more robust operation.

## Contributing

Contributions to improve the application are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for hosting and model access
- OpenAI for GPT-4 API access
- The creators and maintainers of all the open-source libraries used in this project

