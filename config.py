import os

# Paths
COMPLETED_REPORTS_PATH = "/content/drive/MyDrive/Colab/set_up_paths/generated_reports/"
FINE_TUNING_PATH = "/content/drive/MyDrive/Colab/fine_tuning_historical/"
LOGO_PATH = "/content/drive/MyDrive/Colab/set_up_paths/ihc_logo.pdf"
RATE_SHEET_PATH = "/content/drive/MyDrive/Colab/set_up_paths/rate_sheet_2024.csv"

# API Keys
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "your_default_key_here")

# Constants
DAMAGE_TYPES = [
    "Warping in walls, floors, or ceilings",
    "Peeling paint or wallpaper",
    "Staining or discoloration",
    "Efflorescence on concrete or brick",
    "Cracks in the foundation",
    "Corrosion of metal fixtures",
    "Separation of building materials",
    "Sagging or drooping ceilings",
    "Rotting wood",
    "Presence of moisture or dampness",
    "Mold growth"
]
