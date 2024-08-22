import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from config import LOGO_PATH

def generate_report(parsed_data, sow, costs, image_paths):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    logo = ImageReader(LOGO_PATH)
    c.drawImage(logo, 50, 700, width=100, height=100)

    c.drawString(100, 750, "Comprehensive Damage Assessment Report")

    y = 700
    for key, value in parsed_data.items():
        if key != 'full_text':
            y -= 20
            c.drawString(100, y, f"{key.replace('_', ' ').title()}: {value}")

    y -= 40
    c.drawString(100, y, "Scope of Work:")
    for line in sow.split('\n'):
        y -= 15
        c.drawString(120, y, line)

    y -= 40
    c.drawString(100, y, "Cost Estimates:")
    for key, value in costs.items():
        y -= 20
        c.drawString(120, y, f"{key.replace('_', ' ').title()}: ${value:.2f}")

    for i, image_path in enumerate(image_paths):
        if i % 2 == 0:
            c.showPage()
            y = 750
        else:
            y = 400
        c.drawString(100, y, f"Image {i+1}")
        c.drawImage(ImageReader(image_path), 100, y-200, width=400, height=200)

    c.save()
    buffer.seek(0)
    return buffer
