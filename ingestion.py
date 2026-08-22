from PyPDF2 import PdfReader
from config import FILE_UPLOAD_DIR

files_path = FILE_UPLOAD_DIR

def extract_pdf_text(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
    """
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


