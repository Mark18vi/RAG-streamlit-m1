from PyPDF2 import PdfReader
from config import FILE_UPLOAD_DIR
from logging_config import get_logger

files_path = FILE_UPLOAD_DIR
logger = get_logger(__name__)

def extract_pdf_text(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
    """
    logger.info("PDF extraction started: %s", pdf_path)
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        text += page_text
    logger.info("PDF extraction completed: pages=%d characters=%d", len(pdf_reader.pages), len(text))
    return text

