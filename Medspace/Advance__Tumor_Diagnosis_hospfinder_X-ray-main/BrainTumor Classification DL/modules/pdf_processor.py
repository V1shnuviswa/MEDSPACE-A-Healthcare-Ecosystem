import PyPDF2

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
