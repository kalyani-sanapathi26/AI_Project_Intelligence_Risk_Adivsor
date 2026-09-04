from pathlib import Path
import pandas as pd
from pypdf import PdfReader
from docx import Document


def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def load_docx(file):
    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"

    return text


def load_csv(file):
    df = pd.read_csv(file)

    return df.to_string(index=False)


def load_txt(file):
    return file.read().decode("utf-8")


def extract_text(file):

    file_extension = Path(file.name).suffix.lower()

    if file_extension == ".pdf":
        return load_pdf(file)

    elif file_extension == ".docx":
        return load_docx(file)

    elif file_extension == ".csv":
        return load_csv(file)

    elif file_extension == ".txt":
        return load_txt(file)

    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}"
        )