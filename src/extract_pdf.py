import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

if __name__ == "__main__":
    pdf_path = "694e0095a7f16_FedEx_SMART_Hackathon_DCA_Problem_statement_20.12.2025.pdf"
    text = extract_text_from_pdf(pdf_path)
    print(text)