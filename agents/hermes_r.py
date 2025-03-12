import fitz  # PyMuPDF
from core.ollama_client import OllamaClient
from utils.text_processing import remove_think_text

class ReportGenerator:
    def __init__(self):
        self.client = OllamaClient()

    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text() for page in doc)

    def run(self, pdf_path):
        text = self.extract_text(pdf_path)
        role_content = "You are a medical assistant specializing in clinical documentation. Your task is to generate a well-structured medical report from the doctor's unstructured notes. Maintain medical accuracy and professionalism while improving readability."
        prompt = f"Extract a structured report from the following text:\n{text}"
        response = self.client.generate_response(prompt, role_content)
        return remove_think_text(response)


if __name__ == "__main__":
    agent = ReportGenerator()
    report = agent.run("../data/unstructured_medical_notes.pdf")
    print(report)
