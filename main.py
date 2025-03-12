from core.pipeline import MultiAgentPipeline

if __name__ == "__main__":
    pdf_path = "data/unstructured_medical_notes.pdf"
    pipeline = MultiAgentPipeline()
    result = pipeline.run(pdf_path)
    print(result)
