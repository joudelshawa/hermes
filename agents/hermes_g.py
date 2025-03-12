from core.ollama_client import OllamaClient
from utils.text_processing import remove_think_text

class KnowledgeGraph:
    def __init__(self):
        self.client = OllamaClient()

    def run(self, text):
        prompt = f"Extract key entities and relationships as a knowledge graph from the following text:\n{text}"
        role_content = "You are an assistant of a doctor. You generate knowledge graph from the structured notes of the doctor"
        response = self.client.generate_response(prompt, role_content)

        return remove_think_text(response)


if __name__ == "__main__":
    sample_report_text = f"### Structured Report: John Doe  \
        **Patient Information:**  \
        - **Name:** John Doe  \
        - **Age:** 54 years  \
        - **Gender:** Male  \
        \
        ---\
        \
        **Symptoms:**  \
        - Persistent cough  \
        - Occasional wheezing  \
        - Fatigue  \
        - Mild fever (99.2°F)  \
        - Shortness of breath at night  \
        - Chest discomfort (no severe pain)  \
        \
        ---\
        \
        **Vitals:**  \
        - Blood Pressure (BP): 138/85 mmHg  \
        - Heart Rate (HR): 82 bpm  \
        - Temperature: 99.2°F  \
        - SpO2: 96%  \
        - Respiration Rate: 18/min  \
        \
        ---\
        \
        **Medical History:**  \
        - **Past Medical History:**  \
          - Mild hypertension (diagnosed 5 years ago) on Amlodipine 5mg daily.  \
          - Recent fasting blood glucose slightly elevated (105 mg/dL).  \
        - **Family History:**  \
          - Father: Coronary artery disease (died at 67 from a heart attack).  \
          - Mother: Type 2 diabetes and hypertension.  \
        \
        ---\
        \
        **Additional Information:**  \
        - Smoker (15 pack-years, quit 3 years ago).  \
        - No known allergies.  \
        - Reports occasional headaches and dizziness in the evening.  \
        - Works as an office manager with a high-stress job and poor sleep (5-6 hours per night).  \
        - No alcohol consumption; drinks 2-3 cups of coffee daily.  \
        - No recent travel or exposure to infectious diseases.  \
        \
        ---\
        \
        **Doctor's Impressions and Plan:**  \
        - **Possible Early Signs:** COPD due to smoking history.  \
        - **Recommendations:**  \
          - Further pulmonary function tests.  \
          - Chest X-ray.  \
          - Monitor blood pressure and reassess in 3 months. "

    kg = KnowledgeGraph()
    resp = kg.run(sample_report_text)
    print(resp)
