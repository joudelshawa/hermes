from agents import Hermes
import os
from utils.text_processing import remove_think_text
import json

if __name__ == "__main__":
    with open("agents/config.json", "r") as file:
        config = json.load(file)

    hermes = Hermes.HermesAgenticSystem(config=config)

    unstructured_reports_path = "data/unstructured_reports"
    structured_reports_path = "data/structured_reports"

    if os.path.isdir(unstructured_reports_path):
        for filename in os.listdir(unstructured_reports_path):
            print(filename)
            if filename.endswith('.txt'):
                file_path = os.path.join(unstructured_reports_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    rawNotes = file.read()
                report = hermes.getReport(rawNotes, context=config['Agents']['Hermes_R']['Context'])
                structured_file_path = os.path.join(structured_reports_path, filename)
                with open(structured_file_path, "w") as file:
                    file.write(remove_think_text(report))
                print(f"Structured report saved at {structured_file_path}")
    else:
        print(f"No unstructured reports found in {unstructured_reports_path}")

    
