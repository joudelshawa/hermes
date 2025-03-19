from agents import Hermes

if __name__ == "__main__":
    hermes = Hermes.HermesAgenticSystem()
    # Load raw notes
    with open("data/unstructured_reports/report_1.txt", "r", encoding="utf-8") as file:
        rawNotes = file.read()
    kGraph, report = hermes.completeRun(rawNotes=rawNotes)
    print(f"Report: {report}")
    print("="*50)
    print(f"kGraph: {kGraph}")
    
