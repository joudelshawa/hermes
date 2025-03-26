import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import Hermes
import os
from Utils.Helpers import *

REPORT_NUM = 2
PATH_DATA = "Data/"
PATH_EXAMPLE = ""

if __name__ == "__main__":
    hermes = Hermes.HermesAgenticSystem()
    print("\n")
    print("#"*60)
    print(" "*10 + "TESTING: Knowledge Graph Creator Agent")
    print("#"*60)
    
    if (REPORT_NUM != -1):
        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        structured_report = readStructuredReport(PATH_EXAMPLE)
        print(f"Report Number {REPORT_NUM}:\n\tGenerating knowledge graph...")
        KGraph = hermes.getKnowledgeGraph(structured_report)
        saveGraphAsText(KGraph, PATH_EXAMPLE)
        saveGraphAsHTML(KGraph, PATH_EXAMPLE)

    else:
        examples = os.listdir(PATH_DATA)
        for ex in examples[:-1]:
            PATH_EXAMPLE = PATH_DATA + ex + "/"
            structured_report = readStructuredReport(PATH_EXAMPLE)
            print(f"Report Number {ex}:\n\tGenerating knowledge graph...")
            KGraph = hermes.getKnowledgeGraph(structured_report)
            saveGraphAsText(KGraph, PATH_EXAMPLE)
            saveGraphAsHTML(KGraph, PATH_EXAMPLE)

    print("#"*60)
    print(" "*28 + "END")
    print("#"*60)
