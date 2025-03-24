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
    
    if (REPORT_NUM != -1):
        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        structured_report = readStructuredReport(PATH_EXAMPLE)
        KGraph = hermes.getKnowledgeGraph(structured_report)
        saveGraphAsText(KGraph, PATH_EXAMPLE)
        saveGraphAsImage(KGraph, PATH_EXAMPLE)
        print(KGraph)

    else:
        examples = os.listdir(PATH_DATA)
        for ex in examples:
            PATH_EXAMPLE = PATH_DATA + ex + "/"
            structured_report = readStructuredReport(PATH_EXAMPLE)
            KGraph = hermes.getKnowledgeGraph(structured_report)
            saveGraphAsText(KGraph, PATH_EXAMPLE)
            saveGraphAsImage(KGraph, PATH_EXAMPLE)

        
