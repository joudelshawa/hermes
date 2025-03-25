import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import Hermes
import json
from Utils.Helpers import *

PATH_DATA = "data/"
PATH_EXAMPLE = ""
REPORT_NUM = 2 #change to -1 for going through all the examples

if __name__ == "__main__":
    hermes = Hermes.HermesAgenticSystem()
    print("TESTING: Report Creator Agent")

    if (REPORT_NUM != -1):
        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        unstructured_report = readUnstructuredReport(PATH_EXAMPLE)
        print(f"Report Number {REPORT_NUM}:\n\tGenerating Structured Report...")
        report = hermes.getReport(unstructured_report)
        saveReportAsText(report)
        
    else:
        examples = os.listdir(PATH_DATA)
        for ex in examples:
            PATH_EXAMPLE = PATH_DATA + ex + "/"
            unstructured_report = readUnstructuredReport(PATH_EXAMPLE)
            print(f"Report Number {ex}:\n\tGenerating Structured Report...")
            report = hermes.getReport(unstructured_report)
            saveReportAsText(report) 