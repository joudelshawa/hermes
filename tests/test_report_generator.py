import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import Hermes
import json
from utils.helpers import *

PATH_DATA = "data/"
PATH_EXAMPLE = ""
REPORT_NUM = 2 #change to -1 for going through all the examples

if __name__ == "__main__":
    hermes = Hermes.HermesAgenticSystem()

    if (REPORT_NUM != -1):
        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        unstructured_report = readUnstructuredReport(PATH_EXAMPLE)
        report = hermes.getReport(unstructured_report)
        saveReportAsText(remove_think(report))
        
    else:
        examples = os.listdir(PATH_DATA)
        for ex in examples:
            PATH_EXAMPLE = PATH_DATA + ex + "/"
            unstructured_report = readUnstructuredReport(PATH_EXAMPLE)
            report = hermes.getReport(unstructured_report)
            saveReportAsText(remove_think(report))
           

    
