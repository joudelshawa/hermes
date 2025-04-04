import Hermes
import os
from Utils.Helpers import *

REPORT_NUM = 2
PATH_DATA = "Data/"
PATH_EXAMPLE = ""

if not os.path.exists("Temp/"):
    os.mkdir("Temp/")

if __name__ == "__main__":
    print("\n")
    print("#"*60)
    print(" "*20 + "TESTING: MAIN")
    print("#"*60)
    hermes = Hermes.HermesAgenticSystem()
    if (REPORT_NUM != -1):
        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        unstructuredReport = readUnstructuredReport(PATH_EXAMPLE)
        print(f"Report Number {REPORT_NUM}:")
        
        kGraph, structuredReport = hermes.completeRun(unstructuredReport=unstructuredReport)
        saveGraphAsHTML(graph_data=kGraph, folder_path=PATH_EXAMPLE)
        saveReportAsText(text_data=structuredReport, folder_path=PATH_EXAMPLE)

    print("#"*60)
    print(" "*28 + "END")
    print("#"*60)
        
