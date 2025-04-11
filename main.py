import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import Hermes
from Utils.Helpers import *
import time
import shutil

REPORT_NUM = 6
PATH_DATA = "Data/"
PATH_EXAMPLE = ""
OUTPUT_DIR = "Temp/"

if os.path.exists(OUTPUT_DIR): # clear the folder up before we start running 
    shutil.rmtree(OUTPUT_DIR)
os.mkdir(OUTPUT_DIR)

# document report number in readme so we know what the results correspond to
readme_path = os.path.join(OUTPUT_DIR, "README.txt")
with open(readme_path, "w") as f:
    f.write(f"REPORT_NUM: {REPORT_NUM}\n")

if __name__ == "__main__":
    start_time = time.time()

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
        saveGraphAsText(graph_data=kGraph, folder_path=PATH_EXAMPLE)
        saveReportAsText(text_data=structuredReport, folder_path=PATH_EXAMPLE)

    print("#"*60)
    print(" "*28 + "END")
    print("#"*60)
    end_time = time.time()
    print(f"total time taken to run Hermes: {end_time - start_time:.2f} seconds")
        
