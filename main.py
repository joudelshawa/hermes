import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import Hermes
from Utils.Helpers import *
from Utils.Logger import TheLogger, Level
from datetime import datetime
import time

REPORT_NUM = 2
PATH_DATA = "Data/"
PATH_EXAMPLE = ""

if __name__ == "__main__":
    start_time = time.time()
    startedAt = datetime.now().strftime("%d-%b-%y")
    
    startMsg = f"[{startedAt}]\n" + \
    "#"*60 + "\n" + \
    " "*20 + "TESTING: MAIN" + "\n" + \
    "#"*60 + "\n"\
    f"Report Number: {REPORT_NUM}\n---"

    if (REPORT_NUM > 0):

        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        unstructuredReport = readUnstructuredReport(PATH_EXAMPLE)
        hermes = Hermes.HermesAgenticSystem(mainSaveFolder=PATH_EXAMPLE)
        hermes.logger.log(Level.HEADING_2, 0, f"[{startedAt}]", addTimeTab=False)
        hermes.logger.log(Level.HEADING_0, 0, "#"*60, addTimeTab=False)
        hermes.logger.log(Level.INFO, 0, " "*20 + "TESTING: MAIN", addTimeTab=False)
        hermes.logger.log(Level.HEADING_0, 0, "#"*60, addTimeTab=False)
        hermes.logger.log(Level.CRITICAL, 0, f"Report Number: {REPORT_NUM}", addTimeTab=False)
        hermes.logger.log(Level.SUCCESS,  0, f"------------------\n", addTimeTab=False)
        
        # Instead of titel have to pass the whole path to save temp files and the report number so that the title can also be created
        kGraph, structuredReport = hermes.completeRun(unstructuredReport=unstructuredReport)
        saveGraphAsHTML(graph_data=kGraph, folder_path=hermes.logger.mainSaveFolder)
        saveGraphAsText(graph_data=kGraph, folder_path=hermes.logger.mainSaveFolder)
        saveReportAsText(text_data=structuredReport, folder_path=hermes.logger.mainSaveFolder)

    hermes.logger.log(Level.HEADING_0, 0, "#"*60)
    hermes.logger.log(Level.HEADING_0, 0, " "*28 + "END")
    hermes.logger.log(Level.HEADING_0, 0, "#"*60)
    end_time = time.time()
    hermes.logger.log(Level.HEADING_0, 0, f"\nTotal Time Taken: {getFormattedElapsedTime(start_time, end_time)}")
        
