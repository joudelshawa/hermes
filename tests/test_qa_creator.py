import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import os
import Hermes
from utils.helpers import *

PATH_DATA = "data/"
PATH_EXAMPLE = ""
REPORT_NUM = 2 #change to -1 for going through all the examples

if __name__ == "__main__":
    hermes = Hermes.HermesAgenticSystem()
    print("TESTING: Question-Answer Creator Agent")

    if (REPORT_NUM != -1):
        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        print(f"Report Number {REPORT_NUM}:\n\tGenerating question-answer pairs...")
        KGraph = readKGraph(PATH_EXAMPLE)
        qa_pairs = hermes.getQA(KGraph)
        # print(qa_pairs)
        saveQAPairsAsText(qa_pairs, PATH_EXAMPLE)
        
        
    else:
        examples = os.listdir(PATH_DATA)
        for ex in examples:
            PATH_EXAMPLE = PATH_DATA + ex + "/"
            print(f"Report Number {REPORT_NUM}:\n\tGenerating question-answers...")
            KGraph = readKGraph(PATH_EXAMPLE)
            qa_pairs = hermes.getQA(KGraph)
            saveQAPairsAsText(qa_pairs, PATH_EXAMPLE)