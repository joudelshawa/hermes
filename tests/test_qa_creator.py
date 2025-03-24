import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import os
import Hermes
from Utils.Helpers import *

PATH_DATA = "Data/"
PATH_EXAMPLE = ""
REPORT_NUM = 2 #change to -1 for going through all the examples

if __name__ == "__main__":
    hermes = Hermes.HermesAgenticSystem()

    if (REPORT_NUM != -1):
        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        KGraph = readKGraph(PATH_EXAMPLE)
        Questions, Answers = hermes.getQA(KGraph)
        # Save them
        
        
    else:
        examples = os.listdir(PATH_DATA)
        for ex in examples:
            PATH_EXAMPLE = PATH_DATA + ex + "/"
            KGraph = readKGraph(PATH_EXAMPLE)
            Questions, Answers = hermes.getQA(KGraph)
            # Save them