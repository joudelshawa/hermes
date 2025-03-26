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
    print("TESTING: Answer Validator Agent")

    if (REPORT_NUM != -1):
        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        print(f"Report Number {REPORT_NUM}:\n\tGenerating answer validator question-answer pairs...")
        questions = readQuestions(PATH_EXAMPLE)
        unstruct_rep = readUnstructuredReport(PATH_EXAMPLE)
        av_pairs = hermes.getAnswers(questions, unstruct_rep)
        # print(av_pairs)
        saveAVPairsAsText(av_pairs, PATH_EXAMPLE)
        
        
    else:
        examples = os.listdir(PATH_DATA)
        for ex in examples:
            PATH_EXAMPLE = PATH_DATA + ex + "/"
            print(f"Report Number {REPORT_NUM}:\n\tGenerating answer validator question-answer pairs...")
            questions = readQuestions(PATH_EXAMPLE)
            unstruct_rep = readUnstructuredReport(PATH_EXAMPLE)
            av_pairs = hermes.getAnswers(questions, unstruct_rep)
            # print(av_pairs)
            saveAVPairsAsText(av_pairs, PATH_EXAMPLE)