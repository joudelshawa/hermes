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
    mismatch = False

    if (REPORT_NUM != -1):
        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        print(f"Report Number {REPORT_NUM}:\n\tChecking QA and AV pairs...")
        qa_pairs = readPairs("QA", PATH_EXAMPLE)
        av_pairs = readPairs("AV", PATH_EXAMPLE)
        
        for i, (entry1, entry2) in enumerate(zip(qa_pairs, av_pairs), start=1):
                entry1 = {k.lower(): v for k, v in entry1.items()}
                entry2 = {k.lower(): v for k, v in entry2.items()}
                for key in ('question', 'answer'):
                    value1 = entry1.get(key)
                    value2 = entry2.get(key)
                    if value1 != value2:
                        print(f"\n\tMismatch at item {i}, key '{key}':")
                        if key == 'answer':
                            # also show the question for clarity
                            question = entry1.get('question') or entry2.get('question')
                            print(f"\t  Question: {question}")
                        print(f"\t  qa_file: {value1}")
                        print(f"\t  av_file: {value2}")
                        mismatch = True

        print(f"\n\tDone comparing QA and AV outputs for report {REPORT_NUM}. {'Mismatch found.' if mismatch else 'No mismatch found.'}\n")

    else:
        examples = os.listdir(PATH_DATA)
        for ex in examples:
            PATH_EXAMPLE = PATH_DATA + ex + "/"