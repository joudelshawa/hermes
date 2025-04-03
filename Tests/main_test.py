import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import Hermes
import os
from Utils.Helpers import *

REPORT_NUM = 3
PATH_DATA = "Data/"
PATH_EXAMPLE = ""

if __name__ == "__main__":
    hermes = Hermes.HermesAgenticSystem()
    print("\n")
    print("#"*60)
    print(" "*10 + "TESTING: INCREMENTAL")
    print("#"*60)
    
    if (REPORT_NUM != -1):
        PATH_EXAMPLE = PATH_DATA + str(REPORT_NUM) + "/"
        print(f"Report Number {REPORT_NUM}:\n\tStarting...")
        unstructured_report = readUnstructuredReport(PATH_EXAMPLE)
        
        print(f"\tGenerating Structured Report...")
        structured_report = hermes.getReport(unstructured_report)
        saveReportAsText(structured_report, PATH_EXAMPLE)

        print(f"\tGenerating Knowledge Graph...")
        KGraph = hermes.getKnowledgeGraph(structured_report)
        saveGraphAsText(KGraph, PATH_EXAMPLE)
        saveGraphAsHTML(KGraph, PATH_EXAMPLE)

        print(f"\tGenerating Question-Answer Pairs...")
        qa_pairs = hermes.getQA(KGraph)
        saveQAPairsAsText(qa_pairs, PATH_EXAMPLE)
        questions, answers = hermes.QACreator.getSeparatedQA(qa_pairs)

        print(f"\tGenerating Answers from Unstructured Report Pairs...")
        av_pairs = hermes.getAnswers(questions, unstructured_report)
        questions, answers = hermes.AnswerValidator.getSeparatedQA(av_pairs)
        saveAVPairsAsText(av_pairs, PATH_EXAMPLE)

    else:
        examples = os.listdir(PATH_DATA)
        for ex in examples[:-1]:
            PATH_EXAMPLE = PATH_DATA + ex + "/"
            print(f"Report Number {ex}:\n\tStarting...")
            unstructured_report = readUnstructuredReport(PATH_EXAMPLE)
            
            print(f"\tGenerating Structured Report...")
            structured_report = hermes.getReport(unstructured_report)
            saveReportAsText(structured_report, PATH_EXAMPLE)

            print(f"\tGenerating Knowledge Graph...")
            KGraph = hermes.getKnowledgeGraph(structured_report)
            saveGraphAsText(KGraph, PATH_EXAMPLE)
            saveGraphAsHTML(KGraph, PATH_EXAMPLE)

    print("#"*60)
    print(" "*28 + "END")
    print("#"*60)
