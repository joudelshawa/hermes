# use this file to check similarities for outputs of q&a pairs if you already have them and dont want to rerun the whole pipeline!
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import Hermes
from Utils.Helpers import *

def separate(pairs):
    questions = []
    answers = []
    for pair in pairs:
        questions.append(pair["Question"])
        answers.append(pair["Answer"])
    return questions, answers
    
if __name__ == "__main__":
    questions_A, ans_A = separate(readPairs(pair_type="AV", folder_path="Temp/"))
    questions_Q, ans_Q = separate(readPairs(pair_type="QA", folder_path="Temp/"))

    hermes = Hermes.HermesAgenticSystem()
    print(f"\t| Validating Answers...")
    semanticEmbedder = hermes.semanticEmbedder.load() 
    result = hermes.validateAnswers(itr=1, questions=questions_Q, ans_Q=ans_Q, ans_A=ans_A)
    semanticEmbedder = hermes.semanticEmbedder.unload() 

    if(result['is_validated']):
        print("\t| SUCCESS!!")
    else:
        print("FAILED")

