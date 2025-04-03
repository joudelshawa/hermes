from Agents import *
from Utils.Helpers import remove_think
# from . import CONFIG, getAgentPrompt

class HermesAgenticSystem:
    def __init__(self, config:dict=CONFIG):

        self.CONFIG = config
        self.MAX_ITERATIONS = config['Hermes-Iterations']

        # Dictionary mapping LLM model names to model_name required by ollama
        self.LLM_NAME_DICT = {}
        
        self.ReportCreator = ReportCreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_R", 
            system_prompt = getAgentPrompt(self.CONFIG["Agents"]["Hermes_R"]["prompt_path"]),
            max_iter=self.CONFIG["Agents"]["Hermes_R"]["max_iter"],
            temperature=self.CONFIG["Agents"]["Hermes_R"]["temperature"],
            top_p=self.CONFIG["Agents"]["Hermes_R"]["top_p"]
        )

        self.KGraphCreator = KGCreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_G", 
            system_prompt = getAgentPrompt(self.CONFIG["Agents"]["Hermes_G"]["prompt_path"]),
            max_iter=self.CONFIG["Agents"]["Hermes_G"]["max_iter"],
            temperature=self.CONFIG["Agents"]["Hermes_G"]["temperature"],
            top_p=self.CONFIG["Agents"]["Hermes_G"]["top_p"]
        )

        self.QACreator = QACreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_Q", 
            system_prompt = getAgentPrompt(self.CONFIG["Agents"]["Hermes_Q"]["prompt_path"]),
            max_iter=self.CONFIG["Agents"]["Hermes_Q"]["max_iter"],
            temperature=self.CONFIG["Agents"]["Hermes_Q"]["temperature"],
            top_p=self.CONFIG["Agents"]["Hermes_Q"]["top_p"]
        )

        self.AnswerValidator = AnswerValidator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_A", 
            system_prompt = getAgentPrompt(self.CONFIG["Agents"]["Hermes_A"]["prompt_path"]),
            max_iter=self.CONFIG["Agents"]["Hermes_A"]["max_iter"],
            temperature=self.CONFIG["Agents"]["Hermes_A"]["temperature"],
            top_p=self.CONFIG["Agents"]["Hermes_A"]["top_p"]
        )

    def getReport(self, prompt, context = "") -> str:
        return self.ReportCreator.run(prompt=prompt, context=context)

    def getKnowledgeGraph(self, prompt, context = "") -> str:
        return self.KGraphCreator.run(prompt=prompt, context=context)

    def getQA(self, prompt, context = "") -> tuple[str, str]:
        return self.QACreator.run(prompt=prompt, context=context)
    
    def getAnswers(self, questions, unstructured_report, context = "") -> str:
        return self.AnswerValidator.run(questions=questions, unstructured_report=unstructured_report, context=context)
    
    def validateAnswers(self, questions, ans_Q, ans_A) -> dict:
        """
        Takes answers from HermesQ and from HermesA and validates them.

        Output:
            dict{
                "is_validated": bool,
                "errors": str (questions with wrong and right answers carefully formatted)
            }
        """
        result = {"is_validated": True, "errors": "ERROR: Make sure the answers of the following questions are correctly included in the structured report - \n"}
        for i, (quest, ans1, ans2) in enumerate(zip(questions, ans_Q, ans_A)):
            ans1 = ans1.strip().lower()
            ans2 = ans2.strip().lower()

            if ans1 != ans2:
                result["is_validated"] = False
                result["errors"] += f"\t{i}. {quest}\n"

        return result
    
    def completeRun(self, unstructuredReport) -> tuple[str, str]:

        max_iter = self.MAX_ITERATIONS
        context = ""
        while(max_iter > 0):
            print("\t============================")
            print(f"\t|    Iterations Left: {max_iter}    |")
            print("\t============================")

            # Step1: Get Report from HermesR
            print(f"\t| Generating Structured Report...")
            structuredReport  = self.getReport(unstructuredReport, context=context)
            
            # Step2: Get Knowledge Graph from HermesG
            print(f"\t| Generating Knowledge Graph...")
            KGraph = self.getKnowledgeGraph(structuredReport)
            
            # Step3: Get Question Answer Pairs from HermesQ
            print(f"\t| Generating Question-Answer Pairs...")
            questions_Q, ans_Q = self.QACreator.getSeparatedQA(self.getQA(KGraph))

            # Step4: Get Answers of questions from HermesA
            print(f"\t| Generating Answers from Unstructured Report Pairs...")
            questions_A, ans_A = self.AnswerValidator.getSeparatedQA(self.getAnswers(questions_Q, unstructuredReport))
            
            # Step5: Validate Answers from
            print(f"\t| Validating Answeres...")  
            result = self.validateAnswers(questions=questions_Q, ans_Q=ans_Q, ans_A=ans_A)

            if(result['is_validated']): 
                return KGraph, structuredReport
            else:
                max_iter -= 1
                context = result["errors"]
                print("\t |---> Wrong Answers!")
                temp = context.replace('\n', '\n\t\t\t')
                print(f"\t\t |---> {temp}")
        
        print("="*50)
        print("\tHermes Failed! (T_T)")
        print("="*50)
        