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
    
    def validateAnswers(self, ansQA, ansAV) -> dict:
        """
        Takes answers from HermesQ and from HermesA and validates them.

        Output:
            dict{
                "is_validated": bool,
                "errors": str (questions with wrong and right answers carefully formatted)
            }
        """
        pass
    
    def completeRun(self, rawNotes) -> tuple[str, str]:

        max_iter = self.MAX_ITERATIONS
        while(max_iter > 0):
            
            # Step1: Get Report from HermesR
            report  = self.getReport(rawNotes)
            
            # Step2: Get Knowledge Graph from HermesG
            KGraph = self.getKnowledgeGraph(report)
            
            # Step3: Get Question Answer Pairs from HermesQ
            questions, ansQA = self.QACreator.getSeparatedQA(self.getQA(KGraph))

            # Step4: Get Answers of questions from HermesA
            ansAV = self.getAnswers(questions, rawNotes)
            
            # Step5: Validate Answers from  
            result = self.validateAnswers(ansQA=ansQA, ansAV=ansAV)

            if(result['is_validated']): 
                return KGraph, report
            else:
                max_iter -= 1
        
        print("="*50)
        print("\tHermes Failed! (T_T)")
        print("="*50)
        