from agents import *
from utils.helpers import remove_think
# from . import CONFIG, getAgentPrompt

class HermesAgenticSystem:
    def __init__(self, config:dict=CONFIG):

        self.CONFIG = config

        # Dictionary mapping LLM model names to model_name required by ollama
        self.LLM_NAME_DICT = {}
        
        self.ReportCreator = ReportCreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_R", 
            system_prompt = getAgentPrompt(self.CONFIG["Agents"]["Hermes_R"])
        )

        self.KGraphCreator = KGCreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_G", 
            system_prompt = getAgentPrompt(self.CONFIG["Agents"]["Hermes_G"])
        )

        self.QACreator = QACreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_Q", 
            system_prompt = getAgentPrompt(self.CONFIG["Agents"]["Hermes_Q"])
        )

        self.AnswerValidator = AnswerValidator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_A", 
            system_prompt = getAgentPrompt(self.CONFIG["Agents"]["Hermes_A"])
        )

    def getReport(self, prompt, context = "") -> str:
        return self.ReportCreator.run(prompt=prompt, context=context)

    def getKnowledgeGraph(self, prompt, context = "") -> str:
        return self.KGraphCreator.run(prompt=prompt, context=context)

    def getQA(self, prompt, context = "") -> tuple[str, str]:
        return self.QACreator.run(prompt=prompt, context=context)
    
    def getAnswers(self, questions, context = "") -> str:
        # Have to handle context in this case...
        return self.AnswerValidator.run(prompt=questions, context=context)
    
    def validateAnswers(self, ansQA, ansAV):
        # To implement
        pass
    
    def completeRun(self, rawNotes) -> tuple[str, str]:
        report  = remove_think(self.getReport(rawNotes))
        kGraph = remove_think(self.getKnowledgeGraph(report))
        questions, ansQA = self.getQA(kGraph)
        ansAV = remove_think(self.getAnswers(questions, rawNotes))
        # 
        self.validateAnswers(ansQA=ansQA, ansAV=ansAV)


        return kGraph, report
    
    

    # All Agents should have their own class if they differ in output types and input types