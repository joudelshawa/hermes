from agents import LLMAgent
from . import CONFIG

class HermesAgenticSystem:
    def __init__(self, config:dict=CONFIG):

        self.CONFIG = config

        # Dictionary mapping LLM model names to model_name required by ollama
        self.LLM_NAME_DICT = {}
        
        self.ReportCreator = LLMAgent.ReportCreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_R", 
            system_prompt = self.CONFIG["Agents"]["Hermes_R"]["SystemPrompt"]
        )

        self.KGraphCreator = LLMAgent.KGCreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_G", 
            system_prompt = self.CONFIG["Agents"]["Hermes_G"]["SystemPrompt"]
        )

        self.QACreator = LLMAgent.QACreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_Q", 
            system_prompt = self.CONFIG["Agents"]["Hermes_Q"]["SystemPrompt"]
        )

        self.AnswerValidator = LLMAgent.AnswerValidator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_A", 
            system_prompt = self.CONFIG["Agents"]["Hermes_A"]["SystemPrompt"]
        )

    def getReport(self, prompt, context = "") -> str:
        return self.ReportCreator.run(prompt=prompt, context=context)

    def getKnowledgeGraph(self, prompt, context = "") -> str:
        return self.KGraphCreator.run(prompt=prompt, context=context)

    def getQA(self, kGraph, context = "") -> tuple[str, str]:
        return self.QACreator.run(prompt=kGraph, context=context)
    
    def getAnswers(self, questions, context = "") -> str:
        # Have to handle context in this case...
        return self.AnswerValidator.run(prompt=questions, context=context)
    
    def validateAnswers(self, ansQA, ansAV):
        # To implement
        pass
    
    def completeRun(self, rawNotes) -> tuple[str, str]:
        report  = self.getReport(rawNotes)
        kGraph = self.getKnowledgeGraph(report)
        questions, ansQA = self.getQA(kGraph)
        ansAV = self.getAnswers(questions, rawNotes)
        # 
        self.validateAnswers(ansQA=ansQA, ansAV=ansAV)


        return kGraph, report

    # All Agents should have their own class if they differ in output types and input types