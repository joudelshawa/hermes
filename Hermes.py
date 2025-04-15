from Agents import *
from Utils.Helpers import *
from SemanticMatcher import SemanticEmbedder
from Utils.Logger import TheLogger, Level
import json
# from . import CONFIG, getAgentPrompt

class HermesAgenticSystem:
    def __init__(self, config:dict=CONFIG, mainSaveFolder:str = None):

        self.CONFIG = config
        self.llm = self.CONFIG["LLM"]
        self.MAX_ITERATIONS = config['Hermes-Iterations']
        self.SIMILARITY_THRESHOLD = config["Similarity-Threshold"]
        self.semanticEmbedder = SemanticEmbedder() # can change which model to use, configure using config file 
        self.startFlag = True
        self.logger = TheLogger(self.llm, mainSaveFolder)
        self.logger.log(Level.HEADING_2, 0, "\nInitializing Hermes", addTimePrefix=False, addTimeTab=False)
        self.logger.log(Level.SUCCESS, 0, "---"*60 + "\n", addTimePrefix=False, addTimeTab=False)
        # if startMsg != None: self.logger.log(Level.HEADING_0, 0, startMsg, addTimePrefix=False, addTimeTab=False)

        # Dictionary mapping LLM model names to model_name required by ollama
        self.LLM_NAME_DICT = {}
        
        agent = self.CONFIG["Agents"]["Hermes_R"]
        self.ReportCreator = ReportCreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_R", 
            system_prompt = getAgentPrompt(agent["prompt_path"] + "SystemPrompt.txt"),
            max_iter = agent["max_iter"],
            temperature = agent["temperature"],
            top_p = agent["top_p"],
            osl_userPrompt = getAgentPrompt(agent["prompt_path"] + "OSL-UserPrompt.txt"),
            osl_assistantResponse = getAgentPrompt(agent["prompt_path"] + "OSL-AssistantResponse.txt"),
            logger=self.logger
        )
        self.logPromptWithOSL(self.ReportCreator)
        
        agent = self.CONFIG["Agents"]["Hermes_G"]
        self.KGraphCreator = KGCreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_G", 
            system_prompt = getAgentPrompt(agent["prompt_path"] + "SystemPrompt.txt"),
            max_iter = agent["max_iter"],
            temperature = agent["temperature"],
            top_p = agent["top_p"],
            osl_userPrompt = getAgentPrompt(agent["prompt_path"] + "OSL-UserPrompt.txt"),
            osl_assistantResponse = getAgentPrompt(agent["prompt_path"] + "OSL-AssistantResponse.txt"),
            logger=self.logger

        )
        self.logPromptWithOSL(self.KGraphCreator)
        

        agent = self.CONFIG["Agents"]["Hermes_Q"]
        self.QACreator = QACreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_Q", 
            system_prompt = getAgentPrompt(agent["prompt_path"] + "SystemPrompt.txt"),
            max_iter = agent["max_iter"],
            temperature = agent["temperature"],
            top_p = agent["top_p"],
            osl_userPrompt = getAgentPrompt(agent["prompt_path"] + "OSL-UserPrompt.txt"),
            osl_assistantResponse = getAgentPrompt(agent["prompt_path"] + "OSL-AssistantResponse.txt"),
            logger=self.logger
        )
        self.logPromptWithOSL(self.QACreator)

        agent = self.CONFIG["Agents"]["Hermes_A"]
        self.AnswerValidator = AnswerValidator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_A", 
            system_prompt = getAgentPrompt(agent["prompt_path"] + "SystemPrompt.txt"),
            max_iter = agent["max_iter"],
            temperature = agent["temperature"],
            top_p = agent["top_p"],
            osl_userPrompt = getAgentPrompt(agent["prompt_path"] + "OSL-UserPrompt.txt"),
            osl_assistantResponse = getAgentPrompt(agent["prompt_path"] + "OSL-AssistantResponse.txt"),
            logger=self.logger
        )
        self.logPromptWithOSL(self.AnswerValidator)

        self.logger.log(Level.HEADING_0, 0, "---"*60 + "\n", addTimeTab=False)
    
    def _getFormattedStringForOSL(self, osl:list) -> str:
        final_string = "One Shot Learning:"
        if len(osl) == 0:
            final_string += "Not Implemented"
        else:
            userContent = osl[0].get('content', 'error!!').replace('\n','\n\t\t')
            asstContent = osl[1].get('content', 'error!!').replace('\n','\n\t\t')
            final_string += "\n```"
            final_string += f"\n\tUser: \n\t\"\"\"{userContent}\n\t\"\"\""
            final_string += f"\n\tAsst: \n\t\"\"\"{asstContent}\n\t\"\"\""

        return final_string.replace("\n", "\n\t")
    
    def _getFormattedPrompt(self, prompt:str) ->str:
        prompt = prompt.replace("\n", "\n\t")
        return f"Prompt:\n```\n\t{prompt}\n```".replace("\n", "\n\t")
    
    def logPromptWithOSL(self, agent):
        temp_prompt = self._getFormattedPrompt(agent.systemPrompt)
        temp_osl = self._getFormattedStringForOSL(agent.oneShotLearningExample)
        self.logger.log(Level.INFO, 0, agent.name, addTimePrefix=False, addTimeTab=False, onlyLocalWrite=True)
        self.logger.log(Level.INFO, 1, temp_prompt, addTimePrefix=False, addTimeTab=False, onlyLocalWrite=True)
        self.logger.log(Level.INFO, 1, temp_osl, addTimePrefix=False, addTimeTab=False, onlyLocalWrite=True)

    def getReport(self, prompt, context = "") -> str:
        return self.ReportCreator.run(prompt=prompt, context=context)

    def getKnowledgeGraph(self, prompt, context = "") -> str:
        return self.KGraphCreator.run(prompt=prompt, context=context)

    def getQA(self, prompt, context = "") -> tuple[str, str]:
        return self.QACreator.run(prompt=prompt, context=context)
    
    def getAnswers(self, questions, unstructured_report, context = "") -> str:
        return self.AnswerValidator.run(questions=questions, unstructured_report=unstructured_report, context=context)
    
    def createDocumentEmbedding(self):
        pass

    def log(self, level, initialTabs, msg, onlyLocalWrite=False, addTimePrefix=True):
        self.logger.log(level, initialTabs, msg, onlyLocalWrite, addTimePrefix)

    def validateAnswers(self, itr, questions, ans_Q, ans_A) -> dict:
        """
        Takes answers from HermesQ and from HermesA and validates them.

        Output:
            dict{
                "is_validated": bool,
                "errors": str (questions with wrong and right answers carefully formatted)
            }
        """
        result = {"is_validated": True, "errors": "ERROR: Make sure the answers of the following questions are correctly included in the structured report - \n"}
        
        invalid = ""
        if self.startFlag:
            invalid = all = "\n////////////////// START ////////////////////\n"
            # invalid += f"Title: {title}\n\n"
            self.startFlag = False
        invalid += "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        invalid += f"| Invalid Answers for Iteration {itr} |\n"
        invalid  += "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        all = invalid
        for i, (quest, ans1, ans2) in enumerate(zip(questions, ans_Q, ans_A)):
            # ans1 = ans1.strip().lower()
            # ans2 = ans2.strip().lower()

            sim = self.semanticEmbedder.getSemanticSimilarity(sent1=ans1, sent2=ans2)
            all += f"Q{i} [{sim: 0.4}]. {quest}\n"
            all += f"Hermes-Q: {ans1}\n"
            all += f"Hermes-A: {ans2}\n"
            all += "---------------------------------------------\n"
            if sim < self.SIMILARITY_THRESHOLD:
                result["is_validated"] = False
                result["errors"] += f"\t{i}. {quest}\n"
                invalid += f"Q{i} [{sim: 0.4}]. {quest}\n"
                invalid += f"Hermes-Q: {ans1}\n"
                invalid += f"Hermes-A: {ans2}\n"
                invalid += "---------------------------------------------\n"
        
        invalid += "\n/////////////////// END /////////////////////\n\n"
        
        # saveInvalidAnswersAsText(invalid, "invalid/")
        return result, all, invalid
    
    def completeRun(self, unstructuredReport) -> tuple[str, str]:

        itr = 0
        context = ""
        while(itr < self.MAX_ITERATIONS):
            self.logger.log(Level.HEADING_1, 1,"==========================")
            self.logger.log(Level.HEADING_1, 1,f"|    Iteration: [{itr+1}/{self.MAX_ITERATIONS}]    |", addTimePrefix=True)
            self.logger.log(Level.HEADING_1, 1,"==========================")

            # Step1: Get Report from HermesR
            self.logger.log(Level.HEADING_2, 1,f"| Hermes-R", addTimePrefix=True)
            self.logger.log(Level.INFO, 1,f"| Generating Structured Report...")
            structuredReport  = self.getReport(unstructuredReport, context=context)
            self.logger.log(Level.INFO, 1,"|--------------------------------------------")
            self.logger.log(Level.INFO, 1,"|--------------------------------------------")
            
            # Step2: Get Knowledge Graph from HermesG
            self.logger.log(Level.HEADING_2, 1,f"| Hermes-G", addTimePrefix=True)
            self.logger.log(Level.INFO, 1, f"| Generating Knowledge Graph...")
            KGraph = self.getKnowledgeGraph(structuredReport)
            self.logger.log(Level.INFO, 1,"|--------------------------------------------")
            self.logger.log(Level.INFO, 1,"|--------------------------------------------")
            
            # Step3: Get Question Answer Pairs from HermesQ
            self.logger.log(Level.HEADING_2, 1,f"| Hermes-Q", addTimePrefix=True)
            self.logger.log(Level.INFO, 1, f"| Generating Question-Answer Pairs...")
            qa_pairs = self.getQA(KGraph)
            questions_Q, ans_Q = self.QACreator.getSeparatedQA(qa_pairs)
            self.logger.log(Level.INFO, 1,"|--------------------------------------------")
            self.logger.log(Level.INFO, 1,"|--------------------------------------------") 

            # Step4: Get Answers of questions from HermesA
            self.logger.log(Level.HEADING_2, 1,f"| Hermes-A", addTimePrefix=True)
            self.logger.log(Level.INFO, 1, f"| Generating Answers from Unstructured Report...")
            av_pairs = self.getAnswers(questions_Q, unstructuredReport)
            questions_A, ans_A = self.AnswerValidator.getSeparatedQA(av_pairs)
            self.logger.log(Level.INFO, 1,"|--------------------------------------------")
            self.logger.log(Level.INFO, 1,"|--------------------------------------------")
            
            # Step5: Validate Answers from
            self.logger.log(Level.WARNING, 1 ,f"| Validating Answers...", addTimePrefix=True)
            self.semanticEmbedder.load() 
            result, allAnswers, invalidAnswers = self.validateAnswers(itr=itr, questions=questions_Q, ans_Q=ans_Q, ans_A=ans_A)
            self.semanticEmbedder.unload()
            

            if(result['is_validated']):
                self.logger.log(Level.INFO, 0, "\n\nFinal Structured Report", onlyLocalWrite=True, addTimeTab=False)
                self.logger.log(Level.INFO, 0, structuredReport, onlyLocalWrite=True, addTimeTab=False)
                self.logger.log(Level.INFO, 0, "\n\nFinal Graph", onlyLocalWrite=True, addTimeTab=False)
                self.logger.log(Level.INFO, 0, KGraph, onlyLocalWrite=True, addTimeTab=False)
                self.logger.log(Level.INFO, 0, "\n\nQA vs AV: Invalid Answers", onlyLocalWrite=True, addTimeTab=False)
                self.logger.log(Level.INFO, 0, invalidAnswers, onlyLocalWrite=True, addTimeTab=False)
                self.logger.log(Level.INFO, 0, "\n\nQA vs AV: All Answers", onlyLocalWrite=True, addTimeTab=False)
                self.logger.log(Level.INFO, 0, allAnswers, onlyLocalWrite=True, addTimeTab=False)

                self.logger.log(Level.SUCCESS, 1 ,"| |---> SUCCESS!!", addTimePrefix=True)
                self.logger.log(Level.HEADING_1, 1 ,"==============================================\n")
                return KGraph, structuredReport
            else:
                itr += 1
                context = result["errors"]
                self.logger.log(Level.ERROR, 1,"|\t|---> Wrong Answers!")
                temp = context.replace('\n', '\n\t|\t\t')
                self.logger.log(Level.ERROR, 1,f"|\t|---> {temp}")
            
            self.logger.log(Level.INFO, 1,"|--------------------------------------------")
            self.logger.log(Level.INFO, 1,"\t|--------------------------------------------")
            
        self.logger.log(Level.CRITICAL, 0,"="*50)
        self.logger.log(Level.CRITICAL, 0,"\tHermes Failed! (T_T)")
        self.logger.log(Level.CRITICAL, 0,"="*50)
        