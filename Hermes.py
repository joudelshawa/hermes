from Agents import *
from Utils.Helpers import *
from SemanticMatcher import SemanticEmbedder
# from . import CONFIG, getAgentPrompt

class HermesAgenticSystem:
    def __init__(self, config:dict=CONFIG):

        self.CONFIG = config
        self.MAX_ITERATIONS = config['Hermes-Iterations']
        self.SIMILARITY_THRESHOLD = config["Similarity-Threshold"]
        self.semanticEmbedder = SemanticEmbedder() # can change which model to use, configure using config file 

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
        )

        agent = self.CONFIG["Agents"]["Hermes_G"]
        self.KGraphCreator = KGCreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_G", 
            system_prompt = getAgentPrompt(agent["prompt_path"] + "SystemPrompt.txt"),
            max_iter = agent["max_iter"],
            temperature = agent["temperature"],
            top_p = agent["top_p"],
            osl_userPrompt = getAgentPrompt(agent["prompt_path"] + "OSL-UserPrompt.txt"),
            osl_assistantResponse = getAgentPrompt(agent["prompt_path"] + "OSL-AssistantResponse.txt")

        )

        agent = self.CONFIG["Agents"]["Hermes_Q"]
        self.QACreator = QACreator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_Q", 
            system_prompt = getAgentPrompt(agent["prompt_path"] + "SystemPrompt.txt"),
            max_iter = agent["max_iter"],
            temperature = agent["temperature"],
            top_p = agent["top_p"],
            osl_userPrompt = getAgentPrompt(agent["prompt_path"] + "OSL-UserPrompt.txt"),
            osl_assistantResponse = getAgentPrompt(agent["prompt_path"] + "OSL-AssistantResponse.txt")

        )

        agent = self.CONFIG["Agents"]["Hermes_A"]
        self.AnswerValidator = AnswerValidator(
            base_llm = self.CONFIG["LLM"],
            name = "Hermes_A", 
            system_prompt = getAgentPrompt(agent["prompt_path"] + "SystemPrompt.txt"),
            max_iter = agent["max_iter"],
            temperature = agent["temperature"],
            top_p = agent["top_p"],
            osl_userPrompt = getAgentPrompt(agent["prompt_path"] + "OSL-UserPrompt.txt"),
            osl_assistantResponse = getAgentPrompt(agent["prompt_path"] + "OSL-AssistantResponse.txt")
        )

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
            # ans1 = ans1.strip().lower()
            # ans2 = ans2.strip().lower()

            sim = self.semanticEmbedder.getSemanticSimilarity(sent1=ans1, sent2=ans2)
            if sim < self.SIMILARITY_THRESHOLD:
                result["is_validated"] = False
                result["errors"] += f"\t{i}. {quest}\n"

        return result
    
    def completeRun(self, unstructuredReport) -> tuple[str, str]:

        max_iter = self.MAX_ITERATIONS
        context = ""
        while(max_iter > 0):
            print("\t==========================")
            print(f"\t|    Iteration: [{self.MAX_ITERATIONS-max_iter+1}/{self.MAX_ITERATIONS}]    |")
            print("\t==========================")

            # Step1: Get Report from HermesR
            print(f"\t| Generating Structured Report...")
            structuredReport  = self.getReport(unstructuredReport, context=context)
            saveReportAsText(structuredReport, "Temp/")
            print("\t|--------------------------------------------")
            print("\t|--------------------------------------------")
            
            # Step2: Get Knowledge Graph from HermesG
            print(f"\t| Generating Knowledge Graph...")
            KGraph = self.getKnowledgeGraph(structuredReport)
            saveGraphAsHTML(KGraph, "Temp/")
            print("\t|--------------------------------------------")
            print("\t|--------------------------------------------")
            
            # Step3: Get Question Answer Pairs from HermesQ
            print(f"\t| Generating Question-Answer Pairs...")
            qa_pairs = self.getQA(KGraph)
            questions_Q, ans_Q = self.QACreator.getSeparatedQA(qa_pairs)
            saveQAPairsAsText(qa_pairs, "Temp/")
            print("\t|--------------------------------------------")
            print("\t|--------------------------------------------")
            

            # Step4: Get Answers of questions from HermesA
            print(f"\t| Generating Answers from Unstructured Report Pairs...")
            av_pairs = self.getAnswers(questions_Q, unstructuredReport)
            questions_A, ans_A = self.AnswerValidator.getSeparatedQA(av_pairs)
            saveAVPairsAsText(av_pairs, "Temp/")
            print("\t|--------------------------------------------")
            print("\t|--------------------------------------------")
            
            # Step5: Validate Answers from
            print(f"\t| Validating Answeres...")
            self.semanticEmbedder.load() 
            result = self.validateAnswers(questions=questions_Q, ans_Q=ans_Q, ans_A=ans_A)
            self.semanticEmbedder.unload()
            print("\t|--------------------------------------------")
            print("\t|--------------------------------------------")
            

            if(result['is_validated']): 
                return KGraph, structuredReport
            else:
                max_iter -= 1
                context = result["errors"]
                print("\t|---> Wrong Answers!")
                temp = context.replace('\n', '\n\t\t')
                print(f"\t\t |---> {temp}")
        
        print("="*50)
        print("\tHermes Failed! (T_T)")
        print("="*50)
        