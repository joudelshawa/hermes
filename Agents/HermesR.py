from Agents.LLMAgent import Agent
import re
from Utils.Helpers import *

class ReportCreator(Agent):
    def __init__(
            self, 
            base_llm = "deepseek-r1:14b", 
            name = "", 
            system_prompt = "", 
            stream = False,
            max_iter = 3,
            temperature:int = 0.3, 
            top_p:int = 0.4,
            osl_userPrompt:str = "",
            osl_assistantResponse:str = "",
            contextLengthMultiplier:int = 8
    ):
        oneShotLearningExample = []
        if osl_userPrompt != "" and osl_assistantResponse != "":
            oneShotLearningExample = [
                {
                    "role": "user",
                    "content": osl_userPrompt
                },
                {
                    "role": "assistant",
                    "content": osl_assistantResponse
                }
            ]
        else: 
            print(f"{name}: Not Using One-Shot-Learning")
        super().__init__(base_llm, name, system_prompt, stream, max_iter, temperature, top_p, oneShotLearningExample, contextLengthMultiplier)
        self.main_headings = [
            "patient",
            "complaint",
            "history",
            "discharge",
            "medications"
        ]


    def validateResponse(self, response):
        # Add logic for response validation
        # look at super class for information on input output
        # pattern = r"```output(.*)```"
        # match = re.search(pattern, response)
        result = {"is_valid": True, "response": response, "errors": ""}
        # if match:
        #     result["is_valid"] = True
        #     result["response"] = match.group(1).strip()
        # else:
        #     result["errors"].append("Invalid Response - ```output\n...\n``` pattern not found")
        #     return result
        
        found_headings = re.findall(r"\*\*(.*?)\*\*", result["response"])
        found_headings = [
            word.lower()
            for heading in found_headings
            for word in re.sub(r'[^a-zA-Z ]', '', heading).split()
        ]

        missing_headings = []
        # Check if any major heading word is missing
        for heading in self.main_headings:
            if not any(word in found_headings for word in heading.lower().split()):
                # print(f"\t\tDID NOT FIND HEADING: {heading}")
                missing_headings.append(heading)
                result["is_valid"] = False
            else:
                pass
                # print(f"\t\tFOUND HEADING: {heading}")

        if not result["is_valid"]:
            result["errors"] += f"You must include these words in your headings: {', '.join(missing_headings)}"
        return result
    
    def run(self, prompt, context = ""):
        # Response generation step
        max_iter = self.MAX_ITERATIONS

        while(max_iter > 0):
            print("\t|")
            print(f"\t|\tIteration [{self.MAX_ITERATIONS-max_iter+1}/{self.MAX_ITERATIONS}]")
            response = remove_think(super().run(prompt, context))
            validation = self.validateResponse(response)
            # Temporary Save
            saveReportAsText(validation["response"], "Temp/")
            
            if validation["is_valid"]:
                print("\t|\t|---> Successfully Generated Structured Report!")
                return validation["response"]
            else:
                print("\t|\tERROR BY: HermesR")
                print(f"\t|\t|---> {validation['errors']}")
                print("\t|\t|---> Trying again...")        
                context = f"Your Previous Response: \"\"\"{validation['response']}\"\"\"\n## NOTE\nThe following errors were made in your previous response: \n{validation['errors']}\n"
            max_iter -= 1
        
        print("="*50)
        print("STRUCTURED REPORT CREATION ERROR!!\nExiting....")
        print("="*50)
        exit()