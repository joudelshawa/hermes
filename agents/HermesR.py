from Agents.LLMAgent import Agent
import re
from Utils.Helpers import *

class ReportCreator(Agent):
    def __init__(self, base_llm = "deepseek-r1:14b", name = "", system_prompt = "", stream = False):
        super().__init__(base_llm, name, system_prompt, stream)
        self.main_headings = [
            "Patient Details",
            "Complaint",
            "History",
            "Discharge",
            "Results",
            "Medications"
        ]
        self.MAX_ITERATION = 2


    def validateResponse(self, response):
        # Add logic for response validation
        # look at super class for information on input output
        # pattern = r"```output(.*)```"
        # match = re.search(pattern, response)
        result = {"is_valid": True, "response": response, "errors": []}
        # if match:
        #     result["is_valid"] = True
        #     result["response"] = match.group(1).strip()
        # else:
        #     result["errors"].append("Invalid Response - ```output\n...\n``` pattern not found")
        #     return result

        found_headings = re.findall(r"\*\*(.*?)\*\*", result["response"])

        # Check if any major heading word is missing
        for heading in self.main_headings:
            if not any(word in found_headings for word in heading.split()):
                result["errors"].append(f"{heading} not found in Headings")
                result["is_valid"] = False
        return result
    
    def run(self, prompt, context = ""):
        # Response generation step
        prompt = f'Prompt:\n"""\n{prompt}\nOutput:\n"""'
        response = remove_think(super().run(prompt, context))
        validation = self.validateResponse(response)
        max_iter = self.MAX_ITERATION

        while(max_iter > 0):

            if validation["is_valid"]:
                print("\tSuccessfully Generated Structured Report!")
                return validation["response"]
            else:
                print("\tERROR BY: HermesR")
                print(f"\t|---> {validation['errors']}")
                print(f"\t|---> OUTPUT: {response}")
                print("\t|---> Trying again...")
                
                context = f"**NOTE**\nTake note that your response should not have these errors -\n{validation['errors']}\n"
                response = super().run(prompt, context)
                validation = self.validateResponse(response)
            max_iter -= 1
        
        print("="*50)
        print("STRUCTURED REPORT CREATION ERROR!!\nExiting....")
        print("="*50)
        exit()