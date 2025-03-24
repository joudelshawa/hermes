from agents.LLMAgent import Agent
import json
import re
from utils.helpers import remove_think

class QACreator(Agent):
    def __init__(self, base_llm = "deepseek-r1:14b", name = "", system_prompt = "", stream = False):
        super().__init__(base_llm, name, system_prompt, stream)
    
    def validateResponse(self, response):
        """
        Validate LLM agent output against required question + answer pairs structure
        Returns dict with 'is_valid' boolean and 'errors' list
        """
        result = {"is_valid": True, "errors": [], "extracted_response": f"**UNEXTRACTED**\n {response}"}

        pattern = r'```json\s*(.*?)\s*```' # or --> ```json\s*(\{.*?\}|\[.*?\])\s*```
        match = re.search(pattern, response, re.DOTALL)

        if match:
            json_content = match.group(1)
            try:
                data = json.loads(json_content)
                result["extracted_response"] = json_content
            except json.JSONDecodeError:
                result["is_valid"] = False
                result["errors"].append("Invalid JSON format")
                return result
        else:
            result["is_valid"] = False
            result["errors"].append("Invalid JSON format")
            return result

        # check layout
        if not isinstance(data, list): # check if its a list first
            result["is_valid"] = False
            result["errors"].append("Top-level JSON must be a list.")
            return result

        # validate each Q&A pair
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                result["is_valid"] = False
                result["errors"].append(f"item at index {i} is not a dictionary.")
                continue

            keys = set(item.keys())
            if keys != {"Question", "Answer"}:
                result["is_valid"] = False
                result["errors"].append(f"Item {i} missing required keys or has extra keys: {keys}")
            
            # checking that both values are not empty strings
            if not isinstance(item.get("Question"), str) or not item.get("Question").strip():
                result["is_valid"] = False
                result["errors"].append(f"Item {i} has invalid or empty 'Question'.")
            
            if not isinstance(item.get("Answer"), str) or not item.get("Answer").strip():
                result["is_valid"] = False
                result["errors"].append(f"Item {i} has invalid or empty 'Answer'.")

        return result
    
    def run(self, prompt, context = ""):
        prompt_dict = prompt
        prompt = json.dumps(prompt_dict, indent=2) # convert to string since its a json dict
        prompt = "**Start**\ncurrent state:\n{}\n\nprompt:\n" + prompt + "\nnew state:\n"
        qa_pairs = super().run(prompt, context)
        validation = self.validateResponse(remove_think(qa_pairs))
        max_iter = 2
        while(max_iter > 0):
            if(validation["is_valid"]):
                print("\tSuccessfully Generated Question and Answer pairs!")
                return validation["extracted_response"]
            else:
                print("\tERROR BY: HermesQ")
                print(f"\t|---> {validation['errors']}")
                print(f"\t|---> OUTPUT: {qa_pairs}")
                print("\t|---> Trying again...")
                context = f"**NOTE**\nTake note that your response should not have these errors -\n{validation['errors']}\n"
                qa_pairs = super().run(prompt, context)
                validation = self.validateResponse(remove_think(qa_pairs))
            max_iter-=1
        
        print("="*50)
        print("QUESTION ANSWER GENERATION ERROR!")
        print("="*50)
        exit()
