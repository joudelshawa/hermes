from Agents.LLMAgent import Agent
import json
import re
from Utils.Helpers import remove_think
from pydantic import BaseModel

class QACreator(Agent):
    def __init__(
            self,
            base_llm = "deepseek-r1:14b", 
            name = "", system_prompt = "", 
            stream = False,
            max_iter:int=3, 
            temperature:int = 0.3, 
            top_p:int = 0.4
        ):
       super().__init__(base_llm, name, system_prompt, stream, max_iter, temperature, top_p)
       self.FORMAT = QAPairs.model_json_schema()
    
    def validateResponse(self, response):
        """
        Validate LLM agent output against required question + answer pairs structure
        Returns dict with 'is_valid' boolean and 'errors' list
        """
        result = {"is_valid": True, "errors": [], "extracted_response": f"**UNEXTRACTED**\n {response}"}
        # print(response)
        try:
            data = json.loads(response)["pairs"]
            result["extracted_response"] = json.dumps(data)
        except json.JSONDecodeError:
            result["is_valid"] = False
            # reminding it of the structure required
            result["errors"].append("Invalid JSON format. Needs to be: ```json[{'Question': <text>, 'Answer': <one-word answer>}, { 'Question': <text>, 'Answer': <one-word answer>}]```")
            return result
        
        # pattern = r'```json\s*(.*?)\s*```' # or --> ```json\s*(\{.*?\}|\[.*?\])\s*```
        # match = re.search(pattern, response, re.DOTALL)

        # if match:
        #     json_content = match.group(1)
        #     try:
        #         data = json.loads(json_content)
        #         result["extracted_response"] = json_content
        #     except json.JSONDecodeError:
        #         result["is_valid"] = False
        #         # reminding it of the structure required
        #         result["errors"].append("Invalid JSON format. Needs to be: ```json[{'Question': <text>, 'Answer': <one-word answer>}, { 'Question': <text>, 'Answer': <one-word answer>}]```")
        #         return result
        # else:
        #     result["is_valid"] = False
        #     result["errors"].append("Invalid JSON format. Needs to be: ```json[{'Question': <text>, 'Answer': <one-word answer>}, { 'Question': <text>, 'Answer': <one-word answer>}]```")
        #     return result

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

            standardized_item = {k.lower(): v for k, v in item.items()} # setting to lowercase so it's fine if they say Question vs question
            keys = set(standardized_item.keys())
            if keys != {"question", "answer"}:
                result["is_valid"] = False
                result["errors"].append(f"Item {i} missing required keys or has extra keys: {keys}")
            
            # checking that both values are not empty strings
            if not isinstance(standardized_item.get("question"), str) or not standardized_item.get("question").strip():
                result["is_valid"] = False
                result["errors"].append(f"Item {i} has invalid or empty 'Question'.")
            
            if not isinstance(standardized_item.get("answer"), str) or not standardized_item.get("answer").strip():
                result["is_valid"] = False
                result["errors"].append(f"Item {i} has invalid or empty 'Answer'.")

        return result
    
    def getSeparatedQA(self, qa_pairs:str):
        qa_pairs = json.loads(qa_pairs)
        questions = []
        answers = []
        for pair in qa_pairs:
            questions.append(pair["Question"])
            answers.append(pair["Answer"])
        
        return questions, answers
    
    def run(self, prompt, context = ""):
        prompt_dict = prompt
        prompt = json.dumps(prompt_dict, indent=2) # convert to string since its a json dict
        prompt = "### Start\ncurrent state:\n{}\n\nprompt:\n\"\"\"" + prompt + "\"\"\"\n\nnew state:\n"
        qa_pairs = remove_think(super().run(prompt, context))
        validation = self.validateResponse(qa_pairs)
        max_iter = self.MAX_ITERATIONS
        while(max_iter > 0):
            if(validation["is_valid"]):
                print("\t|---> Successfully Generated Question and Answer pairs!")
                return validation["extracted_response"]
            else:
                print("\tERROR BY: HermesQ")
                print(f"\t|---> {validation['errors']}")
                print("\t|---> Trying again...")
                print(f"\t|---> OUTPUT: {qa_pairs}")
                context = f"Your Previous Response: \"\"\"{validation['extracted_response']}\"\"\" had these errors -\n{validation['errors']}\n"
                qa_pairs = remove_think(super().run(prompt, context))
                validation = self.validateResponse(qa_pairs)
            max_iter-=1
        
        print("="*50)
        print("\tQUESTION ANSWER GENERATION ERROR!\nExiting...")
        print("="*50)
        exit()

class QAPairs(BaseModel):
    class QA(BaseModel):
        Question: str
        Answer: str
    
    pairs: list[QA]