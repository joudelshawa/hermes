from pydantic import BaseModel
from Agents.LLMAgent import Agent
import json
from Utils.Helpers import *

class AnswerValidator(Agent):
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

            # if len(standardized_item.get("answer").split()) > 1:
            #     result["is_valid"] = False
            #     result["errors"].append(f"Invalid format (should be one word answers) for question {i}\nQuestion: \"{standardized_item.get('question')}\"\nAnswer: \"{standardized_item.get('answer')}\"")
             

        return result

    def _makeDictFormat(self, questions:list):
        dictQA = []
        for q in questions:
            dictQA.append({
                "Question": q,
                "Answer": ""
            })
        return json.dumps(dictQA, indent=4)
    
    def getSeparatedQA(self, qa_pairs:str):
        qa_pairs = json.loads(qa_pairs)
        questions = []
        answers = []
        for pair in qa_pairs:
            questions.append(pair["Question"])
            answers.append(pair["Answer"])
        
        return questions, answers
    
    
    def run(self, questions, unstructured_report, context = ""):
        """
        Input: 
            prompt: list[questions] 
        """

        prompt = self._makeDictFormat(questions) # convert to string since its a json dict
        # print(prompt)
        prompt = f"## Unstructured Report:\n\"\"\"\n{unstructured_report}\n\"\"\"\n\n## Questions To Answer:\n```json\n{prompt}```"
        max_iter = self.MAX_ITERATIONS
        while(max_iter > 0):
            print("\t|")
            print(f"\t|\tIteration [{self.MAX_ITERATIONS-max_iter+1}/{self.MAX_ITERATIONS}]")
            av_pairs = remove_think(super().run(prompt, context))
            validation = self.validateResponse(av_pairs)

            if(validation["is_valid"]):
                print("\t|\t|---> Successfully Generated Answer Validator pairs!")
                return validation["extracted_response"]
            else:
                print("\t|\tERROR BY: HermesA")
                print(f"\t|\t|---> {validation['errors']}")
                print("\t|\t|---> Trying again...")
                context = f"Your Previous Response: \n\"\"\"{validation['extracted_response']}\"\"\"\n---\n## NOTE\nThe following errors were made in your previous response: \n{validation['errors']}\n"
            max_iter-=1
        
        print("="*50)
        print("ANSWER VALIDATOR QUESTION ANSWER GENERATION ERROR!")
        print("="*50)
        exit()

class QAPairs(BaseModel):
    class QA(BaseModel):
        Question: str
        Answer: str
    
    pairs: list[QA]