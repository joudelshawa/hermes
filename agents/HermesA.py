from agents.LLMAgent import Agent

class AnswerValidator(Agent):
    def __init__(self, base_llm = "deepseek-r1:14b", name = "", system_prompt = "", stream = False):
        super().__init__(base_llm, name, system_prompt, stream)

    def validateResponse(self, response): # same validation as hermesQ since we want to get back a json of q+a pairs
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
                # reminding it of the structure required
                result["errors"].append("Invalid JSON format. Needs to be: ```json[{'Question': <text>, 'Answer': <one-word answer>}, { 'Question': <text>, 'Answer': <one-word answer>}]```")
                return result
        else:
            result["is_valid"] = False
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

        return result
    
    def run(self, prompt, context = ""):
        prompt_dict = prompt
        prompt = json.dumps(prompt_dict, indent=2) # convert to string since its a json dict
        prompt = "**Start**\ncurrent state:\n{}\n\nprompt:\n" + prompt + "\nnew state:\n"
        av_pairs = super().run(prompt, context)
        validation = self.validateResponse(remove_think(av_pairs))
        max_iter = 2
        while(max_iter > 0):
            if(validation["is_valid"]):
                print("\tSuccessfully Generated Answer Validator pairs!")
                return validation["extracted_response"]
            else:
                print("\tERROR BY: HermesA")
                print(f"\t|---> {validation['errors']}")
                print(f"\t|---> OUTPUT: {remove_think(av_pairs)}")
                print("\t|---> Trying again...")
                context = f"**NOTE**\nTake note that your response should not have these errors -\n{validation['errors']}\n"
                av_pairs = super().run(prompt, context)
                validation = self.validateResponse(remove_think(av_pairs))
            max_iter-=1
        
        print("="*50)
        print("ANSWER VALIDATOR QUESTION ANSWER GENERATION ERROR!")
        print("="*50)
        exit()