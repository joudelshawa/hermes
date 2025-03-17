"""
This contains the main agent class.
All agents will use this class as the parent class.
"""

import ollama

class LLM:
    _instance = None
    
    @classmethod
    def getInstance(cls, model_name:str):
        if cls._instance is None:
            cls._instance = ollama.Client()
        return cls._instance

class Agent:
    def __init__(self, base_llm:str = "llama3", name:str = "", systemPrompt:str = "", stream:bool = False):
        self.llm_client = LLM.getInstance(base_llm)
        self.llm = base_llm
        self.name = f"{name}_({base_llm})"
        self.systemPrompt = systemPrompt
        self.stream = stream

    def run(self, prompt:str, context:str = ""):
        """
        context addition remains
        i.e. ==> corrective prompts
        """
        response = self.llm_client.chat(model="llama3", messages=self._get_msgs(prompt=prompt, context=context))
        return response['message']['content']
    
    def validateResponse(self, response):
        """
        A method to valdiate response for the agent.
        Should be different for each agent.
        
        Input:
            response - Agent's output
        Output:
            (bool, error) - (True, None) if validated else (False, "Error: ...")
        """
        pass
    
    def _get_msgs(self, prompt:str, context:str = ""):
        msgs = [
            {
              "role": "system", # "system" is a prompt to define how the model should act.
              "content": self.systemPrompt # system prompt should be written here
            }
        ]

        if context != "":
            msgs.append(
                {
                    "role": "system",
                    "content": context
                }
            )
        
        msgs.append(
            {
              "role": "user", # "user" is a prompt provided by the user.
              "content": prompt # user prompt should be written here
            }
        )
        return msgs
    
    def _remove_think(self, text:str):
        s = text.split("</think>")
        return s[1]


class ReportCreator(Agent):
    def __init__(self, base_llm = "llama3", name = "", system_prompt = "", stream = False):
        super().__init__(base_llm, name, system_prompt, stream)

    def validateResponse(self, response):
        # Add logic for response validation
        # look at super class for information on input output 
        
        return 
    
    def run(self, prompt, context = ""):
        # Response generation step
        response = super().run(prompt, context)
        valdiated_response = None
        
        # Validation Step


        # Re-run Step if validation fails


        return valdiated_response

class KGCreator(Agent):
    def __init__(self, base_llm = "llama3", name = "", system_prompt = "", stream = False):
        super().__init__(base_llm, name, system_prompt, stream)
    
class QACreator(Agent):
    def __init__(self, base_llm = "llama3", name = "", system_prompt = "", stream = False):
        super().__init__(base_llm, name, system_prompt, stream)
    
class AnswerValidator(Agent):
    def __init__(self, base_llm = "llama3", name = "", system_prompt = "", stream = False):
        super().__init__(base_llm, name, system_prompt, stream)