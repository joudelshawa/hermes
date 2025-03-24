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
    def __init__(self, base_llm:str = "deepseek-r1:14b", name:str = "", systemPrompt:str = "", stream:bool = False):
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
        response = self.llm_client.chat(
            model=self.llm, 
            messages=self._get_msgs(prompt=prompt, context=context),
            options={"temperature": 0.2, "top_p": 0.5}
        )
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
        return
    
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