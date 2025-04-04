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
    def __init__(
            self,
            base_llm:str,
            name:str, 
            systemPrompt:str, 
            stream:bool, 
            max_iter:int, 
            temperature:int, 
            top_p:int,
            oneShotLearningExample: list
        ):
        self.llm_client = LLM.getInstance(base_llm)
        self.llm = base_llm
        self.name = f"{name}_({base_llm})"
        self.systemPrompt = systemPrompt
        self.oneShotLearningExample = oneShotLearningExample
        self.stream = stream
        self.MAX_ITERATIONS = max_iter
        self.FORMAT = None
        self.TEMPERATURE = temperature
        self.TOP_P = top_p

    def run(self, prompt:str, context:str = ""):
        """
        context addition remains
        i.e. ==> corrective prompts
        """
        response = self.llm_client.chat(
            model=self.llm, 
            messages=self._get_msgs(prompt=prompt, context=context),
            options={"temperature": self.TEMPERATURE, "top_p": self.TOP_P, "num_ctx": 2048*4},
            format=self.FORMAT
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
        msgs:list = [
            {
              "role": "system", # "system" is a prompt to define how the model should act.
              "content": self.systemPrompt # system prompt should be written here
            },
        ]
        msgs.extend(self.oneShotLearningExample)

        if context != "":
            msgs.append(
                {
                    "role": "user",
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