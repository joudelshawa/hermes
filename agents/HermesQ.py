from Agents.LLMAgent import Agent

class QACreator(Agent):
    def __init__(self, base_llm = "deepseek-r1:14b", name = "", system_prompt = "", stream = False):
        super().__init__(base_llm, name, system_prompt, stream)
    
    def validateResponse(self, response):
        return super().validateResponse(response)
    
    def run(self, prompt, context = ""):
        return super().run(prompt, context)