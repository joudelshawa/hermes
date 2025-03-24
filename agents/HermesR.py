from agents.LLMAgent import Agent

class ReportCreator(Agent):
    def __init__(self, base_llm = "deepseek-r1:14b", name = "", system_prompt = "", stream = False):
        super().__init__(base_llm, name, system_prompt, stream)

    def validateResponse(self, response):
        # Add logic for response validation
        # look at super class for information on input output 
        
        return 
    
    def run(self, prompt, context = ""):
        # Response generation step
        response = super().run(prompt, context)
        
        # Validation Step
        # TODO: Add validation logic
        valdiated_response = response


        # Re-run Step if validation fails


        return valdiated_response