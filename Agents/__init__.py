import json
from .HermesA import AnswerValidator
from .HermesG import KGCreator
from .HermesR import ReportCreator
from .HermesQ import QACreator

OLLAMA_BASE_URL = "http://localhost:11434/api/chat"
with open("agents/config.json", "r") as file:
    CONFIG = json.load(file)

def getAgentPrompt(promptFile):
    with open(promptFile) as file:
        return file.read()
    
__all__ = ["AnswerValidator", "KGCreator", "ReportCreator", "QACreator", "CONFIG", "getAgentPrompt"]
