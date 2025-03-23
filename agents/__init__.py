import json


OLLAMA_BASE_URL = "http://localhost:11434/api/chat"
with open("agents/config.json", "r") as file:
    CONFIG = json.load(file)

