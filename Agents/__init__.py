import json


OLLAMA_BASE_URL = "http://localhost:11434/api/chat"
with open("./config.json", "r") as file:
    CONFIG = json.load(file)

