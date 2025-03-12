import requests

OLLAMA_BASE_URL = "http://localhost:11434/api/chat"

class OllamaClient:
    def __init__(self, model="deepseek-r1:14b"):
        self.model = model

    def generate_response(self, prompt: str, role_content:str):
        payload = {
          "model": self.model, # insert any models from Ollama that are on your local machine
          "messages": [
            {
              "role": "system", # "system" is a prompt to define how the model should act.
              "content": role_content # system prompt should be written here
            },
            {
              "role": "user", # "user" is a prompt provided by the user.
              "content": prompt # user prompt should be written here
            }
          ],
          "stream": False # returns as a full message rather than a streamed response
        }

        response = requests.post(OLLAMA_BASE_URL, json=payload)
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            raise Exception(f"API Error: {response.text}")

