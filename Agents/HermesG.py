from Agents.LLMAgent import Agent
import json
import re
from Utils.Helpers import *
from pydantic import BaseModel

class KGCreator(Agent):
    def __init__(
        self, 
        base_llm = "deepseek-r1:14b", 
        name = "", 
        system_prompt = "", 
        stream = False,
        max_iter:int = 3, 
        temperature:int = 0.3, 
        top_p:int = 0.4,
        osl_userPrompt:str = "",
        osl_assistantResponse:str = ""
    ):
        oneShotLearningExample = [
            {
                "role": "user",
                "content": osl_userPrompt
            },
            {
                "role": "assistant",
                "content": osl_assistantResponse
            }
        ]
        super().__init__(base_llm, name, system_prompt, stream, max_iter, temperature, top_p, oneShotLearningExample)
        self.FORMAT = KGraph.model_json_schema()
        

    def validateResponse(self, response:str) -> dict:
        # return super().validateResponse(response)
        """
        Validate LLM agent output against required graph structure
        Returns dict with 'is_valid' boolean and 'errors' list
        """
        result = {"is_valid": True, "errors": [], "extracted_response": f"**UNEXTRACTED**\n {response}"}
        response = response.replace("\"source\"", "\"from\"")

        try:
            data = json.loads(response)
            result["extracted_response"] = response
        except json.JSONDecodeError:
            result["is_valid"] = False
            result["errors"].append("Invalid JSON format")
            return result

        # Check top-level keys
        required_keys = {"nodes", "edges"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            result["is_valid"] = False
            result["errors"].append(f"Missing required keys: {', '.join(missing_keys)}")
            return result

        # Validate nodes
        node_ids = set()
        for i, node in enumerate(data["nodes"]):
            if not all(key in node for key in ["id", "label", "color"]):
                result["is_valid"] = False
                result["errors"].append(f"Node at index {i} missing required fields (id, label, or color)")
            
            if not isinstance(node["id"], int):
                result["is_valid"] = False
                result["errors"].append(f"Node {node.get('id')} has non-integer ID")
                
            if node["id"] in node_ids:
                result["is_valid"] = False
                result["errors"].append(f"Duplicate node ID: {node['id']}")
            node_ids.add(node["id"])

        # Validate edges
        invalid_edge = False
        existing_node_ids = {n["id"] for n in data["nodes"]}
        for i, edge in enumerate(data["edges"]):
            if not all(key in edge for key in ["from", "to", "label"]):
                result["is_valid"] = False
                result["errors"].append(f"Edge at index {i} missing required fields (from, to, or label)")
                invalid_edge = True
            if not invalid_edge:    
                for endpoint in ["from", "to"]:
                    if edge[endpoint] not in existing_node_ids:
                        result["is_valid"] = False
                        result["errors"].append(f"Edge {i} references non-existent node ID: {edge[endpoint]}")

        return result
    
    def run(self, prompt, context = ""):
        # prompt = '### Start\ncurrent state:\n{}\n\nprompt:\n"""' + prompt + '"""\n\nnew state:\n'
        max_iter = self.MAX_ITERATIONS
        
        while(max_iter > 0):
            print("\t|")
            print(f"\t|\tIteration [{self.MAX_ITERATIONS-max_iter+1}/{self.MAX_ITERATIONS}]")
            graph = remove_think(super().run(prompt, context))
            validation = self.validateResponse(graph)
            # Temporary Save
            saveGraphAsHTML(validation["extracted_response"], "Temp/")

            if(validation["is_valid"]):
                print("\t|\t|---> Successfully Generated Knowledge Graph!")
                return validation["extracted_response"]
            else:
                print("\t|\tERROR BY: HermesG")
                print(f"\t|\t|---> {validation['errors']}")
                print("\t\|\t|---> Trying again...")
                context = f"Your Previous Response: \"\"\"{remove_think(graph)}\"\"\"\n\n**NOTE**\nYour previous response had these errors -\n{validation['errors']}\n"
            max_iter-=1
        
        print("="*50)
        print("Knowledge Graph Generation Error\nExiting....")
        print("="*50)
        exit()

class KGraph(BaseModel):
    class Node(BaseModel):
        id: int
        label: str
        color: str
    class Edge(BaseModel):
        source: int
        to: int
        label: str
    
    nodes: list[Node]
    edges: list[Edge]