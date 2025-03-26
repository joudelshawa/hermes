from Agents.LLMAgent import Agent
import json
import re
from Utils.Helpers import remove_think
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
            top_p:int = 0.4
        ):
        super().__init__(base_llm, name, system_prompt, stream, max_iter, temperature, top_p)
        self.FORMAT = KGraph.model_json_schema()
        # self.FORMAT = {
        #     "type": "dictionary",
        #     "properties": {
        #     "nodes": {
        #         "type": "list",
        #         "items": {
        #             "type": "dictionary",
        #             "required": [
        #                 "id",
        #                 "label",
        #                 "color"
        #             ],
        #             "properties": {
        #                 "id": "integer",
        #                 "label": "string",
        #                 "color": "string"
        #             }
        #         }
        #     },
        #     "edges": {
        #         "type": "list",
        #         "items": {
        #             "type": "dictionary",
        #             "required": [
        #                 "from",
        #                 "to",
        #                 "label"
        #             ],
        #             "properties":{
        #                 "from": "integer",
        #                 "to": "integer",
        #                 "label": "string"
        #             }
        #         }
        #     },
        #     }
        # }

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

        # pattern = r'```json\s*(.*?)\s*```' # or --> ```json\s*(\{.*?\}|\[.*?\])\s*```
        # match = re.search(pattern, response, re.DOTALL)

        # if match:
        #     json_content = match.group(1)
        #     try:
        #         data = json.loads(json_content)
        #         result["extracted_response"] = json_content
        #     except json.JSONDecodeError:
        #         result["is_valid"] = False
        #         result["errors"].append("Invalid JSON format")
        #         return result
        # else:
        #     result["is_valid"] = False
        #     result["errors"].append("Invalid JSON format: Pattern \"```json\s*(.*?)\s*```\" not found in response")
        #     return result

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
        prompt = '### Start\ncurrent state:\n{}\n\nprompt:\n"""' + prompt + '"""\n\nnew state:\n'
        graph = super().run(prompt, context)
        validation = self.validateResponse(remove_think(graph))
        max_iter = self.MAX_ITERATIONS
        while(max_iter > 0):
            if(validation["is_valid"]):
                print("\t|---> Successfully Generated Knowledge Graph!")
                return validation["extracted_response"]
            else:
                print("\tERROR BY: HermesG")
                print(f"\t|---> OUTPUT: {remove_think(graph)}")
                print(f"\t|---> {validation['errors']}")
                print("\t|---> Trying again...")
                context = f"Your Previous Response: \"\"\"{remove_think(graph)}\"\"\"\n\n**NOTE**\nYour previous response had these errors -\n{validation['errors']}\n"
                graph = super().run(prompt, context)
                validation = self.validateResponse(graph)
            max_iter-=1
        
        print("="*50)
        print("Knowledge Graph Generation Error: Hermes-G was not able to generate validated output.\nExiting....")
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