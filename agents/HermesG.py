from Agents.LLMAgent import Agent
import json

class KGCreator(Agent):
    def __init__(self, base_llm = "deepseek-r1:14b", name = "", system_prompt = "", stream = False):
        super().__init__(base_llm, name, system_prompt, stream)

    def validateResponse(self, response) -> dict:
        # return super().validateResponse(response)
        """
        Validate LLM agent output against required graph structure
        Returns dict with 'is_valid' boolean and 'errors' list
        """
        result = {"is_valid": True, "errors": []}
        
        try:
            data = json.loads(response)
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
        existing_node_ids = {n["id"] for n in data["nodes"]}
        for i, edge in enumerate(data["edges"]):
            if not all(key in edge for key in ["from", "to", "label"]):
                result["is_valid"] = False
                result["errors"].append(f"Edge at index {i} missing required fields (from, to, or label)")
                
            for endpoint in ["from", "to"]:
                if edge[endpoint] not in existing_node_ids:
                    result["is_valid"] = False
                    result["errors"].append(f"Edge {i} references non-existent node ID: {edge[endpoint]}")

        return result
    
    def run(self, prompt, context = ""):
        prompt = "**Start**\ncurrent state:\n{}\n\nprompt:\n" + prompt
        graph = super().run(prompt, context)
        validation = self.validateResponse(graph)
        max_iter = 2
        while(max_iter > 0):
            if(validation["is_valid"]):
                return graph
            else:
                print("ERROR BY: HermesG")
                print(f"{validation["errors"]}")
                print("Trying again...")
                prompt = "**Start**\ncurrent state:\n{}\n\nprompt:\n" + prompt
                context = f"**WARNING**\nYour previous response had these errors -\n{validation['errors']}\nDo not repeat these errors"
                graph = super().run(prompt, context)
                validation = self.validateResponse(graph)
            max_iter-=1
    