from core.agent_manager import AgentManager

class MultiAgentPipeline:
    def __init__(self):
        self.manager = AgentManager()

    def run(self, pdf_path):
        print("Generating structured report...")
        report = self.manager.run("report", pdf_path=pdf_path)
        print("Generated structured report. Generating the knowledge graph...")
        knowledge_graph = self.manager.run("knowledge_graph", text=report)
        print("Generated the knowledge graph.")

        ### TODO: Add the rest of the agents.

        return knowledge_graph

