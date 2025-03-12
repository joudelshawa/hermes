from agents.hermes_r import ReportGenerator
from agents.hermes_g import KnowledgeGraph

class AgentManager:
    def __init__(self):
        self.agents = {
            "report": ReportGenerator(),
            "knowledge_graph": KnowledgeGraph(),
        }

    def run(self, task_name, **kwargs):
        if task_name in self.agents:
            return self.agents[task_name].run(**kwargs)
        else:
            raise ValueError(f"Unknown task: {task_name}")


if __name__ == "__main__":
    manager = AgentManager()
    report = manager.run("report", pdf_path="sample.pdf")
    print(report)
