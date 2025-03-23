from agents import Hermes
import json
import os

if __name__ == "__main__":
    with open("agents/config.json", "r") as file:
        config = json.load(file)

    hermes = Hermes.HermesAgenticSystem(config=config)

    knowledge_graphs_path = "data/knowledge_graphs" # TODO update based on what aarat decides to name it
    qa_path = "data/qa_pairs"

    if os.path.isdir(knowledge_graphs_path):
        for filename in os.listdir(knowledge_graphs_path):
            print(filename)
            if filename.endswith('.txt'): # check what the file types of the knowledge graphs are
                file_path = os.path.join(knowledge_graphs_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    knowledge_graph = file.read()
                qa_pairs = hermes.getQA(knowledge_graph, context=config['Agents']['Hermes_Q']['Context'])
                qa_file_path = os.path.join(qa_path, filename)
                # FIX - will probably want to output to json so we can retrieve the questions more easily later
                with open(qa_path, "w") as file:
                    file.write(qa_pairs)
                print(f"Question-Answer pairs saved at {qa_file_path}")
    else:
        print(f"No knowledge graphs found in {knowledge_graphs_path}")