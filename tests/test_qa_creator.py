from agents import Hermes
import json
import os

if __name__ == "__main__":
    with open("agents/config.json", "r") as file:
        config = json.load(file)

    hermes = Hermes.HermesAgenticSystem(config=config)

    knowledge_graphs_path = "data/knowledge_graphs_TMP" # TODO update based on what aarat decides to name it but now setting for testing
    qa_path = "data/qa_pairs"

    if os.path.isdir(knowledge_graphs_path):
        for filename in os.listdir(knowledge_graphs_path):
            print(filename)
            if filename.endswith('.txt'): # check what the file types of the knowledge graphs are
                
                file_path = os.path.join(knowledge_graphs_path, filename)
                
                with open(file_path, "r", encoding="utf-8") as file:
                    knowledge_graph = file.read()
                
                qa_pairs = hermes.getQA(
                                        knowledge_graph, 
                                        context=config['Agents']['Hermes_Q']['Context']
                                        )
                
                base_name = os.path.splitext(filename)[0]
                output_filename = base_name + ".json"

                qa_file_path = os.path.join(qa_path, output_filename)
                
                # output to json so we can retrieve the questions more easily later for the answer validator
                with open(qa_file_path, "w", encoding="utf-8") as file:
                    json.dump(qa_pairs, file, ensure_ascii=False, indent=2)
                
                print(f"Question-Answer pairs saved at {qa_file_path}")
    else:
        print(f"No knowledge graphs found in {knowledge_graphs_path}")