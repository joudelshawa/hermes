import networkx as nx
import matplotlib.pyplot as plt
import json
import re

def saveGraphAsImage(graph_data, folder_path):
    # Create a graph
    G = nx.DiGraph()

    # Add nodes and edges from your data
    graph_data = json.loads(graph_data)
    for node in graph_data["nodes"]:
        G.add_node(node["id"], label=node["label"])
        
    for edge in graph_data["edges"]:
        G.add_edge(edge["from"], edge["to"], label=edge["label"])

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Layout engine
    labels = {node["id"]: node["label"] for node in graph_data["nodes"]}
    edge_labels = {(edge["from"], edge["to"]): edge["label"] for edge in graph_data["edges"]}

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, labels=labels, with_labels=True, node_color="white", edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
    plt.savefig(folder_path + "KGraph.png", bbox_inches="tight")

def saveGraphAsText(graph_data, folder_path):
    graph_data = json.loads(graph_data)
    with open(folder_path + "KGraph.json", "w") as file:
        json.dump(graph_data, file, indent=4)

def saveReportAsText(text_data, folder_path):
    with open(folder_path + "structured_report.txt", "w") as file:
        file.write(text_data)

def readStructuredReport(folder_path):
    with open(folder_path + "structured_report.txt", "r") as file:
        return file.read()
    
def readUnstructuredReport(folder_path):
    with open(folder_path + "report.txt", "r") as file:
        return file.read()
    
def readKGraph(folder_path):
    with open(folder_path + "KGraph.json", "r") as file:
        return json.load(file)

def saveQAPairsAsText(qa_pairs, folder_path):
    with open(folder_path + "QAPairs.json", "w") as file:
        qa_pairs_json = json.loads(qa_pairs) # load json so properly formatting
        json.dump(qa_pairs_json, file, indent=4)
    
def remove_think(text:str):
    s = text.split("</think>")
    return s[1].strip()
