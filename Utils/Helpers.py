import networkx as nx
import matplotlib.pyplot as plt
import json
from pyvis.network import Network

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

def saveGraphAsHTML(graph_data, folder_path):
    # Initialize the network
    graph_data = json.loads(graph_data)
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#222222",  # Dark background for visibility
        font_color="white",
        notebook=False
    )

    # Add nodes with labels and colors
    for node in graph_data["nodes"]:
        net.add_node(
            node["id"],
            label=node["label"],
            color=node["color"],
            font={"color": "black"}  # Text color for visibility
        )

    # Add edges with labels
    for edge in graph_data["edges"]:
        net.add_edge(
            edge["from"],
            edge["to"],
            title=edge["label"],  # Shows on hover
            label=edge["label"],  # Displayed on the edge
            color="white",  # Edge color
            arrows="to"
        )

    # Configure physics for better layout
    net.set_options("""
    {
    "physics": {
        "stabilization": {
        "enabled": true,
        "iterations": 1000
        },
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
        "gravitationalConstant": -50,
        "centralGravity": 0.01,
        "springLength": 100,
        "springConstant": 0.08,
        "damping": 0.4,
        "avoidOverlap": 0.1
        }
    }
    }
    """)

    # Save and show the graph
    net.write_html(folder_path + "KGraph.html")

def saveReportAsText(text_data, folder_path):
    with open(folder_path + "structured_report.md", "w") as file:
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

def saveAVPairsAsText(av_pairs, folder_path):
    with open(folder_path + "AVPairs.json", "w") as file:
        av_pairs_json = json.loads(av_pairs) # load json so properly formatting
        json.dump(av_pairs_json, file, indent=4)

def saveInvalidAnswersAsText(invAnswers, folder_path):
    with open(folder_path + "InvalidAnswers.txt", "a") as file:
        file.write(invAnswers)

def readQuestions(folder_path):
    with open(folder_path + "QAPairs.json", "r") as file:
        qa_pairs = json.load(file)
        questions = []
        for item in qa_pairs:
            standardized_item = {k.lower(): v for k, v in item.items()}
            if "question" in standardized_item:
                questions.append(standardized_item["question"])
        return questions

def readPairs(pair_type, folder_path):
    with open(folder_path + f"{pair_type.upper()}Pairs.json", "r") as file:
        return json.load(file)

def remove_think(text:str):
    if "</think>" not in text:
        return text
    s = text.split("</think>")
    return s[1].strip()
