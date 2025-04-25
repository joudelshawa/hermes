import networkx as nx
import matplotlib.pyplot as plt
import json
import re
from pyvis.network import Network
from collections import Counter

# ===========================================
# ============== SAVE STUFF =================
# ===========================================
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
    plt.savefig(folder_path + "/KGraph.png", bbox_inches="tight")

def saveGraphAsText(graph_data, folder_path):
    graph_data = json.loads(graph_data)
    with open(folder_path + "/KGraph.json", "w") as file:
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
    net.write_html(folder_path + "/KGraph.html")

def saveReportAsText(text_data, folder_path):
    with open(folder_path + "/structured_report.md", "w") as file:
        file.write(text_data)

def saveQAPairsAsText(qa_pairs, folder_path):
    with open(folder_path + "/QAPairs.json", "w") as file:
        qa_pairs_json = json.loads(qa_pairs) # load json so properly formatting
        json.dump(qa_pairs_json, file, indent=4)

def saveAVPairsAsText(av_pairs, folder_path):
    with open(folder_path + "/AVPairs.json", "w") as file:
        av_pairs_json = json.loads(av_pairs) # load json so properly formatting
        json.dump(av_pairs_json, file, indent=4)

def saveInvalidAnswersAsText(invAnswers, folder_path):
    with open(folder_path + "/InvalidAnswers.txt", "a") as file:
        file.write(invAnswers)

# ===========================================
# ============== READ STUFF =================
# ===========================================
def readStructuredReport(folder_path):
    with open(folder_path + "/structured_report.txt", "r") as file:
        return file.read()
    
def readUnstructuredReport(folder_path):
    with open(folder_path + "/report.txt", "r") as file:
        return file.read()
    
def readKGraph(folder_path):
    with open(folder_path + "/KGraph.json", "r") as file:
        return json.load(file)

def readQuestions(folder_path):
    with open(folder_path + "/QAPairs.json", "r") as file:
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

# ===========================================
# ============= EXTRA STUFF =================
# ===========================================
def remove_think(text:str):
    if "</think>" not in text:
        return text
    s = text.split("</think>")
    return s[1].strip()

def getFormattedElapsedTime(start, end) -> str:
    elapsed = end-start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # Format the result as MM:SS
    return f"{minutes}min {seconds:02d}sec"

def extractAllNumbers(text:str):
        pattern = r"(?<![a-zA-Z])\d+\.\d+|(?<![a-zA-Z])(?<!\n)\d+" #(?<![a-zA-Z])\d+\.\d+|(?<![a-zA-Z])\d+
        matches = [round(float(num), 8) for num in re.findall(pattern, text)]
        return matches
    
def areNumbericallyEquivalent(text1:str, text2:str):
    nums1 = extractAllNumbers(text1)
    nums2 = extractAllNumbers(text2)
    return sorted(nums1) == sorted(nums2)

def getMissingNumbers(text_og: str, text_gen: str, return_nums_og: bool = False):
    nums_og = extractAllNumbers(text_og)
    nums_gen = extractAllNumbers(text_gen)

    # Use Counter to count occurrences like a multiset
    counter_og = Counter(nums_og)
    counter_gen = Counter(nums_gen)

    # Subtract counters to find what's missing
    missing_in_gen = dict(counter_og - counter_gen) 
    missing_in_og = dict(counter_gen - counter_og)
    # print("Missing: ", missing_in_gen)
    # print("Hallucinated: ", missing_in_og)
    
    if return_nums_og:
        return missing_in_og, missing_in_gen, nums_og, nums_gen
    else:
        return missing_in_og, missing_in_gen
     
