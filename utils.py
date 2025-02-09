import json

def load_knowledge_graph(kg_path):
    with open(kg_path, 'rb') as f:
        # Load json file
        kg = json.load(f)
    return kg