import re

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional
from sentence_transformers import SentenceTransformer
import random

from collections import defaultdict

from agent import local_llm_call

def normalize_string(s: str) -> str:
    """Convert string to lowercase and replace spaces/special chars with underscores"""
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

def cosine_similarity(a: np.array, b: np.array) -> float:
    """
    Compute cosine similarity between two vectors a and b
    
    :param a: np.array of shape (n,)
    :param b: np.array of shape (n,)
    :return: float
    """
    assert a.shape == b.shape, "Input vectors are not of the same shape"
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def compute_goal_similarity(original_goal: str, relaxed_goal: str, model: SentenceTransformer) -> float:
    """
    Compute the similarity between two goals
    
    :param original_goal: Original goal string
    :param relaxed_goal: Relaxed goal string
    :param model: SentenceTransformer model
    :return: float
    """

    original_goal = normalize_string(original_goal)
    relaxed_goal = normalize_string(relaxed_goal)
    original_goal_embedding = model.encode(original_goal)
    relaxed_goal_embedding = model.encode(relaxed_goal)
    score = cosine_similarity(original_goal_embedding, relaxed_goal_embedding)
    
    return score

def filter_graph(graph: Dict, labels: Set[str]) -> Dict:
    """
    Filter the 3DSG graph to only include objects with labels in the 'labels' set
    
    :param graph: Dictionary containing the 3DSG
    :param labels: Set of labels (strings) to consider, e.g. {"room", "object"}
    :return new_graph: Dictionary containing the filtered 3DSG
    """
    new_graph = {}
    for key, item in graph.items():
        if key in labels:
            new_graph[key] = item
    return new_graph


def get_room_names(graph):
    return [room_data["scene_category"] + "_" + str(room_id) for room_id, room_data in graph["room"].items()]


def choose_random_room(graph):
    return random.choice(get_room_names(graph))


def add_descriptions_to_objects(graph):
    """
    Aggiunge la proprietà "description" a ogni oggetto nel grafo.

    :param graph: Dizionario contenente il 3DSG
    :return: Grafo aggiornato con descrizioni per gli oggetti
    """
    
    for obj_id, obj in graph["object"].items():
        obj["description"] = local_llm_call("Describe the object in one sentence in the following format A <color> <name of object> made of <material>\
        in which you replace <color> with the color of the object, <name of object> with the name of the object and <material> with the material of the\
        object. For example A brown table made of wood. Do not add other information.", obj["class_"])
    return graph


def add_objects(dict,):
    """
    Convert a list of dictionaries to a single dictionary in the format readble for scene graph
    
    :param list: List of dictionaries
    :return: Dictionary
    """


def save_graph(graph: Dict, path: str):
    """
    Save the 3DSG graph to a file
    
    :param graph: Dictionary containing the 3DSG
    :param path: Path to the file
    """
    np.savez(path, output=graph)

def add_objects(graph, dict_objects):
    """
    Aggiunge oggetti casuali a stanze casuali nel grafo.

    :param graph: Dizionario contenente il 3DSG
    :param num_objects: Numero di oggetti da aggiungere
    :return: Grafo aggiornato con nuovi oggetti
    """
    
    possible_objects = []
    for _, obj_value in dict_objects.items():
        for obj in obj_value:
            possible_objects.append({"class_": obj, "action_affordance": []})

    # Trova ID massimo attuale per evitare conflitti
    if graph["object"]:
        max_id = max(graph["object"].keys())
    else:
        max_id = 0

    # Ottieni stanze esistenti
    room_ids = list(graph["room"].keys())

    # Aggiungi oggetti random (scegli a caso il numero di oggetti uguali)
    for obj_data in possible_objects:
        obj_id = max_id + 1
        obj_type = obj_data["class_"]
        room_id = random.choice(room_ids)  # Scegli una stanza casuale

        # Genera posizione casuale all'interno della stanza scelta
        location = np.array([
            np.zeros(3),
            np.zeros(3),
            np.zeros(3)
        ])

        # Crea nuovo oggetto
        new_object = {
            "id": obj_id,
            "class_": obj_data["class_"],
            "action_affordance": obj_data["action_affordance"],
            "location": location,
            "parent_room": room_id,
        }

        # Aggiungi al grafo
        graph["object"][obj_id] = new_object
        max_id += 1  # Aggiorna ID massimo

    return graph




def read_graph_from_path(path: Path) -> Dict:
    """
    Read 3DSG from file (.npz) and returns it stored in a dictionary
    
    :param path: Path to the .npz file 
    :return: Dictionary containing the 3DSG
    """
    
    #print(path)
    assert (
        isinstance(path, Path)
    ), "Input file is not a Path"
    assert (
        str(path).endswith(".npz")
    ), "Input file is not .npz object"
    
    graph = np.load(path, allow_pickle=True)['output'].item()
    
    keeps = set(["object", "room"])
    graph = filter_graph(graph, keeps)
    
    return graph

def get_objects(graph: Dict) -> Dict:
    """
    Given a dictionary containing a 3DSG, return a dictionary containing only the objects
    
    :param graph: Dictionary containing the 3DSG
    :return: Dictionary containing only the objects
    """
    for objId, obj in graph["object"].items():
        graph["object"][objId]["class_"] = graph["object"][objId]["class_"].replace(" ", "_")
    
    return graph["object"]

def get_rooms(graph: Dict) -> Dict:
    """
    Given a dictionary containing a 3DSG, return a dictionary containing only the rooms
    
    :param graph: Dictionary containing the 3DSG
    :return: Dictionary containing only the rooms
    """
    return graph["room"]

def get_room_keypoints(graph: Dict, room_name: str) -> Optional[np.array]:
    """
    Get the keypoints of a room given its name. The keypoint represent the pose x,y,z
    of the room.

    :param graph: Dictionary containing the 3DSG
    :param room_name: String representing the room name
    :return: np.array of shape (3,) or None if the room does not exist
    """
    assert isinstance(room_name, str), "Room name is not a string"

    room_ = get_rooms(graph)
    keypoints = []
    for room_id, room_data in room_.items():
        if room_name == room_data['scene_category']:
            print("Room found with id: ", room_id)
            keypoints = np.array(room_data['location'], dtype=float)
            break
    return keypoints

def get_room_from_labels_and_pose(graph: Dict,
                            position: np.array, 
                            labels: Set[str]) -> Optional[str]:
    """
    Given a position (np.array), a set of labels, a dictionary of objects,
    and a dictionary of places, return the place in which we are likely located.

    :param graph: Dictionary containing the 3DSG
    :param position: np.array of shape (3,) representing (x, y, z)
    :param labels: set of labels (strings) to consider, e.g. {"microwave", "oven"}
    :return: place_name (string) or None if no match is found
    """
    assert position.shape == (3,) or position.shape == (3,1), "Position is not of shape (3,) or (3,1)"    
    
    objects_ = get_objects(graph)
    room_ = get_rooms(graph)
    
    objects = {}
    for key, item in objects_.items():
        objects[key] = {"class": item["class_"], "location": item["location"]}
    
    places = {}
    for obj_id, obj_data in objects_.items():
        parent_room = obj_data['parent_room']
        if parent_room in room_:
            scene_category = room_[parent_room]['scene_category']
            # Initialize the list if this scene_category is not yet in the dict
            if scene_category not in places:
                places[scene_category] = []
            places[scene_category].append(obj_id)

    
    closest_obj_id = None
    min_dist = float("inf")

    # 1) Find the closest object among the objects that match any label in 'labels'
    for obj_id, obj_info in objects.items():
        if obj_info["class"] in labels:
            dist = np.linalg.norm(obj_info["location"] - position)
            if dist < min_dist:
                min_dist = dist
                closest_obj_id = obj_id

    # If we did not find any object matching those labels, return None (or do something else)
    if closest_obj_id is None:
        print("No objects found for given labels.")
        return None

    # 2) Identify which place has that object_id
    for place_name, list_of_ids in places.items():
        if closest_obj_id in list_of_ids:
            return place_name

    # If we cannot find a place containing that object, return None (or do something else)
    return None

def embed_objects(objects: Dict, model: SentenceTransformer, with_material: Optional[bool] = False) -> Dict:
    """
    Given a dictionary of objects from a 3DSG, embeds the object descriptions using a SentenceTransformer model.
    
    :param objects: Dictionary containing the 3DSG objects
    :param model: Pre-trained SentenceTransformer model for embedding
    :param with_material: If True, include material information in the embedding
    :return: Dictionary with object embeddings
    """
    object_nodes = {}
    for key in list(objects.keys()):
        obj = objects[key]
        cls = obj['class_']
        if with_material:
            if obj['material']:
                materials = [m for m in list(obj['material']) if m != None]
            else:
                materials = []
            if materials:
                object_nodes[key] = {"label": f"{cls} made of {', '.join(materials)}"}
            else:
                object_nodes[key] = {"label": cls}
        else:
            object_nodes[key] = {"label": cls}
        object_nodes[key]["embedding"] = model.encode(object_nodes[key]["label"])
        
    return object_nodes

def find_top_5_similar(object_dict_with_embedding: dict) -> dict:
    """
    Given a dictionary of objects from a 3DSG with embeddings,
    for each object finds the top 5 most similar objects.
    
    :param objects: Dictionary containing the 3DSG objects with their embeddings
    :return: Dictionary with top 5 similar objects for each object
    """
    
    object_dict = object_dict_with_embedding
    
    # Prepare a place to store the results
    results = {}

    # For each object in the dictionary:
    for key_a, data_a in object_dict.items():
        embedding_a = data_a["embedding"]
        
        # List to store (other_object_key, similarity_score)
        similarities = []
        
        # Compare with every other object
        for key_b, data_b in object_dict.items():
            # Skip itself
            if key_a == key_b:
                continue
            
            embedding_b = data_b["embedding"]
            sim = cosine_similarity(embedding_a, embedding_b)
            similarities.append((key_b, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top 5
        top_5 = similarities[:5]
        
        # Store results - you could store in the original dict or return separately
        results[key_a] = top_5
    
    return results

def get_room_objects_from_pose(graph: Dict, pose: np.array) -> Dict:
    """
    Given a position (np.array) and a 3DSG, 
    return the objects in the room in which the position is located.
    :param graph: Dictionary containing the 3DSG
    :param pose: np.array of shape (3,) representing (x, y, z)
    :return: Dictionary containing the objects in the room
    """
    assert pose.shape == (3,) or pose.shape == (3,1), "Position is not of shape (3,) or (3,1)"
    
    objects_ = get_objects(graph)
    room_ = get_rooms(graph)


    # 1. Find the closest room
    min_distance = float('inf')
    closest_room_id = None

    for room_id, room_data in room_.items():
        room_location = np.array(room_data['location'], dtype=float)
        distance = np.linalg.norm(pose - room_location)
        
        if distance < min_distance:
            min_distance = distance
            closest_room_id = room_id
    print(f"Closest room: {closest_room_id}")

    # 2. Gather all objects in that room
    objects_in_closest_room = [
        obj_data
        for obj_data in objects_.values()
        if obj_data['parent_room'] == closest_room_id
    ]

    return objects_in_closest_room

def load_planning_log(file_path):
    """
    Load the planning log from a file and return it as a list of strings.
    
    :param file_path: Path to the file containing the planning log
    :return: List of strings
    """
    with open(file_path, "r") as file:
        log = file.readlines()
    return log
from collections import defaultdict
from typing import Dict

def get_verbose_scene_graph(graph: Dict, as_string=True) -> str:
    """
    Given a 3DSG, return a verbose description of the scene graph with meaningful information,
    including room names, objects, and their descriptions.

    :param graph: Dictionary containing the 3DSG
    :return: String with the verbose scene graph
    """
    
    rooms = graph.get("room", {})
    objects = graph.get("object", {})
    
    # 1. Create a label for each room: always append "_roomId".
    room_id_to_label = {
        r_id: f"{room_info.get('scene_category', 'UnnamedRoom')}_{r_id}"
        for r_id, room_info in rooms.items()
    }

    # 2. Create a label for each object: always append "_objId".
    obj_id_to_label = {
        o_id: f"{obj_info.get('class_', 'UnnamedObject')}_{o_id}"
        for o_id, obj_info in objects.items()
    }

    # 3. Group objects by their roomId.
    room_to_objects = defaultdict(list)
    for o_id, obj_info in objects.items():
        r_id = obj_info.get('parent_room')
        if r_id in rooms:
            obj_label = obj_id_to_label[o_id]
            obj_description = obj_info.get('description', 'No description available')
            room_to_objects[r_id].append((obj_label, obj_description))
    
    # 4. Construct the output string.
    if as_string:
        lines = []
        for r_id in rooms:
            room_label = room_id_to_label[r_id]
            objs_in_room = room_to_objects.get(r_id, [])
            obj_lines = []
            if objs_in_room:
                for obj in objs_in_room:
                    obj_lines.append(f"{obj[0]} - {obj[1]}")
            
                objs_str = "\n  - " + "\n  - ".join(obj_lines)
            else:
                objs_str = "  - No objects"
            lines.append(f"\n\n{room_label}:{objs_str}")

        return "\n".join(lines)
    else:
        output = {}
        for r_id in rooms:
            room_label = room_id_to_label[r_id]
            objs_in_room = room_to_objects.get(r_id, [])
            output[room_label] = objs_in_room
        return output
    


def copy_file(src: str, dst: str):
    """
    Copy a file from src to dst
    
    :param src: Source file path
    :param dst: Destination file path
    """
    with open(src, "r") as f:
        data = f.read()

    with open(dst, "w") as f:
        f.write(data)

def save_file(data: str, path: str):
    """
    Save a string to a file
    
    :param data: String to save
    :param path: Path to the file
    """
    with open(path, "w") as f:
        f.write(data)

def setup_planning_log(path: str):
    with open(path, mode="w") as file:
        file.write("Planning log\n")

def print_to_planning_log(path: str, message: str):
    with open(path, mode="a") as file:
        file.write(message + "\n")

def load_knowledge_graph(kg_path):
    with open(kg_path, 'rb') as f:
        # Load json file
        kg = json.load(f)
    return kg

def print_red(text: str):
    print(f"\033[91m{text}\033[0m")

def print_green(text: str):
    print(f"\033[92m{text}\033[0m")

def print_yellow(text: str):
    print(f"\033[93m{text}\033[0m")

def print_blue(text: str):
    print(f"\033[94m{text}\033[0m")

def print_magenta(text: str):
    print(f"\033[95m{text}\033[0m")

def print_cyan(text: str):
    print(f"\033[96m{text}\033[0m")