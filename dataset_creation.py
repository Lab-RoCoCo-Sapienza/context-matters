from utils import *
import os
from agent import Agent
from pprint import pprint

#for file in os.listdir("3dscenegraph"):
#    if any(substring in file for substring in ["Allensville"]):
#        print(file.replace(".npz", ""))
#        print("OLD\n\n")
#        graph = ((read_graph_from_path(Path("3dscenegraph/"+file))))
#        rooms = get_rooms(graph)
#        for room_id, room_data in rooms.items():
#            print(room_data['scene_category'])
#        print(get_verbose_scene_graph(graph))
#
#        # Add random laundry objects across the scene (random number of same type, in random rooms)
#        graph = add_objects(graph, add_laundry_objects(graph))
#
#        graph = add_descriptions_to_objects(graph)
#
#        print("NEW\n\n")
#        print(get_verbose_scene_graph(graph))
#        print("\n\n\n")
#        break

# Iterate over all .yaml files in "dataset/scenes"
for file in os.listdir(os.path.join("dataset","scenes")):
    if file.endswith(".yaml"):
        print(file)
        # Read the scene graph from the file
        graph = read_graph_from_path(Path("dataset/scenes/"+file))

        # Get the rooms in the scene
        rooms = get_rooms(graph)
        for room_id, room_data in rooms.items():
            print(room_data['scene_category'])
        print(get_verbose_scene_graph(graph))

        # Add random laundry objects across the scene (random number of same type, in random rooms)
        graph = add_objects(graph, add_laundry_objects(graph))

        graph = add_descriptions_to_objects(graph)

        print(get_verbose_scene_graph(graph))
        print("\n\n\n")
        break