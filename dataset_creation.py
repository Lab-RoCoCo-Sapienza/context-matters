from utils import *
import os
from agent import Agent
from pprint import pprint
import json

dataset_dir = "dataset"

for root_dir, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".yaml"):
            print(file)
            path = os.path.join(root_dir, file)

            #Load json path
            with open(path) as f:
                scenes = json.load(f)
                f.close()


            for scene in scenes.keys():
                objects = scenes[scene]["objects"]
                graph_id = scenes[scene]["graph"]
                domain_dir = os.path.join("dataset", graph_id)
                os.makedirs(domain_dir, exist_ok=True)
                problem_dir = os.path.join(domain_dir,scene)
                if not os.path.exists(problem_dir):
                    os.makedirs(problem_dir, exist_ok=True)
                    path_graph = os.path.join("3dscenegraph",os.listdir("3dscenegraph")[graph_id in os.listdir("3dscenegraph")])
                    print(path_graph)
                    graph = read_graph_from_path(Path(path_graph))
                    print(scenes[scene]["description"])
                    graph = (add_objects(graph, objects))
                    graph = add_descriptions_to_objects(graph)
                    save_graph(graph,  os.path.join(problem_dir, graph_id.replace(".npz", "enhanced.npz")))
                    task_path = os.path.join(problem_dir, "task.txt")
                    with open(task_path, "w") as f:
                        f.write(scenes[scene]["description"])
                        f.close()
                    
                    description_path = os.path.join(problem_dir, "description.txt")
                    with open(description_path, "w") as f:
                        f.write(scenes[scene]["description"])
                        f.close()
                else:
                    print("Already exists")

