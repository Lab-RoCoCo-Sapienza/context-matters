from utils import *
import os
from pprint import pprint
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

for root_dir, dirs, files in os.walk(DATASET_DIR):
    for file in files:

        # Load JSON task configuration file
        if file.endswith(".json"):
            print(f"Generating split for task: {file}")
            path = os.path.join(root_dir, file)

            # Get file name without extension
            task_name = os.path.splitext(file)[0]

            #Load json path
            with open(path) as f:
                problems = json.load(f)
                f.close()

            # Create a dir for the current task, this dir will contain all the scenes related to this task
            domain_dir = os.path.join("dataset", task_name)
            os.makedirs(domain_dir, exist_ok=True)

            for problem_id in problems.keys():

                objects = problems[problem_id]["objects"]

                graphs = problems[problem_id]["graph"]
                for graph_id in graphs:
                    print(graph_id)


                    scene_dir = os.path.join(domain_dir, graph_id)
                    os.makedirs(scene_dir, exist_ok=True)
                    
                    problem_dir = os.path.join(scene_dir, problem_id)
                    if not os.path.exists(problem_dir):
                        os.makedirs(problem_dir, exist_ok=True)

                        path_graph = os.path.join("3dscenegraph", "3DSceneGraph_" + graph_id + ".npz")
                        print(path_graph)

                        graph = read_graph_from_path(Path(path_graph))
                        #pprint(graph)
                        print(problems[problem_id]["description"])

                        graph = (add_objects(graph, objects))
                        graph = add_descriptions_to_objects(graph)

                        save_graph(graph,  os.path.join(problem_dir, graph_id.replace(".npz", "enhanced.npz")))

                        task_path = os.path.join(problem_dir, "task.txt")
                        with open(task_path, "w") as f:
                            f.write(problems[problem_id]["description"])
                            f.close()
                        
                        description_path = os.path.join(problem_dir, "description.txt")
                        with open(description_path, "w") as f:
                            f.write(problems[problem_id]["description"])
                            f.close()

                        init_loc_path = os.path.join(problem_dir, "init_loc.txt")
                        with open(init_loc_path, mode="w") as f:
                            random_loc = choose_random_room(graph)
                            print(random_loc)
                            f.write(random_loc)
                        
                        print("\n\n")
                    else:
                        print("Already exists")

