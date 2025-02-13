import os

import pprint
from pathlib import Path
from dotenv import load_dotenv  # Fix import statement

from workflow import run_pipeline_CM

from utils import (
    read_graph_from_path,
    get_objects, get_rooms,
    get_room_keypoints,
    copy_file, save_file
)

# Directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation")

def run_CM(selected_dataset_splits):
    
    # Load key from key.txt
    with open("key.txt", "r") as f:
        api_key = f.read().strip()   


    # The results folder will follow closely the dataset folder structure 
    # - task_N
    #     ...
    #     - scene_N
    #           - scene_graph.npz
    #           - description.txt
    #           - task.txt
    #           - init_loc.txt
    # 
    # The domain.pddl is instead linked to the scene graph

    # Create the results folder
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for task_dir_name in selected_dataset_splits:
        if not os.path.isdir(os.path.join(DATASET_DIR, task_dir_name)):
            continue

        os.makedirs(os.path.join(RESULTS_DIR, task_dir_name), exist_ok=True)

        # Create a CSV report file
        filename = task_dir_name+".csv"
        with open(os.path.join(RESULTS_DIR, filename), mode="w") as f:
            f.write("Task,Scene,Problem,Success\n")

        print(task_dir_name)
        # Look for a pddl file, that is the domain
        domain_file_path = None
        task_dir = os.path.join(DATASET_DIR, task_dir_name)
        for file in os.listdir(task_dir):
            if file.endswith(".pddl"):
                domain_file_path = os.path.join(task_dir, file)
                break
        
        if domain_file_path is None:
            raise Exception("No domain file found")
        
        copy_file(domain_file_path, os.path.join(RESULTS_DIR, task_dir, "domain.pddl"))
        
        for scene_name in os.listdir(task_dir):

            dataset_scene_dir = os.path.join(task_dir, scene_name)
            if not os.path.isdir(dataset_scene_dir):
                continue

            
            for problem_id in os.listdir(dataset_scene_dir):

                dataset_problem_dir = os.path.join(dataset_scene_dir, problem_id)
                if not os.path.isdir(dataset_problem_dir) or not problem_id.startswith("problem"):
                    continue
             
                results_problem_dir = os.path.join(RESULTS_DIR, task_dir_name, scene_name, problem_id)
                print(results_problem_dir)
            
                # Create the problem directory
                os.makedirs(results_problem_dir, exist_ok=True)
            
                # Copy the goal file
                goal_file_path = os.path.join(dataset_problem_dir, "task.txt")
                copy_file(goal_file_path, os.path.join(results_problem_dir, "task.txt"))      

                # Copy the initial location file
                initial_location_file_path = os.path.join(dataset_problem_dir, "init_loc.txt")
                copy_file(initial_location_file_path, os.path.join(results_problem_dir, "init_loc.txt"))

                # Copy the description file
                description_file_path = os.path.join(dataset_problem_dir, "description.txt")   
                copy_file(description_file_path, os.path.join(results_problem_dir, "description.txt"))   

                # Copy the scene graph file
                scene_graph_file_path = os.path.join(dataset_problem_dir, scene_name+".npz")

                # Create the log directory for the refinement step
                os.makedirs(os.path.join(results_problem_dir, "logs"), exist_ok=True)
                    
                # Run the pipeline
                final_problem_file_path, final_plan_file_path, success, n_relaxations, refinements_per_iteration = run_pipeline_CM(
                    api_key,
                    goal_file_path = goal_file_path,
                    initial_location_file_path = initial_location_file_path,
                    scene_graph_file_path = scene_graph_file_path,
                    description_file_path = description_file_path,
                    domain_file_path=domain_file_path,
                    scene_name = scene_name,
                    problem_id = problem_id,
                    results_dir=results_problem_dir,
                    WORKFLOW_ITERATIONS = 4,
                    PDDL_GENERATION_ITERATIONS=2
                )


                # Save the final problem and plan
                final_generated_problem = open(final_problem_file_path, "r").read()
                save_file(final_generated_problem, os.path.join(task_variation_dir, "problem_final.pddl"))

                final_generated_plan = open(final_plan_file_path, "r").read()
                save_file(final_generated_plan, os.path.join(task_variation_dir, "plan_final.txt"))

                # Compute the length of the plan as the number of lines
                plan_length = len(final_generated_plan.split("\n"))


                # Refinements per iteration
                for refinements_n in refinements_per_iteration:
                    refinements_per_iteration_str += f"{refinements_n}-"

                # Save the results to the CSV file
                with open(os.path.join(RESULTS_DIR, filename), mode="a") as f:
                    f.writeln(f"{task_dir},{scene_name},{problem_id},{success},{plan_length},{n_relaxations},{refinements_per_iteration_str}\n")

if __name__=="__main__":
    
    DATASET_SPLITS = [
        "dining_setup",
        "house_cleaning",
        "laundry",
        "office_setup",
        "other_1",
        "other_2"
    ]

    run_CM(DATASET_SPLITS)