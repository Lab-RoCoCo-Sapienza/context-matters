import os
import datetime
from pathlib import Path
import csv
from workflow_delta import (
    generate_pddl_domain,
    prune_scene_graph,
    generate_pddl_problem,
    decompose_pddl_goal
)
from utils import (
    read_graph_from_path,
    copy_file,
    save_file
)

# Directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def run_delta(selected_dataset_splits):
    # Create a timestamp for the results folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR = os.path.join(BASE_DIR, "results", "delta", timestamp)
    
    # Load key from key.txt
    api_key = Path("key.txt").read_text().strip()

    # Create the results folder
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for task_dir_name in selected_dataset_splits:
        if not os.path.isdir(os.path.join(DATASET_DIR, task_dir_name)):
            continue

        os.makedirs(os.path.join(RESULTS_DIR, task_dir_name), exist_ok=True)

        # Create a CSV report file
        filename = task_dir_name + ".csv"
        with open(os.path.join(RESULTS_DIR, filename), mode="w") as f:
            f.write("Task|Scene|Problem|Planning Succesful|Grounding Successful|Plan Length|Relaxations|Refinements per iteration|Goal relaxations\n")

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
        
        copy_file(domain_file_path, os.path.join(RESULTS_DIR, task_dir_name, "domain.pddl"))
        
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
                scene_graph_file_path = os.path.join(dataset_problem_dir, scene_name + ".npz")

                # Create the log directory for the refinement step
                os.makedirs(os.path.join(results_problem_dir, "logs"), exist_ok=True)
                
                try:
                    # Run the pipeline
                    domain_pddl = generate_pddl_domain(goal_file_path)
                    pruned_sg = prune_scene_graph(scene_graph_file_path, description_file_path)
                    problem_pddl = generate_pddl_problem(scene_graph_file_path, description_file_path, domain_file_path)
                    sub_goals_pddl = decompose_pddl_goal(problem_pddl, domain_file_path)

                    # Save the final problem and plan
                    save_file(problem_pddl, os.path.join(results_problem_dir, "problem_final.pddl"))
                    save_file(sub_goals_pddl, os.path.join(results_problem_dir, "sub_goals_final.pddl"))

                    # Compute the length of the plan as the number of lines
                    plan_length = len(sub_goals_pddl.split("\n"))

                    # Save the results to the CSV file
                    with open(os.path.join(RESULTS_DIR, filename), mode="a", newline='') as f:
                        writer = csv.writer(f, delimiter='|')
                        writer.writerow([task_dir_name, scene_name, problem_id, True, True, plan_length, 0, "", ""])
                except Exception as e:
                    with open(os.path.join(RESULTS_DIR, filename), mode="a", newline='') as f:
                        writer = csv.writer(f, delimiter='|')
                        writer.writerow([task_dir_name, scene_name, problem_id, f"Exception: {str(e)}", "", "", "", ""])

if __name__ == "__main__":
    # RUN ON DATASET SPLIT #
    DATASET_SPLITS = [
        "dining_setup",
        "house_cleaning",
        "laundry"
    ]
    run_delta(DATASET_SPLITS)
