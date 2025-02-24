import os

import pprint
from pathlib import Path
from dotenv import load_dotenv  # Fix import statement
import datetime
import json

from workflow_CM import run_pipeline_CM, CURRENT_PHASE
import csv
import traceback

from utils import (
    read_graph_from_path,
    get_objects, get_rooms,
    get_room_keypoints,
    copy_file, save_file,
    save_statistics
)

# Directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")




def run_CM(selected_dataset_splits, GENERATE_DOMAIN=False, GROUND_IN_SCENE_GRAPH=False, MODEL="gpt-4o", DETERMINE_PROBLEM_POSSIBILITY=False, PREVENT_IMPOSSIBLE_PROBLEMS=False):
    
    # Create a timestamp for the results folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Compose experiment_name
    experiment_name = f"CM_{MODEL}"

    if GENERATE_DOMAIN:
        experiment_name += "_gendomain"
    else:
        experiment_name += "_NOgendomain"
    
    if GROUND_IN_SCENE_GRAPH:
        experiment_name += "_groundsg"
    else:
        experiment_name += "_NOgroundsg"

    RESULTS_DIR = os.path.join(BASE_DIR, "results", experiment_name, timestamp)
    
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
        
        task_dir = os.path.join(DATASET_DIR, task_dir_name)
        
        if not os.path.isdir(task_dir):
            continue

        os.makedirs(os.path.join(RESULTS_DIR, task_dir_name), exist_ok=True)

        # Create a CSV report file
        filename = task_dir_name+".csv"
        with open(os.path.join(RESULTS_DIR, filename), mode="w") as f:
            f.write("Task|Scene|Problem|Planning Succesful|Grounding Successful|Plan Length|Relaxations|Refinements per iteration|Goal relaxations|Failure Stage|Failure Reason\n")

        print(task_dir_name)

        # Load the task JSON description
        task_file_path = os.path.join(DATASET_DIR, task_dir_name + ".json")
        with open(task_file_path) as f:
            task_description = json.load(f)
            

        
        # Look for a pddl file, that is the domain
        domain_file_path = None
        domain_description = None
        if GENERATE_DOMAIN:
            domain_description = task_description["domain"]
        else:
            for file in os.listdir(task_dir):
                if file.endswith(".pddl"):
                    domain_file_path = os.path.join(task_dir, file)
                    break

            if domain_file_path is None:
                raise Exception("No domain file found")
        
            copy_file(domain_file_path, os.path.join(RESULTS_DIR, task_dir_name, "domain.pddl"))
        
        assert domain_file_path is not None or domain_description is not None, "No domain file found"
        


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
                
                try:
                    # Run the pipeline
                    final_problem_file_path, final_plan_file_path, final_goal, planning_succesful, grounding_succesful, task_possible, possibility_explanation, n_relaxations, refinements_per_iteration, goal_relaxations, failure_stage, failure_reason = run_pipeline_CM(
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
                        PDDL_GENERATION_ITERATIONS=4,
                        domain_description = domain_description,
                        GROUND_IN_SCENE_GRAPH = GROUND_IN_SCENE_GRAPH,
                        model = MODEL,
                        PREVENT_IMPOSSIBLE_PROBLEMS = PREVENT_IMPOSSIBLE_PROBLEMS,
                        DETERMINE_PROBLEM_POSSIBILITY=DETERMINE_PROBLEM_POSSIBILITY
                    )

                    # Save the final problem and plan
                    if planning_succesful and grounding_succesful:
                        final_generated_problem = open(final_problem_file_path, "r").read()
                        save_file(final_generated_problem, os.path.join(results_problem_dir, "problem_final.pddl"))

                        final_generated_plan = open(final_plan_file_path, "r").read()
                        save_file(final_generated_plan, os.path.join(results_problem_dir, "plan_final.txt"))

                        # Compute the length of the plan as the number of lines
                        plan_length = len(final_generated_plan.split(", "))
                    else:
                        plan_length = 0

                    # Save the final relaxed goal
                    if n_relaxations > 0:
                        # Save the relaxed goal
                        save_file(final_goal, os.path.join(results_problem_dir, "task_final.txt"))

                    # Refinements per iteration
                    refinements_per_iteration_str = ";".join(map(str, refinements_per_iteration))

                    # Goal relaxations
                    goal_relaxations_str = "; ".join("'{}'".format(goal_relaxation.strip().replace('\n', ' ').replace('\r', '')) for goal_relaxation in goal_relaxations)

                    # Save the results to the CSV file
                    with open(os.path.join(RESULTS_DIR, filename), mode="a", newline='') as f:
                        writer = csv.writer(f, delimiter='|')
                        writer.writerow([task_dir_name, scene_name, problem_id, planning_succesful, grounding_succesful, plan_length, n_relaxations, refinements_per_iteration_str, goal_relaxations_str, failure_stage, failure_reason])
                    
                    # Save the possibility result if any, in a separate CSV
                    if DETERMINE_PROBLEM_POSSIBILITY:
                        with open(os.path.join(RESULTS_DIR, "possibility.csv"), mode="a", newline='') as f:
                            writer = csv.writer(f, delimiter='|')
                            writer.writerow([task_dir_name, scene_name, problem_id, task_possible, possibility_explanation.strip().replace('\n', ' ').replace('\r', '')])

                except Exception as e:
                    exception_str = str(e).strip().replace('\n', ' ').replace('\r', '')
                    with open(os.path.join(RESULTS_DIR, filename), mode="a", newline='') as f:
                        writer = csv.writer(f, delimiter='|')
                        writer.writerow([task_dir_name, scene_name, problem_id, f"Exception: {exception_str}","", "", "", "", "", "", ""])
                    
                    # Save the possibility result if any, in a separate CSV
                    if DETERMINE_PROBLEM_POSSIBILITY:
                        with open(os.path.join(RESULTS_DIR, "possibility.csv"), mode="a", newline='') as f:
                            writer = csv.writer(f, delimiter='|')
                            writer.writerow([task_dir_name, scene_name, problem_id, False, f"Exception: {exception_str}"])

                    # Write the exception traceback to error.txt in the logs directory
                    error_log_path = os.path.join(results_problem_dir, "logs", "error.txt")
                    with open(error_log_path, "w") as error_log_file:
                        traceback.print_exc(file=error_log_file)
                    
                    # Save the exception to statistics.json
                    save_statistics(
                        dir=results_problem_dir,
                        workflow_iteration=0,
                        phase=CURRENT_PHASE,
                        exception=e
                    )

if __name__=="__main__":
    
    DATASET_SPLITS = [
        "dining_setup",
        "house_cleaning",
        "laundry",
        "office_setup",
#        "office_setup_one_slot",
#        "other_1",
#        "other_2",
#        "pc_assembly"
#        "impossible_tasks"
    ]
    run_CM(DATASET_SPLITS, GENERATE_DOMAIN=False, GROUND_IN_SCENE_GRAPH=False, MODEL="gpt-4o", DETERMINE_PROBLEM_POSSIBILITY=True, PREVENT_IMPOSSIBLE_PROBLEMS=True)
    #run_CM(DATASET_SPLITS, GENERATE_DOMAIN=False, GROUND_IN_SCENE_GRAPH=True, MODEL="gpt-4o", PREVENT_IMPOSSIBLE_PROBLEMS=True)
    #run_CM(DATASET_SPLITS, GENERATE_DOMAIN=True, GROUND_IN_SCENE_GRAPH=False, MODEL="gpt-4o")
    #run_CM(DATASET_SPLITS, GENERATE_DOMAIN=True, GROUND_IN_SCENE_GRAPH=True, MODEL="gpt-4o")



    # RUN ON SINGLE TASK SCENE #
    #run_CM_for_task_scene("dining_setup", "Allensville", "problem_5")