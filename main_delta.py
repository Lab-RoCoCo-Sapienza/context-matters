import os
import datetime
from pathlib import Path
import csv
import traceback
import json


from workflow_delta import run_pipeline_delta
from utils import (
    read_graph_from_path,
    copy_file,
    save_file,
    print_blue,
    print_green,
    print_red
)


# Directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def run_delta(selected_dataset_splits, GENERATE_DOMAIN = True, GROUND_IN_SCENE_GRAPH = False, MODEL = "gpt-4o"):
    print_blue("Running delta...")
    # Create a timestamp for the results folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Compose experiment_name
    experiment_name = f"delta_{MODEL}"

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
            f.write("Task|Scene|Problem|Planning Succesful|Grounding Successful|Plan Length|Number of subgoals|Failure stage|Failure reason\n")

        task_dir = os.path.join(DATASET_DIR, task_dir_name)

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

            #print_green(f"Processing scene: {scene_name}")

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
                

                print_blue(f"\n\n\n------------------\n\nRunning pipeline delta on task {task_dir_name}, scene {scene_name}, problem {problem_id}")

                print_blue("#########\n# SETUP #\n#########")
                planning_succesful = False
                grounding_succesful = False
                subplans = []
                failure_stage = None
                failure_reason = None
                final_generated_plan = None
                try:
                    final_domain_file_path, final_pruned_scene_graph, final_problem_file_path, final_subgoals_file_paths, planning_succesful, grounding_succesful, subplans, failure_stage, failure_reason = run_pipeline_delta(
                        goal_file_path, 
                        initial_location_file_path,
                        scene_graph_file_path, 
                        description_file_path, 
                        task_name = task_dir_name,
                        scene_name = scene_name,
                        problem_id = problem_id,
                        results_dir=results_problem_dir,
                        domain_file_path = domain_file_path,
                        domain_description = domain_description,
                        GROUND_IN_SCENE_GRAPH = GROUND_IN_SCENE_GRAPH,
                        model = MODEL
                    )


                    # Save the final problem and plan
                    if planning_succesful and grounding_succesful:
                        final_plan_file_path = os.path.join(results_problem_dir, "plan_final.txt")
                        # Concatenate all subplans and write the resulting plan
                        with open(final_plan_file_path, "w") as f:
                            for subplan in subplans:
                                f.write(subplan + "\n")
                        
                        # Compute the length of the plan as the number of lines
                        plan_length = len(final_generated_plan.split(", "))

                        # Write final generated domain
                        final_generated_domain = open(final_domain_file_path, "r").read()
                        save_file(final_generated_domain, os.path.join(results_problem_dir, "domain_final.pddl"))
                        
                        # Save the results to the CSV file
                        with open(os.path.join(RESULTS_DIR, filename), mode="a", newline='') as f:
                            writer = csv.writer(f, delimiter='|')
                            writer.writerow([task_dir_name, scene_name, problem_id, planning_succesful, grounding_succesful, plan_length, len(subplans), "", ""])
                    else:
                        with open(os.path.join(RESULTS_DIR, filename), mode="a", newline='') as f:
                            writer = csv.writer(f, delimiter='|')
                            writer.writerow([task_dir_name, scene_name, problem_id, planning_succesful, grounding_succesful, "", "", str(failure_stage).replace('\n', ' ').replace('\r', ''), str(failure_reason).replace('\n', ' ').replace('\r', '')])

                    if planning_succesful and grounding_succesful:
                        print_green("Pipeline delta completed successfully. Planning succesful: " + str(planning_succesful) + " Grounding succesful: " + str(grounding_succesful))
                    else:
                        print_red("Pipeline delta completed with issues. Planning succesful: " + str(planning_succesful) + " Grounding succesful: " + str(grounding_succesful))
                
                except Exception as e:
                    print_red(f"Exception occurred: {str(e)}")
                    traceback.print_exc()
                    exception_str = str(e).strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
                    with open(os.path.join(RESULTS_DIR, filename), mode="a", newline='') as f:
                        writer = csv.writer(f, delimiter='|')
                        writer.writerow([task_dir_name, scene_name, problem_id, planning_succesful, grounding_succesful, "", "", "Exception", exception_str])
                    
                    break

if __name__ == "__main__":
    print_blue("Starting main execution...")
    # RUN ON DATASET SPLIT #
    DATASET_SPLITS = [
        "dining_setup",
        "house_cleaning",
        "laundry",
#        "office_setup",
#        "other_1",
#        "other_2",
        "pc_assembly"
    ]
    run_delta(DATASET_SPLITS, GENERATE_DOMAIN = True, GROUND_IN_SCENE_GRAPH = False, MODEL = "gpt-4o")
