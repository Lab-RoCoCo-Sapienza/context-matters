import os

import pprint
from pathlib import Path
from dotenv import load_dotenv  # Fix import statement

from evaluation_utils import load_dataset, run_pipeline
from planner_pddl.pddl_generation import PDDLGenerator

from utils import (
    read_graph_from_path,
    get_objects, get_rooms,
    get_room_keypoints,
    copy_file, save_file
)

# Directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "evaluation", "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")


# Example usage
if __name__ == "__main__":
    # Load environment variables from the correct location
    load_dotenv("environment.env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")


    test_dataset = load_dataset(EVALUATION_BASELINE)


    # The results folder will follow closely the dataset folder structure 
    # - task_N
    #   - task_N_M
    #     - domain_initial.pddl
    #     - problem_initial.pddl
    #     - goal.txt
    #     - situation.txt

    # Create the results folder
    RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for task_name in test_dataset.keys():
        for task_variation in test_dataset[task_name].keys():
            task_dir = os.path.join(RESULTS_DIR, task_name)
            task_variation_dir = os.path.join(task_dir, task_variation)
            os.makedirs(task_variation_dir, exist_ok=True)


            # Copy the experiment data to the results folder #

            # Copy the domain file
            domain_file_path = os.path.join(task_variation_dir, "domain_initial.pddl")
            copy_file(test_dataset[task_name][task_variation]["domain"], domain_file_path)

            # Copy the problem file
            problem_file_path = os.path.join(task_variation_dir, "problem_initial.pddl")
            copy_file(test_dataset[task_name][task_variation]["problem"], problem_file_path)

            # Copy the goal file
            goal_file_path = os.path.join(task_variation_dir, "goal.txt")
            copy_file(test_dataset[task_name][task_variation]["goal"], goal_file_path)            

            # Copy the situation file
            situation_file_path = os.path.join(task_variation_dir, "situation.txt")
            copy_file(test_dataset[task_name][task_variation]["situation"], situation_file_path)


            # TODO: assign initial location
            initial_robot_location = None
                
            # Run the pipeline
            final_problem_file_path, final_plan_file_path = run_pipeline(
                domain_file_path, 
                goal_file_path, 
                task_variation_dir, 
                initial_robot_location,
                api_key,
                initial_problem_file_path = problem_file_path,
                situation_file_path = situation_file_path,
                WORKFLOW_ITERATIONS = 4,
                PDDL_GENERATION_ITERATIONS=2, 
                USE_SITUATION=True,
                PERFORM_GOAL_REASONING = True,
                PERFORM_GROUNDING=True
            )

            # Save the final problem and plan
            final_problem = open(final_problem_file_path, "r").read()
            save_file(final_problem, os.path.join(task_variation_dir, "problem_final.pddl"))

            final_plan = open(final_plan_file_path, "r").read()
            save_file(final_plan, os.path.join(task_variation_dir, "plan.txt"))
