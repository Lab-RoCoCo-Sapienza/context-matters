
import os
import hashlib
from pathlib import Path
from pprint import pprint

from planner import plan_with_output
from pddl_generation import generate_problem, refine_problem

from utils import (
    load_planning_log,
    read_graph_from_path,
    compute_goal_similarity,
    print_to_planning_log,
    get_verbose_scene_graph
)
from grounding import verify_groundability



def run_pipeline_CM(
    api_key,
    goal_file_path,
    initial_location_file_path,
    scene_graph_file_path,
    description_file_path,
    domain_file_path,
    scene_name,
    problem_id,
    results_dir,
    WORKFLOW_ITERATIONS = 4,
    PDDL_GENERATION_ITERATIONS = 4
):

    # SETUP #
    scene_graph = read_graph_from_path(Path(scene_graph_file_path))
    extracted_scene_graph = get_verbose_scene_graph(scene_graph, as_string=False)
    extracted_scene_graph_str = get_verbose_scene_graph(scene_graph, as_string=True)
    current_goal = open(goal_file_path, "r").read()
    initial_robot_location = open(initial_location_file_path, "r").read()

    # Write the verbose scene graph to file
    with open(os.path.join(results_dir, "extracted_scene_graph.txt"), "w") as file:
        file.write(extracted_scene_graph_str)

    planning_succesful = False
    iteration = 0
    while iteration < WORKFLOW_ITERATIONS:
        print(f"\n\n\n\n#########################\n# Main loop iteration {iteration} out of {WORKFLOW_ITERATIONS} #\n#########################\n\n")

        # Create iteration directory
        iteration_dir = os.path.join(results_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        problem_dir = os.path.join(iteration_dir, "refinement_0")
        os.makedirs(problem_dir, exist_ok=True)

        problem_file_path = os.path.join(problem_dir, "problem.pddl")

        logs_dir = os.path.join(iteration_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        

        
        print("\n\n\n#########################\nGENERATING PROBLEM\n#########################\n\n\n")
        
        problem = generate_problem(
            domain_file_path=domain_file_path, 
            initial_robot_location=initial_robot_location, 
            task=current_goal, 
            environment=extracted_scene_graph_str, 
            problem_file_path=problem_file_path,
            logs_dir=logs_dir,
            workflow_iteration=iteration,
        )

        plan_file_path = os.path.join(problem_dir, f"plan_0.out")
        planner_output_file_path = os.path.join(problem_dir, "logs", f"planner_output_0.log")

        plan, pddlenv_error_log, planner_error_log = plan_with_output(domain_file_path, problem_dir, plan_file_path)      



        if plan is None:

            # PDDL REFINEMENT LOOP #

            PDDL_loop_iteration = 0
            while PDDL_loop_iteration < PDDL_GENERATION_ITERATIONS:
                print(f"\n\n\t#######################\n\t# Problem refinement loop iteration {PDDL_loop_iteration} out of {PDDL_GENERATION_ITERATIONS} \n\t#########################\n\n")
                print(f"\n\n\tProblem directory: {problem_dir}\n\n")
                

                planning_succesful = plan is not None
                if planning_succesful:
                    print(f"\n\n\tPlanning successful\n\n")
                    break


                # PDDL PROBLEM REFINEMENT #
                print("\n\n\tRefining problem...")
                # Refine the problem and return the path to the refined version
                new_problem = refine_problem(
                    planner_output_file_path, 
                    domain_file_path, 
                    problem_file_path,
                    scene_graph=extracted_scene_graph_str, 
                    task=current_goal,
                    logs_dir=logs_dir,
                    workflow_iteration=iteration,
                    refinement_iteration=PDDL_loop_iteration,
                    pddlenv_error_log=pddlenv_error_log,
                    planner_error_log=planner_error_log
                )


                # New problem directory given by original problem directory where the final _N is replaced by _(N+1)
                new_problem_dir = problem_dir.split("/")[:-1] + [f"refinement_{PDDL_loop_iteration+1}"]
                new_problem_dir = "/"+os.path.join(*new_problem_dir)
                os.makedirs(new_problem_dir, exist_ok=True)

                # Save the new problem to a new file
                new_problem_file_path = os.path.join(new_problem_dir, "problem.pddl")

                # Save the new problem
                print(f"Saving new problem to {new_problem_file_path}")
                with open(new_problem_file_path, "w") as file:
                    file.write(new_problem)


                # Prepare next refinement iteration
                problem_dir = new_problem_dir
                problem_file_path = new_problem_file_path
                PDDL_loop_iteration += 1

                plan_file_path = os.path.join(problem_dir, f"plan_{PDDL_loop_iteration}.out")
                planner_output_file_path = os.path.join(problem_dir, "logs", f"planner_output_{PDDL_loop_iteration}.log")
                
                # Attempt planning with refined problem
                plan, pddlenv_error_log, planner_error_log = plan_with_output(domain_file_path, problem_dir, plan_file_path)      


        # GROUNDING #

        if plan is not None:       
            print("Grounding started...")
            grounding_success_percentage, failure_object, failure_room = verify_groundability(
                plan, 
                extracted_scene_graph, 
                domain_file_path=domain_file_path, 
                problem_dir=problem_dir, 
                move_action_str="move_to",
                location_relation_str="at",
                location_type_str="room",
                initial_robot_location=initial_robot_location
            )
            
            print(grounding_success_percentage, " ", failure_object, " ", failure_room)

            # If we achieved 100% grounding success, we can break the loop as we correctly achieved the original goal
            if grounding_success_percentage == 1:
                break


        # GOAL RELAXATION #

        print("Grounding not successful. Performing goal reasoning...")

        # Initialize the goal reasoner
        print("Creating GoalReasoner instance...")
        reasoner = GoalReasoner(api_key)
        
        # Load the file in planner/experiments/experiment_3/logs/pddl_ai.log
        planning_log = load_planning_log(os.path.join(results_dir,"logs","pddl_ai.log"))
        
        print("Performing goal reasoning...")
        # Reason about the goal
        action_modifications, goal_relaxation = reasoner.reason_about_goal(
            GOAL, 
            extracted_scene_graph_str,
            planning_log, 
            domain_file_path, 
            problem_file_path, 
            last_planning_succesful = planning_succesful, 
            output_dir = iteration_dir
        )       
                    
        #TODO: goal relaxation

        # Write the relaxed goal to goal_file_path, where we replace .txt with _final.txt
        final_goal_file_path = goal_file_path.replace(".txt", "_final.txt")
        with open(final_goal_file_path, "w") as file:
            file.write(new_goal)
            # Write similarity in the next row
            file.write(f"Similarity: \n{goal_similarity}")

        iteration += 1
        raise
    # Return the final problem and plan
    return problem_file_path, plan_file_path