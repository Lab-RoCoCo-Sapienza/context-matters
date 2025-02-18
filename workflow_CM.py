import os
import hashlib
from pathlib import Path
from pprint import pprint

from planner import plan_with_output
from pddl_generation import generate_problem, refine_problem
from goal_relaxation import relax_goal, dict_replaceable_objects

from utils import (
    load_planning_log,
    read_graph_from_path,
    compute_goal_similarity,
    print_to_planning_log,
    get_verbose_scene_graph,
    print_red,
    print_green,
    print_yellow,
    print_blue,
    print_magenta,
    print_cyan
)
from pddl_verification import verify_groundability_in_scene_graph, VAL_validate, VAL_parse, VAL_ground



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
    PDDL_GENERATION_ITERATIONS = 4,
    domain_description = None,
    PERFORM_GROUNDING = False
):

    # SETUP #
    print_blue("\n#########\n# SETUP #\n#########")
    scene_graph = read_graph_from_path(Path(scene_graph_file_path))
    extracted_scene_graph = get_verbose_scene_graph(scene_graph, as_string=False)
    extracted_scene_graph_str = get_verbose_scene_graph(scene_graph, as_string=True)
    current_goal = open(goal_file_path, "r").read()
    initial_robot_location = open(initial_location_file_path, "r").read()

    # Write the verbose scene graph to file
    with open(os.path.join(results_dir, "extracted_scene_graph.txt"), "w") as file:
        file.write(extracted_scene_graph_str)

        
    # Initialize workflow variables
    planning_succesful = False 
    grounding_succesful = False

    grounding_error_log = None
    pddlenv_error_log = None
    planner_error_log = None

    iteration = 0 # Iteration counter of the outer loop
    refinements_per_iteration = [] # Output data: number of inner loop iterations per outer loop iteration
    goals = [current_goal] # Output data: all relaxed goals


    while iteration < WORKFLOW_ITERATIONS:
        print_blue(f"\n##################################\n# Main loop iteration {iteration} out of {WORKFLOW_ITERATIONS} #\n##################################\n")

        # Create iteration directory
        iteration_dir = os.path.join(results_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        problem_dir = os.path.join(iteration_dir, "refinement_0")
        os.makedirs(problem_dir, exist_ok=True)

        problem_file_path = os.path.join(problem_dir, "problem.pddl")

        logs_dir = os.path.join(iteration_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        

        
        print_yellow("\n######################\n# GENERATING PROBLEM #\n######################")
        
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


        PDDL_loop_iteration = 0
        if plan is None:

            # PDDL REFINEMENT LOOP #
            print_magenta("\n\t#########################\n\t# PDDL REFINEMENT LOOP #\n\t#########################")

            while PDDL_loop_iteration < PDDL_GENERATION_ITERATIONS:
                print_magenta(f"\n\t################################################\n\t# Problem refinement loop iteration {PDDL_loop_iteration} out of {PDDL_GENERATION_ITERATIONS} #\n\t################################################\n")
                print(f"\n\tProblem directory: {problem_dir}\n")
                

                planning_succesful = plan is not None
                if planning_succesful:
                    print_green(f"\n\tPlanning successful\n")
                    break


                # PDDL PROBLEM REFINEMENT #
                print("\n\tRefining problem...")
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
                    planner_error_log=planner_error_log,
                    grounding_error_log=grounding_error_log
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
        else:
            planning_succesful = True
            pddlenv_error_log = None
            planner_error_log = None

        # Record the refinements for each iteration
        refinements_per_iteration.append(PDDL_loop_iteration)

        # GROUNDING #

        if plan is not None:       
            print_cyan("\nGrounding started...")
            grounding_success_percentage, grounding_error_log = verify_groundability_in_scene_graph(
                plan, 
                extracted_scene_graph, 
                domain_file_path=domain_file_path, 
                problem_dir=problem_dir, 
                move_action_str="move_to",
                location_relation_str="at",
                location_type_str="room",
                initial_robot_location=initial_robot_location
            )
            
            print_cyan(f"Grounding result: {grounding_success_percentage} {grounding_error_log}")

            grounding_succesful = grounding_success_percentage == 1

            # If we achieved 100% grounding success, we can break the loop as we correctly achieved the original goal
            if grounding_succesful:
                grounding_error_log = None
                break



        # GOAL RELAXATION #

        print_red("\nGrounding not successful. Performing goal relaxation...") 
        
        # Find alternative objects
        alternatives, current_goal = dict_replaceable_objects(extracted_scene_graph_str, current_goal, workflow_iteration=iteration, logs_dir=logs_dir)
        print(alternatives)
        print(current_goal)

        # Generate new goal
        current_goal = relax_goal(extracted_scene_graph_str, current_goal)


        # Record relaxed goal
        goals.append(current_goal)


        # Prepare next iteration
        iteration += 1

    
    # Return the final problem and plan
    return problem_file_path, plan_file_path, current_goal, planning_succesful, grounding_succesful, iteration, refinements_per_iteration, goals