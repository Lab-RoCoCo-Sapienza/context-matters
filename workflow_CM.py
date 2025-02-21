import os
import hashlib
from pathlib import Path
from pprint import pprint

from planner import plan_with_output
from pddl_generation import generate_domain, generate_problem, refine_problem
from pddl_verification import translate_plan, VAL_validate, VAL_parse, VAL_ground, verify_groundability_in_scene_graph
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
    print_cyan,
    save_statistics
)


# Global variable to track the termination phase in case of an exception in the pipeline
CURRENT_PHASE = None


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
    GROUND_IN_SCENE_GRAPH = False,
    model = "gpt-4o"
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

    scene_graph_grounding_log = None
    pddlenv_error_log = None
    planner_error_log = None
    VAL_validation_log = None
    VAL_grounding_log = None
    planner_statistics = None

    iteration = 0 # Iteration counter of the outer loop
    refinements_per_iteration = [] # Output data: number of inner loop iterations per outer loop iteration
    goals = [current_goal] # Output data: all relaxed goals


    while iteration < WORKFLOW_ITERATIONS:
        VAL_grounding_succesful = False
        VAL_ground_succesful = False
        VAL_validation_succesful = False


        print_blue(f"\n##################################\n# Main loop iteration {iteration} out of {WORKFLOW_ITERATIONS} #\n##################################\n")

        # Create iteration directory
        iteration_dir = os.path.join(results_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        problem_dir = os.path.join(iteration_dir, "refinement_0")
        os.makedirs(problem_dir, exist_ok=True)

        problem_file_path = os.path.join(problem_dir, "problem.pddl")
        
        if domain_file_path is None:
            domain_file_path = os.path.join(iteration_dir, "generated_domain.pddl")

        logs_dir = os.path.join(iteration_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        

        if domain_description is not None:
            print_yellow("\n######################\n# GENERATING DOMAIN #\n######################")

            # Run the pipeline
            print_green("\n######################\n# GENERATING PDDL DOMAIN #\n######################")
            domain_pddl = generate_domain(
                domain_file_path=domain_file_path,
                goal_file_path=goal_file_path, 
                domain_description=domain_description, 
                logs_dir=logs_dir, 
                model=model
                )

            print_cyan("Domain VAL validation check...")
            # CHECK #1
            # Use VAL_parse to check if the domain is valid
            VAL_succesful, VAL_validation_log = VAL_validate(domain_file_path)
            if not VAL_succesful:
                print_red("VAL validation check failed. Reason: "+VAL_validation_log)
            else:
                print_cyan("VAL validation check passed.")

            # Store statistics at this stage
            CURRENT_PHASE = "DOMAIN_GENERATION"
            save_statistics(
                dir=results_dir,
                workflow_iteration = iteration, 
                phase = CURRENT_PHASE,
                VAL_validation_log=VAL_validation_log
            )
            
        print_yellow("\n######################\n# GENERATING PROBLEM #\n######################")
        
        
        problem = generate_problem(
            domain_file_path=domain_file_path, 
            initial_robot_location=initial_robot_location, 
            task=current_goal, 
            environment=extracted_scene_graph_str, 
            problem_file_path=problem_file_path,
            logs_dir=logs_dir,
            workflow_iteration=iteration,
            model=model,
            ADD_PREDICATE_UNDERSCORE_EXAMPLE=domain_description is not None
        )
        # Notice: the underscore example is only used when generating the domain from a description

        # CHECK #2
        print_cyan("Domain and Problem VAL validation check...")
        # Use VAL_parse to check if the domain is valid
        VAL_succesful, VAL_validation_log = VAL_validate(domain_file_path, problem_file_path)
        if not VAL_succesful:
            print_red("VAL validation check failed. Reason: "+VAL_validation_log)
        else:
            print_cyan("VAL validation check passed.")


        # PLANNING CHECK #

        plan_file_path = os.path.join(problem_dir, f"plan_0.out")
        planner_output_file_path = os.path.join(problem_dir, "logs", f"planner_output_0.log")

        plan, pddlenv_error_log, planner_error_log, planner_statistics = plan_with_output(domain_file_path, problem_dir, plan_file_path, search_flag='--search "astar(ff())"')      
        planning_succesful = plan is not None



        # CHECK #3
        VAL_ground_succesful = False
        # Use VAL to obtain feedback by first validating the refined domain and then grounding it
        # Translate the plan into a format parsable by VAL
        translated_plan_path = os.path.join(problem_dir, f"translated_plan_{iteration}.txt")
        translate_plan(plan_file_path, translated_plan_path)

        print_cyan("Plan VAL validation...")
        # Use VAL to validate the plan
        VAL_succesful, VAL_validation_log = VAL_validate(domain_file_path, problem_file_path, translated_plan_path)
        print_cyan(f"\tresult: {VAL_succesful} {VAL_validation_log}")

        if VAL_succesful:
            print_cyan("\nVAL grounding...")
            # Use VAL to ground the plan
            VAL_ground_succesful, VAL_grounding_log = VAL_ground(domain_file_path, problem_file_path)
            print_cyan(f"\tresult: {VAL_ground_succesful} {VAL_grounding_log}")  

        VAL_grounding_succesful = VAL_succesful and VAL_ground_succesful

        # Store statistics at this stage
        CURRENT_PHASE = "INITIAL_PLANNING"
        save_statistics(
            dir=results_dir,
            workflow_iteration = iteration, 
            plan_succesful=plan is not None,
            pddlenv_error_log=pddlenv_error_log,
            planner_error_log=planner_error_log,
            VAL_grounding_log=VAL_grounding_log,
            VAL_validation_log=VAL_validation_log,
            planner_statistics=planner_statistics,
            phase = CURRENT_PHASE
        )



        # PDDL REFINEMENT LOOP #

        PDDL_loop_iteration = 0
        if not planning_succesful or not VAL_grounding_succesful:
            
            print_magenta("\n\t#########################\n\t# PDDL REFINEMENT LOOP #\n\t#########################")
            CURRENT_PHASE = "PDDL_REFINEMENT"

            # Keep refining if planning or VAL validation or VAL grounding was not successful
            # AND
            # we have performed less than the maximum number of PDDL refinement iterations 
            while PDDL_loop_iteration < PDDL_GENERATION_ITERATIONS and\
                (not planning_succesful or not VAL_grounding_succesful):
                
                print_red(f"\tPlan validation problem:\n\t\tPlanning: {planning_succesful}\n\t\tVAL validation: {VAL_succesful}\n\t\tVAL grounding: {VAL_ground_succesful}\n")

                print_magenta(f"\n\t################################################\n\t# Problem refinement loop iteration {PDDL_loop_iteration} out of {PDDL_GENERATION_ITERATIONS} #\n\t################################################\n")
                print(f"\n\tProblem directory: {problem_dir}\n")
                

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
                    VAL_validation_log=VAL_validation_log,
                    VAL_grounding_log=VAL_grounding_log,
                    scene_graph_grounding_log=scene_graph_grounding_log
                )

                # Reset the scene graph grounding log
                scene_graph_grounding_log=None


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




                # PLANNING CHECK #

                plan_file_path = os.path.join(problem_dir, f"plan_{PDDL_loop_iteration}.out")
                planner_output_file_path = os.path.join(problem_dir, "logs", f"planner_output_{PDDL_loop_iteration}.log")
                
                # Attempt planning with refined problem
                # NOTE: here we use the "blind" heuristic which is slower but returns more explanatory outputs
                #plan, pddlenv_error_log, planner_error_log, planner_statistics = plan_with_output(domain_file_path, problem_dir, plan_file_path, search_flag='--search "astar(blind())"')    
                plan, pddlenv_error_log, planner_error_log, planner_statistics = plan_with_output(domain_file_path, problem_dir, plan_file_path)    
                planning_succesful = plan is not None


                # CHECK #4
                VAL_ground_succesful = False
                # Use VAL to obtain feedback by first validating the refined domain and then grounding it
                # Translate the plan into a format parsable by VAL
                translated_plan_path = os.path.join(problem_dir, f"translated_plan_{PDDL_loop_iteration}_{iteration}.txt")
                translate_plan(plan_file_path, translated_plan_path)

                print_cyan("\nVAL validation...")
                # Use VAL to validate the plan
                VAL_succesful, VAL_validation_log = VAL_validate(domain_file_path, problem_file_path, translated_plan_path)
                print_cyan(f"\tresult: {VAL_succesful} {VAL_validation_log}")

                if VAL_succesful:
                    print_cyan("\nVAL grounding...")
                    # Use VAL to ground the plan
                    VAL_ground_succesful, VAL_grounding_log = VAL_ground(domain_file_path, problem_file_path)
                    print_cyan(f"\tresult: {VAL_ground_succesful} {VAL_grounding_log}")  

                VAL_grounding_succesful = VAL_succesful and VAL_ground_succesful

                # Store statistics at this stage
                save_statistics(
                    dir=results_dir,
                    workflow_iteration = iteration, 
                    plan_succesful=plan is not None, 
                    pddlenv_error_log=pddlenv_error_log, 
                    planner_error_log=planner_error_log, 
                    planner_statistics=planner_statistics, 
                    phase = CURRENT_PHASE, 
                    pddl_refinement_iteration=PDDL_loop_iteration, 
                    VAL_validation_log=VAL_validation_log, 
                    VAL_grounding_log=VAL_grounding_log
                )

        else:
            planning_succesful = True
            pddlenv_error_log = None
            planner_error_log = None
            planner_statistics = None
            
        # Record the number of refinements for each iteration
        refinements_per_iteration.append(PDDL_loop_iteration)



        if planning_succesful and VAL_grounding_succesful:
            print_green(f"\n\tPlanning successful and VAL validation and VAL grounding succesful\n")
        else:
            print_red(f"\n\tOut of PDDL refinement iterations\n")





        # SCENE GRAPH GROUNDING #

        # If this experiment requires it, try grounding the plan in the real scene graph
        # ELSE skip this step
        if planning_succesful and plan and VAL_grounding_succesful:
        
            if GROUND_IN_SCENE_GRAPH:
                print_cyan("\nGrounding in scene graph started...")

                grounding_success_percentage, scene_graph_grounding_log = verify_groundability_in_scene_graph(
                    plan, 
                    extracted_scene_graph, 
                    domain_file_path=domain_file_path, 
                    problem_dir=problem_dir, 
                    move_action_str="move_to",
                    location_relation_str="at",
                    location_type_str="room",
                    initial_robot_location=initial_robot_location
                )
            
                print_cyan(f"Grounding result: {grounding_success_percentage} {scene_graph_grounding_log}")

                grounding_succesful = grounding_success_percentage == 1

                # Store statistics at this stage
                CURRENT_PHASE = "SCENE_GRAPH_GROUNDING"
                save_statistics(
                    dir=results_dir,
                    workflow_iteration = iteration, 
                    planner_statistics=planner_statistics, 
                    phase = CURRENT_PHASE, 
                    scene_graph_grounding_log=scene_graph_grounding_log,
                    grounding_success_percentage=grounding_success_percentage
                )

            else:
                print_cyan("\Skipping grounding in scene graph...")

                grounding_succesful = True
                scene_graph_grounding_log = None


        # If we achieved 100% grounding success, we can break the loop as we correctly achieved the original goal
        if grounding_succesful:
            print_green("\nPlan succesfully generated and grounded. Terminating workflow.") 
            scene_graph_grounding_log = None
            break
        else:
            # GOAL RELAXATION #

            print_red("\nGrounding not successful. Performing goal relaxation...") 
            
            # Find alternative objects
            alternatives, current_goal = dict_replaceable_objects(extracted_scene_graph_str, current_goal, workflow_iteration=iteration, logs_dir=logs_dir)
            #print(alternatives)
            #print(current_goal)

            # Generate new goal
            current_goal = relax_goal(extracted_scene_graph_str, current_goal)

            # Record relaxed goal
            goals.append(current_goal)

            # Prepare next iteration
            iteration += 1

    
    # Return the final problem and plan
    return problem_file_path, plan_file_path, current_goal, planning_succesful, grounding_succesful, iteration, refinements_per_iteration, goals, "", ""