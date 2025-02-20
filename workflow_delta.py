import os

from utils import *
from pathlib import Path
import json

from pprint import pprint


from agent import llm_call
from planner import plan_with_output
from pddlgym.core import PDDLEnv
from pddl_verification import verify_groundability_in_scene_graph, convert_JSON_to_locations_dictionary, VAL_validate, VAL_parse, VAL_ground, translate_plan

from pddl_generation import _save_prompt_response
from delta_prompts import generate_pddl_domain, prune_scene_graph, generate_pddl_problem, decompose_pddl_goal

# Global variable to track the termination phase in case of an exception in the pipeline
CURRENT_PHASE = None


def run_pipeline_delta(
    goal_file_path, 
    initial_location_file_path,
    scene_graph_file_path, 
    description_file_path, 
    task_name,
    scene_name,
    problem_id,
    results_dir,
    domain_file_path = None,
    domain_description = None,
    GROUND_IN_SCENE_GRAPH = False,
    model = "gpt-4o"
):

    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    initial_robot_location = open(initial_location_file_path, "r").read()


    if domain_file_path is None:
        
        CURRENT_PHASE = "DOMAIN_GENERATION"
        assert domain_description is not None, "Provide the domain or the domain description"

        # Run the pipeline
        print_green("\n######################\n# GENERATING PDDL DOMAIN #\n######################")
        domain_pddl = generate_pddl_domain(goal_file_path, domain_description, logs_dir=logs_dir, model=model)

        # Save the generated PDDL domain
        domain_file_path = os.path.join(results_dir, "domain.pddl")
        with open(domain_file_path, "w") as f:
            f.write(domain_pddl)
    

    CURRENT_PHASE = "DOMAIN_VALIDATION"
    print_cyan("VAL validation check...")
    # CHECK #1
    # Use VAL_parse to check if the domain is valid
    val_parse_success, val_parse_log = VAL_validate(domain_file_path)
    
    save_statistics(
        dir=results_dir,
        workflow_iteration=0,
        phase=CURRENT_PHASE,
        VAL_validation_log=val_parse_log
    )

    if not val_parse_success:
        return domain_file_path, None, None, None, False, False, None, "DOMAIN_GENERATION", val_parse_log
    print_cyan("VAL validation check passed.")



    print_cyan("\n######################\n# PRUNING SCENE GRAPH #\n######################")
    CURRENT_PHASE = "PRUNING_SCENE_GRAPH"
    pruned_sg = prune_scene_graph(scene_graph_file_path, goal_file_path, initial_robot_location, logs_dir=logs_dir, model=model)

    # Save the pruned scene graph
    pruned_sg_path_readable = os.path.join(results_dir, "pruned_sg.txt")
    with open(pruned_sg_path_readable, "w") as f:
        f.write(json.dumps(pruned_sg, indent=4))

    pruned_sg_path_npz = os.path.join(results_dir, "pruned_sg.npz")
    save_graph(pruned_sg, pruned_sg_path_npz)

    

    print_yellow("\n######################\n# GENERATING PDDL PROBLEM #\n######################")
    CURRENT_PHASE = "PROBLEM_GENERATION"
    problem_pddl = generate_pddl_problem(pruned_sg_path_npz, goal_file_path, domain_file_path, initial_robot_location, logs_dir=logs_dir, model=model)
    
    # Save the generated PDDL problem
    problem_pddl_path = os.path.join(results_dir, "problem.pddl")
    with open(problem_pddl_path, "w") as f:
        f.write(problem_pddl)



    print_cyan("VAL validation check...")
    # CHECK #2
    # Use VAL_parse to check if the problem is valid
    val_parse_success, val_parse_log = VAL_validate(domain_file_path, problem_pddl_path)

    save_statistics(
        dir=results_dir,
        workflow_iteration=0,
        phase=CURRENT_PHASE,
        VAL_validation_log=val_parse_log
    )

    if not val_parse_success:
        return domain_file_path, pruned_sg, problem_pddl_path, None, False, False, None, "PROBLEM_GENERATION", val_parse_log
    print_cyan("VAL validation check passed.")



    print_magenta("\n######################\n# GENERATING PDDL SUB-GOALS #\n######################")
    CURRENT_PHASE = f"SUBGOAL_GENERATION"
    sub_goals_pddl = decompose_pddl_goal(problem_pddl_path, domain_file_path, initial_robot_location, logs_dir=logs_dir, model=model)


    print_blue("\n######################\n# PLANNING AND GROUNDING AUTOREGRESSIVE #\n######################")
    CURRENT_PHASE = f"PLANNING"
    
    old_pddl_env = None
    # Save sub-goals to files
    plans = []
    sub_goals_file_paths = []
    for i, sub_goal in enumerate(sub_goals_pddl):

        sub_goal_dir = os.path.join(results_dir, f"sub_goal_{i+1}")
        os.makedirs(sub_goal_dir, exist_ok=True)
        sub_goal_file = os.path.join(sub_goal_dir, f"sub_goal_{i+1}.pddl")
        sub_goals_file_paths.append(sub_goal_file)
        with open(sub_goal_file, "w") as f:
            f.write(sub_goal)
    
    
    # Evaluate the sequence of sub-goals
    # 1) Replace the :object and :init sections of each subgoal with the goal state of the prev subgoal (or the initial state of the non-decomposed problem)
    # 2) Generate a plan
    # 3) [OPTIONAL] Ground the plan in the scene graph
    for i, sub_goal in enumerate(sub_goals_pddl):
        CURRENT_PHASE = f"SUBGOAL_{i+1}:GENERATION"

        grounding_succesful = False
        planning_succesful = False

        sub_goal_dir = os.path.join(results_dir, f"sub_goal_{i+1}")
        sub_goal_file = os.path.join(sub_goal_dir, f"sub_goal_{i+1}.pddl")


        # If this is the first iteration (old_pddl_env is None), use the PDDLEnv of the main (non-decomposed problem) to get the initial state of the first subgoal
        if old_pddl_env is None:
            print(domain_file_path)
            print(results_dir)
            old_pddl_env = PDDLEnv(domain_file_path, results_dir, operators_as_actions=True)
            old_pddl_env.reset()


        # Get the final state of the previous environment
        obs = old_pddl_env.get_state()

        # Extract PDDL objects
        pddl_objects = []
        pddl_objects_str = "(:objects\n"
        for obj in obs.objects:
            obj = str(obj).split(":")
            pddl_objects.append((obj[0], obj[1]))
            pddl_objects_str += f"    {obj[0]} - {obj[1]}\n"
        pddl_objects_str += ")\n\n"

        #print(pddl_objects_str)
        

        # Extract PDDL predicates        
        state = obs.literals

        pddl_predicates = []
        pddl_predicates_str = "(:init\n"
        for literal in state:
            predicate_name = literal.predicate.name
            predicate_variables = []
            for variable in literal.variables:
                predicate_variables.append(variable.split(":")[0])
            pddl_predicates_str += f"    ({predicate_name} {' '.join(predicate_variables)})\n"
            pddl_predicates.append((predicate_name, *predicate_variables))
        pddl_predicates_str += ")\n\n"

        #print(pddl_predicates_str)
        

        # In the sub-goal problem file, replace the :objects and :init sections with the goal state of the previous sub-goal
        sub_goal_pddl = open(sub_goal_file, "r").read()
        
        # Replace the whole of the :objects and :init sections of the sub-goal with the goal state of the previous sub-goal
        sub_goal_pddl = sub_goal_pddl.replace(sub_goal_pddl[sub_goal_pddl.index("(:objects"):sub_goal_pddl.index("(:init")], pddl_objects_str)
        sub_goal_pddl = sub_goal_pddl.replace(sub_goal_pddl[sub_goal_pddl.index("(:init"):sub_goal_pddl.index("(:goal")], pddl_predicates_str)

        #print(sub_goal_pddl)

        # Save the modified sub-goal
        with open(sub_goal_file, "w") as f:
            f.write(sub_goal_pddl)




        CURRENT_PHASE = f"SUBGOAL_{i+1}:PLANNING"
        # Initialize env
        new_pddl_env = PDDLEnv(domain_file_path, sub_goal_dir, operators_as_actions=True)


        output_plan_file_path = os.path.join(sub_goal_dir, f"plan_{i+1}.txt")

        # Compute plan on env
        plan, pddlenv_error_log, planner_error_log, planner_statistics = plan_with_output(domain_file_path, sub_goal_dir, output_plan_file_path, env=new_pddl_env)

        if plan is not None and plan:       

            planning_succesful = True
            
            # Verify with VAL
            translated_plan_path = os.path.join(sub_goal_dir, f"translated_plan_{i+1}.txt")

            # Translate the plan into a format parsable by VAL
            translate_plan(output_plan_file_path, translated_plan_path)

            print_cyan("\nVAL validation...")
            CURRENT_PHASE = f"SUBGOAL_{i+1}:VAL:VALIDATION"
            # Use VAL to validate the plan
            val_succesful, val_log = VAL_validate(domain_file_path, sub_goal_file, translated_plan_path)
            print_cyan(f"\tresult: {val_succesful} {val_log}")

            print_cyan("\nVAL grounding...")
            # Use VAL to ground the plan
            val_ground_succesful, val_ground_log = VAL_ground(domain_file_path, sub_goal_file)
            print_cyan(f"\tresult: {val_ground_succesful} {val_ground_log}")

            grounding_succesful = val_succesful and val_ground_succesful

            CURRENT_PHASE = f"SUBGOAL_{i+1}:VAL:GROUNDING"
            save_statistics(
                dir=results_dir,
                workflow_iteration=0,
                phase=CURRENT_PHASE,
                plan_succesful=planning_succesful,
                pddlenv_error_log=pddlenv_error_log,
                planner_error_log=planner_error_log,
                VAL_validation_log=val_log,
                VAL_grounding_log=val_ground_log,
                planner_statistics=planner_statistics
            )


            # If this experiment requires it, try grounding the plan in the real scene graph
            if grounding_succesful and GROUND_IN_SCENE_GRAPH:
                CURRENT_PHASE = f"SUBGOAL_{i+1}:SCENE_GRAPH:GROUNDING"
                print_cyan("\nGrounding in scene graph started...")

                # Convert the scene graph into a format readable by the grounder
                extracted_locations_dictionary = convert_JSON_to_locations_dictionary(pruned_sg)

                # Save the extracted locations dictionary to file
                extracted_locations_dictionary_file_path = os.path.join(sub_goal_dir, "extracted_locations_dictionary.json")
                with open(extracted_locations_dictionary_file_path, "w") as f:
                    json.dump(extracted_locations_dictionary, f, indent=4)

                grounding_success_percentage, grounding_error_log = verify_groundability_in_scene_graph(
                    plan, 
                    None, 
                    domain_file_path=domain_file_path, 
                    problem_dir=sub_goal_dir, 
                    move_action_str="move_to",
                    location_relation_str="at",
                    location_type_str="room",
                    initial_robot_location=initial_robot_location,
                    pddlgym_environment = new_pddl_env,
                    locations_dictionary = extracted_locations_dictionary
                )
                
                print_cyan(f"Grounding result: {grounding_success_percentage} {grounding_error_log}")

                grounding_succesful = grounding_success_percentage == 1

                save_statistics(
                    dir=results_dir,
                    workflow_iteration=0,
                    phase=CURRENT_PHASE,
                    plan_succesful=planning_succesful,
                    pddlenv_error_log=pddlenv_error_log,
                    planner_error_log=planner_error_log,
                    VAL_validation_log=val_log,
                    VAL_grounding_log=val_ground_log,
                    grounding_success_percentage=grounding_success_percentage,
                    scene_graph_grounding_log=grounding_error_log
                )

            # If we achieved 100% grounding success, we can break the loop as we correctly achieved the original goal
            if not grounding_succesful:
                plans.append(plan)
                return domain_file_path, pruned_sg, problem_pddl_path, sub_goals_file_paths, True, False, plans, CURRENT_PHASE, grounding_error_log
                break
        else:
            error_log = pddlenv_error_log if pddlenv_error_log is not None else planner_error_log

            save_statistics(
                dir=results_dir,
                workflow_iteration=0,
                phase=CURRENT_PHASE,
                plan_succesful=planning_succesful,
                pddlenv_error_log=pddlenv_error_log,
                planner_error_log=planner_error_log
            )
            
            return domain_file_path, pruned_sg, problem_pddl_path, sub_goals_file_paths, False, False, plans, "PLANNING:SUBGOAL_"+str(i), error_log
                

        plans.append(plan)

        new_pddl_env = old_pddl_env

        print_cyan(f"Sub-goal {i+1} completed.\n---")

    return domain_file_path, pruned_sg, problem_pddl_path, sub_goals_file_paths, planning_succesful, grounding_succesful, plans, "", ""