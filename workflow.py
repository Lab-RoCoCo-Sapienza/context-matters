
import os
import hashlib
from pathlib import Path

from sentence_transformers import SentenceTransformer
from planner_pddl.pddl_generation import PDDLGenerator
from planner_pddl.planner import PlannerWrapper
from knowledge_graph import KnowledgeGraph
from goal_generation import GoalReasoner

from utils import generate_scene_graph

from utils import (
    load_planning_log,
    read_graph_from_path,
    copy_file, compute_goal_similarity,
    print_to_planning_log,
    get_verbose_scene_graph
)
from ground import perform_grounding

# Constants

ALWAYS_REGENERATE_KNOWLEDGE_GRAPH = False
PERFORM_GROUNDING = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "evaluation", "dataset")
TASKOGRAPHY_DIR = os.path.join(BASE_DIR, "evaluation", "taskography","problems")
SCENE_GRAPH_DIR = os.path.join(BASE_DIR, "3dscenegraph")

#COWP_GRAPH = SCENE_GRAPH_DIR + "/3DSceneGraph_Allensville.npz"

model =  SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Load all tasks in the dataset_dir, which should have structure
# - task_N
#   - task_N_M
#     - domain_initial.pddl
#     - problem_initial.pddl
#     - goal.txt
#     - situation.txt
# where task_N is related to a specific goal and task_N_M is a combination of that goal plus a random "situation"
# The output will be a dictionary, where we store the task and for each the situation-tasks, for each of these the domain and problem file paths, the goal string and the situation string

def load_dataset_cowp(dataset_dir):
    tasks = {}
    for task_dir in os.listdir(dataset_dir):
        task_path = os.path.join(dataset_dir, task_dir)
        if not os.path.isdir(task_path):
            continue
        tasks[task_dir] = {}
        for situation_dir in os.listdir(task_path):
            situation_path = os.path.join(task_path, situation_dir)
            if not os.path.isdir(situation_path):
                continue
            tasks[task_dir][situation_dir] = {}
            tasks[task_dir][situation_dir]['domain'] = os.path.join(situation_path, 'domain_initial.pddl')
            tasks[task_dir][situation_dir]['problem'] = os.path.join(situation_path, 'problem_initial.pddl')
            tasks[task_dir][situation_dir]['goal'] = os.path.join(situation_path, 'goal.txt')
            tasks[task_dir][situation_dir]['situation'] = os.path.join(situation_path, 'situation.txt')
    return tasks

def load_dataset_taskography(dataset_dir):
    tasks = {}
    for task_dir in os.listdir(dataset_dir):
        scene_path = os.path.join(dataset_dir, task_dir)
        for experiment in os.listdir(scene_path):
            experiment_path = os.path.join(scene_path, experiment)
            domain_path = os.path.join(experiment_path, "domain.pddl")
            problem_path = os.path.join(experiment_path, "problem_gt.pddl")
            goal_path = os.path.join(experiment_path, "goal.txt")
            scene_graph = os.path.join(SCENE_GRAPH_DIR , scene_path.split("/")[-1] + ".npz")
            tasks[experiment] = {
                "domain": domain_path,
                "problem": problem_path,
                "goal": goal_path,
                "scene_graph": scene_graph
            }
    return tasks

def load_dataset(evaluation_baseline):
    if evaluation_baseline == "taskography":
        return load_dataset_taskography(TASKOGRAPHY_DIR)
    else:
        return load_dataset_cowp(DATASET_DIR)



def run_pipeline(
    initial_domain_file_path, 
    goal_file_path,
    results_dir, 
    initial_robot_location,
    api_key,
    initial_problem_file_path = None,
    situation_file_path = None,
    scene_graph_path = None,
    WORKFLOW_ITERATIONS = 4,
    PDDL_GENERATION_ITERATIONS = 2,
    PERFORM_GOAL_REASONING = True,
    USE_SITUATION = False,
    PERFORM_GROUNDING = False
):
    


    ########
    # GOAL #
    ########
    
    # Load the goal
    GOAL = open(goal_file_path, "r").read()

    if USE_SITUATION:
        # Load the situation
        SITUATION = open(situation_file_path, "r").read()
        GOAL = f"{GOAL}. {SITUATION}"

    # Load goal and situation from path
    current_goal = GOAL



    # Extract scene graph
    ###############
    # SCENE GRAPH #
    ###############

    print("Setting up experiment...")
    if scene_graph_path is not None:
        print("Extracting scene graph...from " + scene_graph_path)
        path_graph = Path(scene_graph_path)
        scene_graph = read_graph_from_path(path_graph)
        scene_graph_extracted = get_verbose_scene_graph(scene_graph)
    else:
        print("Generating scene graph...")
        scene_graph_extracted = generate_scene_graph(current_goal, initial_problem_file_path, api_key)
        


    # Perceivable subgraph extraction #

    #print("Subgraph based on the perception of the agent")
    # Subgraph based on the perception of the agent
    #subgraph = get_room_objects_from_pose(scene_graph, keypoints)



    # Step 2)
    #########
    # SETUP #
    #########

    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    planning_log_file_path = os.path.join(results_dir, "logs", "pddl_ai.log")
    
    print_to_planning_log(planning_log_file_path, f"Task: {GOAL}")

    pddl_generator = PDDLGenerator(GOAL, results_dir, api_key = api_key)
    planner = PlannerWrapper(results_dir, api_key = api_key)


    planning_succesful = False
    iteration = 0




    #############
    # MAIN LOOP #
    #############
    # This loop terminates when ???


    while iteration < WORKFLOW_ITERATIONS:
        print(f"\n\n\n\n#########################\n# Main loop iteration {iteration} out of {WORKFLOW_ITERATIONS} #\n#########################\n\n")

        # Create iteration directory
        iteration_dir = os.path.join(results_dir, "iterations", f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        os.makedirs(os.path.join(iteration_dir, "problems"), exist_ok=True)
        
        problem_dir = os.path.join(iteration_dir, "problems", "problem_0")
        os.makedirs(problem_dir, exist_ok=True)
        # Copy the initial problem file path
        problem_file_path = os.path.join(problem_dir, "problem.pddl")
        domain_file_path = initial_domain_file_path

        print(problem_file_path)

        pddl_dir = os.path.join(iteration_dir,"problems")

        # If we are generating problems from scratch, we need to generate the first problem
        # Else we copy the existing one from the dataset
        #if initial_problem_file_path is None:

        #    ###################
        #    # PDDL GENERATION #
        #    ###################
        #    
        #    print("\n\n\n#########################\nGENERATING PROBLEM\n#########################\n\n\n")
        #    
        #    problem = pddl_generator.generate_problem(
        #        domain_file_path=domain_file_path, 
        #        initial_robot_location=initial_robot_location, 
        #        task=current_goal, 
        #        environment=scene_graph_extracted, 
        #        problem_file_path=problem_file_path,
        #        output_dir=results_dir
        #    )
        #    
        #    if "<NOT_ENOUGH_INFORMATION>" in problem:
        #        print("Not enough objects in the scene")
        #        print("Not enough objects in the scene; problem generation failed")
        #        print(f"Problem: {problem}")
        #        break
        #    else:
        #        print("Problem generation succesful")
        #        print(f"Problem: {problem}")
        #        pddl_generator_succesful = True
        #else:
        #    copy_file(initial_problem_file_path, problem_file_path)



        ###################
        # PDDL GENERATION #
        ###################
        
        print("\n\n\n#########################\nGENERATING PROBLEM\n#########################\n\n\n")
        
        problem = pddl_generator.generate_problem(
            domain_file_path=domain_file_path, 
            initial_robot_location=None, 
            task=current_goal, 
            environment=scene_graph_extracted, 
            problem_file_path=problem_file_path,
            output_dir=results_dir
        )
        
        if "<NOT_ENOUGH_INFORMATION>" in problem:
            print("Not enough objects in the scene")
            print("Not enough objects in the scene; problem generation failed")
            print(f"Problem: {problem}")
            break
        else:
            print("Problem generation succesful")
            print(f"Problem: {problem}")
            pddl_generator_succesful = True



        ########################
        # PDDL REFINEMENT LOOP #
        ########################

        PDDL_loop_iteration = 0
        while PDDL_loop_iteration < PDDL_GENERATION_ITERATIONS:
            print(f"\n\n\t#######################\n\t# Problem refinement loop iteration {PDDL_loop_iteration} out of {PDDL_GENERATION_ITERATIONS} \n\t#########################\n\n")


            plan_file_path = os.path.join(pddl_dir, f"plan_{PDDL_loop_iteration}.out")
            latest_planner_output_file_path = os.path.join(iteration_dir, f"planner_output_{PDDL_loop_iteration}.log")


            ############
            # Planning #
            ############
            
            print("\n\n\tPerforming planning...")
            planning_succesful = planner.run_planner(domain_file_path, problem_file_path, plan_file_path, latest_planner_output_file_path, planning_log_file_path)



            if planning_succesful:
                break



            ###################
            # PDDL REFINEMENT #
            ###################

            # PROBLEM REFINEMENT #

            print("\n\n\tRefining problem...")
            # Refine the problem and return the path to the refined version
            new_problem, new_problem_file_path = pddl_generator.refine_problem(
                latest_planner_output_file_path, 
                planning_log_file_path, 
                domain_file_path, 
                problem_file_path, 
                scene_graph=scene_graph_extracted, 
                task=current_goal,
                output_dir=pddl_dir
            )


            problem_file_path = new_problem_file_path
            PDDL_loop_iteration += 1


               
        # Step 2.a)
        ##############################
        # LOOP TERMINATION CONDITION #
        ##############################
        grounding_successful = False
        if PERFORM_GROUNDING:
            print("Grounding started...")
            if planning_succesful:
                grounding_successful = perform_grounding(plan_file_path, scene_graph_extracted)
            
            if grounding_successful:
                break


        # Step 3)
        #############################
        # PLAN/GOAL POST-PROCESSING #
        #############################

        if PERFORM_GOAL_REASONING:
            print("Grounding not successfu. Performing goal reasoning...")

            # Initialize the goal reasoner
            print("Creating GoalReasoner instance...")
            reasoner = GoalReasoner(api_key)
            
            # Load the file in planner/experiments/experiment_3/logs/pddl_ai.log
            planning_log = load_planning_log(os.path.join(results_dir,"logs","pddl_ai.log"))
            
            print("Performing goal reasoning...")
            # Reason about the goal
            action_modifications, goal_relaxation = reasoner.reason_about_goal(
                GOAL, 
                scene_graph_extracted,
                planning_log, 
                domain_file_path, 
                problem_file_path, 
                last_planning_succesful = planning_succesful, 
                output_dir = iteration_dir
            )

            # First attempt all action modifications
            for action_mod in action_modifications:

                print("Attempting action modification:")
                print(action_mod)
                
                # Change the plan to reflect the proposed action modification
                pddl_generator.handle_action_replacement(
                    plan_file_path, 
                    action_mod,
                    planning_log_file_path=planning_log_file_path
                )          
                        
            if goal_relaxation is not None:
                print("Scheduling goal relaxation:")
                print(goal_relaxation)
                print(goal_relaxation.relaxed_goal)
                # Otherwise attempt relaxing goal
                pddl_generator.add_goal_relaxation(goal_relaxation)

                goal_similarity = compute_goal_similarity(GOAL, goal_relaxation.relaxed_goal, model)
                print(f"Goal similarity: {goal_similarity}")
                print(f"Goal: {GOAL}")
                print(f"Relaxed goal: {goal_relaxation.relaxed_goal}")
                new_goal = goal_relaxation.relaxed_goal
            else:
                goal_similarity = 1
                new_goal = GOAL

            # Write the relaxed goal to goal_file_path, where we replace .txt with _final.txt
            final_goal_file_path = goal_file_path.replace(".txt", "_final.txt")
            with open(final_goal_file_path, "w") as file:
                file.write(new_goal)
                # Write similarity in the next row
                file.write(f"Similarity: \n{goal_similarity}")

        iteration += 1
    
    # Return the final problem and plan
    return problem_file_path, plan_file_path