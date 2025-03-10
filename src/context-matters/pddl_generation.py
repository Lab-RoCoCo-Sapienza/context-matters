import json
import datetime
import os
from pathlib import Path

from typing import Optional

from utils import print_to_planning_log, print_blue
from agent import llm_call

##########
# DOMAIN #
##########

def generate_domain(
    domain_file_path,
    goal_file_path, 
    domain_description, 
    logs_dir=None, 
    model="gpt-4o"
):
    print_blue("Generating PDDL domain...")

    prompt = """
    Role: You are an excellent PDDL domain generator. Given a description of the planning domain, given as a description of the domain actions and one of the domain objects, you must generate a PDDL domain file.
    Make sure the domain file respects the provided description for each action.

    IGNORE ROOM CONNECTIONS, YOU CAN MOVE FROM ANY ROOM TO ANY ROOM, WITHOUT THE NEED FOR A PREDICATE
    DON'T USE NUMERIC VALUES FOR LOCATIONS
    MAKE SURE TO DEFINE ALL THE OBJECTS AND PREDICATES NECESSARY FOR THE ACTIONS
    MAKE SURE TO RESPECT THE SYNTAX

    We support the following subset of PDDL1.2:
        STRIPS
        Typing (including hierarchical)
        Quantifiers (forall, exists)
        Disjunctions (or)
        Equality
        Constants
        Derived predicates
        

    Example: 
    DOMAIN DESCRIPTION:
        "actions": [
        {
            "name": "move_to",
            "description": "The robot moves from one room to another. Arguments: robot, starting room, destination room. Preconditions: The robot is in the starting room. Postconditions: The robot is no longer in the starting room and is now in the destination room."
        },
        {
            "name": "grab",
            "description": "The robot grabs a grabbable object in a room. Arguments: robot, grabbable object, room. Preconditions: The robot and the object are in the same room, and the robot is free to grab. Postconditions: The robot is holding the object, and the object is no longer in the room."
        },
        {
            "name": "drop",
            "description": "The robot drops a grabbable object in a room. Arguments: robot, grabbable object, room. Preconditions: The robot is holding the object and is in the room. Postconditions: The robot is no longer holding the object, the robot is free, and the object is in the room."
        },
        {
            "name": "open",
            "description": "The robot opens a washing machine in a room. Arguments: robot, washing machine, room. Preconditions: The robot and the washing machine are in the same room, and the washing machine is closed. Postconditions: The washing machine is open."
        },
        {
            "name": "close",
            "description": "The robot closes a washing machine in a room. Arguments: robot, washing machine, room. Preconditions: The robot and the washing machine are in the same room, and the washing machine is open. Postconditions: The washing machine is closed."
        },
        {
            "name": "refill",
            "description": "The robot refills a washing machine with a cleaning supply in a room. Arguments: robot, washing machine, room, cleaning supply. Preconditions: The robot and the washing machine are in the same room, the robot is holding the cleaning supply, and the washing machine is empty. Postconditions: The washing machine is refilled and no longer empty."
        },
        {
            "name": "put_inside",
            "description": "The robot puts a cleanable object inside a washing machine in a room. Arguments: robot, cleanable object, washing machine, room. Preconditions: The robot and the washing machine are in the same room, the robot is holding the cleanable object, and the washing machine is open. Postconditions: The robot is no longer holding the cleanable object, the cleanable object is inside the washing machine, and the robot is free."
        },
        {
            "name": "wash",
            "description": "The robot washes a cleanable object inside a washing machine in a room. Arguments: robot, cleanable object, washing machine, room. Preconditions: The robot and the washing machine are in the same room, the cleanable object is inside the washing machine, the washing machine is closed, and the washing machine is refilled. Postconditions: The cleanable object is clean and no longer dirty."
        }
        ],
        "objects": [
        {
            "type": "room",
            "description": "A room where actions can take place."
        },
        {
            "type": "locatable",
            "description": "An object that can be located in a room. Includes robots, washing machines, and grabbable objects."
        },
        {
            "type": "robot",
            "description": "A robot that can perform actions such as moving, grabbing, and interacting with other objects."
        },
        {
            "type": "washing-machine",
            "description": "A washing machine that can be opened, closed, refilled, and used to wash cleanable objects."
        },
        {
            "type": "grabbable",
            "description": "An object that can be grabbed by the robot. Includes cleaning supplies and cleanable objects."
        },
        {
            "type": "cleaning-supply",
            "description": "A supply used for cleaning, such as detergent."
        },
        {
            "type": "cleanable",
            "description": "An object that can be cleaned, such as clothes."
        }
        ]
    

    GENERATED PDDL DOMAIN
        (define (domain laundry-domain)

            (:requirements
                :strips
                :typing
            )

            (:types
                room locatable - object
                robot washing-machine grabbable - locatable
                cleaning-supply cleanable - grabbable
            )

            (:predicates
                (at ?something - locatable ?where - room)
                (is-holding ?who - robot ?something - grabbable)
                (is-free ?who - robot)
                (is-dirty ?what - cleanable)
                (is-clean ?what - cleanable)
                (is-open ?what - washing-machine)
                (is-closed ?what - washing-machine)
                (is-refilled ?what - washing-machine)
                (is-empty ?what - washing-machine)
                (inside ?what - cleanable ?where - washing-machine)
            )

            (:action move_to
                :parameters (?who - robot ?from - room ?to - room)
                :precondition (and (at ?who ?from))
                :effect (and (not (at ?who ?from)) (at ?who ?to))
            )
            
            (:action grab
                :parameters (?who - robot ?what - grabbable ?where - room)
                :precondition (and (at ?who ?where) (at ?what ?where) (is-free ?who))
                :effect (and (not (at ?what ?where)) (is-holding ?who ?what) (not (is-free ?who)))
            )
            
            (:action drop
                :parameters (?who - robot ?what - grabbable ?where - room)
                :precondition (and (at ?who ?where) (is-holding ?who ?what))
                :effect (and (not (is-holding ?who ?what)) (is-free ?who) (at ?what ?where))
            )
            
            (:action open
                :parameters (?who - robot ?what - washing-machine ?where - room)
                :precondition (and (at ?who ?where) (at ?what ?where) (is-closed ?what))
                :effect (and (is-open ?what) (not (is-closed ?what)))
            )
            
            (:action close
                :parameters (?who - robot ?what - washing-machine ?where - room)
                :precondition (and (at ?who ?where) (at ?what ?where) (is-open ?what))
                :effect (and (is-closed ?what) (not (is-open ?what)))
            )
            
            (:action refill
                :parameters (?who - robot ?what - washing-machine ?where - room ?with - cleaning-supply)
                :precondition (and (at ?who ?where) (at ?what ?where) (is-holding ?who ?with) (is-empty ?what))
                :effect (and (is-refilled ?what) (not (is-empty ?what)))
            )
            
            (:action put_inside
                :parameters (?who - robot ?what - cleanable ?in - washing-machine ?where - room) 
                :precondition (and (at ?who ?where) (at ?in ?where) (is-holding ?who ?what) (is-open ?in))
                :effect (and (not (is-holding ?who ?what)) (inside ?what ?in) (is-free ?who))
            )
            
            (:action wash
                :parameters (?who - robot ?what - cleanable ?in - washing-machine ?where - room) 
                :precondition (and (at ?who ?where) (at ?in ?where) (inside ?what ?in) (is-closed ?in) (is-refilled ?in))
                :effect (and (is-clean ?what) (not (is-dirty ?what)))
            )
            
        )

    """

    question = f"""
    Instruction:
    Extract new object types and actions from the following description and generate a corresponding PDDL domain file.

    <task>
    
    A new domain has the following new object types and actions.

    <domain_description>

    Please generate a corresponding PDDL domain file that incorporates these elements and respects the provided preconditions and effects.
    Write only the PDDL domain and nothing more.
    """

    task_description = Path(goal_file_path).read_text()
    question = question.replace("<task>", task_description)

    question = question.replace("<domain_description>", str(domain_description))

    answer = llm_call(prompt, question, model=model)

    # Save prompts and response
    _save_prompt_response(
        prompt=f"{prompt}\n\n{question}",
        response=answer,
        prefix="DELTA_pddl_domain_generation",
        suffix="",
        output_dir=logs_dir
    )

    domaind_pddl = answer.replace("`", "").replace("pddl", "").replace("lisp", "")
    with open(domain_file_path, "w") as file:
        file.write(domaind_pddl)


    return domaind_pddl


#################################
# IMPOSSIBLE PROBLEM PREVENTION #
#################################

# Ask the LLM if it is possible to achieve the given task with with the given PDDL domain and the given environment (a map room -> object)
def determine_problem_possibility(
    domain_file_path,
    initial_robot_location,
    task,
    environment,
    logs_dir,
    model = "gpt-4o"
):
    # Read the domain
    with open(domain_file_path, "r") as file:
        domain_pddl = file.read()

    # Prepare the prompt for the LLM
    prompt = f"""
    You are an expert in PDDL planning. Given a task description and a room->object map, determine if it is possible to achieve the task with the objects available in the environment. 
    If it is not possible, explain why.
    Please provide a clear explanation of whether the task is achievable or not, and if not, why it is impossible. 

    Make sure you declare a task impossible only if it can not be satisfied with other objects of the scene and/or in no other way.
    If a relaxed version of the goal can be satisfied, it should be considered possible. A suboptimal solution makes a task possible. 
    A goal is possible if at least its core objective is satisfiable, even in an suboptimal or unexpected way.
    An impossible goal should be a goal that is very far from realizable, even by replacing objects or relaxing some constraints.
    Assume that it is possible to carry all necessary objects.
    
    Use the following format for your response:

    <possible>BOOL</possible>
    <explanation>STRING</explanation>

    where BOOL is either "true" or "false" and STRING is a string explaining why the task is achievable or not.

    Never write outside the pairs of tags <possible> and <explanation>.
    """

    question=f"""Please determine if the task is achievable or not, and if not, why it is impossible, given the following information:

    Task:
    ```
    {task}
    ```

    Environment:
    ```
    {environment}
    ```
    """

    # Call the LLM
    response = llm_call(prompt, question, model=model)

    # Save the response
    _save_prompt_response(
        prompt=prompt,
        response=response,
        prefix="impossible_problem_prevention",
        suffix="",
        output_dir=logs_dir
    )

    # Parse the response to extract <possible> and <explanation> tags
    possible = response.split("<possible>")[1].split("</possible>")[0].strip()
    explanation = response.split("<explanation>")[1].split("</explanation>")[0].strip()

    return possible, explanation


###########
# PROBLEM #
###########

# PROBLEM GENERATION #

def generate_problem(
    domain_file_path, 
    initial_robot_location, 
    task, 
    environment, 
    problem_file_path,
    logs_dir,
    workflow_iteration,
    model = "gpt-4o",
    USE_EXAMPLE = False,
    ADD_PREDICATE_UNDERSCORE_EXAMPLE = True
    ):

    # Directly use the provided problem file path since directory structure should exist
    # Read the domain
    with open(domain_file_path, "r") as file:
        domain = file.read()
    
    system_prompt = f"""
        You are an expert in generating PDDL problem files. 
        In particular, your task is to generate a PDDL problem file given: 
        - the representation of the environment (contained within the <environment> XML tags), where the robotic agent will have to execute the resulting plan, as a room to object map
        - the PDDL domain (contained within the <PDDL_planning_domain> XML tags), which defines the actions and predicates that the robot can use to solve the task
        - the task that the robot has to achieve (contained within the <planning_goal> XML tags)
        - the initial location of the robot in the environment (contained within the <initial_robot_location> XML tags)
        """

    system_prompt += """
        The environment in which the task will have to be satisfied is represented as a dictionary of objects representing a scene graph, with the following features:
        - a list of dictionaries, where each dictionary represents an object with specific properties and attributes.
        - each object includes details such as action affordances, dimensions, material composition, and spatial location.

        The logical structure of the PDDL problem should be consistent with the domain, meaning that you have to respect the name of the predicates and actions defined in the domain file without changing them (avoid typos).
        You can use only the elements and informations provided in the scene graph to generate the PDDL problem by using
        the types and predicates defined in the domain file.

        In the PDDL problem :objects block exclude any element that will surely not be necessary for the satisfaction of the plan.
        (:object
            ONLY USEFUL OBJECTS HERE
        )
        
        In the PDDL problem :init block, avoid initializing predicates that are linked to elements or properties in the planning domain that are surely not necessary for the satisfaction of the plan.
        (:init
            ONLY NECESSARY PREDICATES HERE
            DON'T USE NUMERIC VALUES FOR LOCATIONS
        )

        DON'T USE FORALL OR EXISTS OPERATORS IN THE GENERATED PROBLEM
        IGNORE ROOM CONNECTIONS, YOU CAN MOVE FROM ANY ROOM TO ANY ROOM, WITHOUT THE NEED FOR A PREDICATE

        Make sure that all all the objects in the :init block are initialized in the :objects block (do not forget anything).
        Make sure that all rooms and the initial robot location are present in the :objects block and that the initial robot location is correctly initialized.

        ANSWER WITH ONLY THE PDDL AND NOTHING ELSE. Any code other than the PDDL problem will mean that the system will fail.
    """

    if ADD_PREDICATE_UNDERSCORE_EXAMPLE:
        system_prompt += """
        MAKE SURE TO ALWAYS USE HYPHENS IN VARIABLE NAMES AND PREDICATES:
        Positive example:
        (at-robot ?robot ?room)
        Negative example:
        (at_robot ?robot ?room)
        """

    if USE_EXAMPLE:
        system_prompt += """
        Example:
        Given an EXAMPLE_GRAPH, an EXAMPLE_GOAL, and using the predicates defined in EXAMPLE_DOMAIN, a corresponding PDDL problem file looks like GENERATED_PROBLEM.

        EXAMPLE_GRAPH:
            bathroom_1:
            - sink_3 - A silver sink made of stainless steel.
            - vase_7 - A pink glass vase made of ceramic.
            - toilet_20 - A brown toilet made of ceramic.
            - potted plant_28 - A blue vase made of glass.

            kitchen_2:
            - microwave_1 - A black microwave made of metal.
            - oven_2 - A red stove made of brick.
            - sink_4 - A blue bathtub made of ceramic.
            - refrigerator_6 - Red refrigerator made of metal.
            - bowl_16 - A turquoise bowl made of ceramic.
            - apple_18 - A red phone made of metal.
            - apple_19 - red pineapple
            made of plastic
            in which you replace red with orange, pineapple with banana, plastic with vinyl
            - chair_26 - A brown sofa made of leather.
            - locker_37 - A brown cabinet made of metal.
            - keyboard into the locker_44 - Black laptop into the metal locker.
            - mop_11 - A blue mop made of plastic.
            - bucket_13 - A red bucket made of plastic.


            living_room_1:
            - bowl_17 - A brown plate made of ceramic.
            - chair_22 - red chair made of leather
            - chair_25 - A brown chair made of wood.
            - couch_27 - A brown sofa made of leather.
            - book_2 - into the shelf_38 - A brown book made of leather.
            - glass_15 - A brown glass made of glass.

            corridor_6:
            - headphone into the locker_45 - Black headphones into the locker.

            corridor_7:  - No objects


        EXAMPLE_GOAL:
            Clean the kitchen.


        EXAMPLE_DOMAIN:
            (define (domain house-cleaning-domain)

                (:requirements
                    :strips
                    :typing
                )

                (:types
                    room locatable - object
                    robot grabbable bin - locatable
                    disposable mop - grabbable

                )

                (:predicates
                    (at ?something - locatable ?where - room)
                    (is-holding ?who - robot ?something - grabbable)
                    (is-free ?who - robot)
                    (thrashed ?what - disposable)
                    (is-clean ?what - mop)
                    (is-dirty ?what - mop)
                    (dirty-floor ?what - room)
                    (clean-floor ?what - room)
                )

                (:action move_to
                    :parameters (?who - robot ?from - room ?to - room)
                    :precondition (and (at ?who ?from))
                    :effect (and (not (at ?who ?from)) (at ?who ?to))
                )
                
                (:action grab
                    :parameters (?who - robot ?what - grabbable ?where - room)
                    :precondition (and (at ?who ?where) (at ?what ?where) (is-free ?who))
                    :effect (and (not (at ?what ?where)) (is-holding ?who ?what) (not (is-free ?who)))
                )
                
                (:action drop
                    :parameters (?who - robot ?what - grabbable ?where - room)
                    :precondition (and (at ?who ?where) (is-holding ?who ?what))
                    :effect (and (not (is-holding ?who ?what)) (is-free ?who) (at ?what ?where))
                )
                
                (:action throw_away
                    :parameters (?who - robot ?what - disposable ?in - bin ?where - room)
                    :precondition (and (at ?who ?where) (is-holding ?who ?what) (at ?in ?where))
                    :effect (and (not (is-holding ?who ?what)) (is-free ?who) (thrashed ?what) (not (at ?what ?where)))
                )
                
                (:action mop_floor
                    :parameters (?who - robot ?with - mop ?where - room)
                    :precondition (and (at ?who ?where) (is-holding ?who ?with)
                                (is-clean ?with) (dirty-floor ?where))
                    :effect (and (not (dirty-floor ?where)) (not (is-clean ?with))
                            (clean-floor ?where) (is-dirty ?with)    
                    )
                )
                
                (:action clean_mop
                    :parameters (?who - robot ?what - mop)
                    :precondition (and (is-holding ?who ?what) (is-dirty ?what))
                    :effect (and (not (is-dirty ?what)) (is-clean ?what))
                )
            )


        GENERATED_PROBLEM:
            (define (problem house_cleaning)
            (:domain house-cleaning-domain)

                (:objects
                    robot - robot
                    kitchen_2 living_room_1 - room
                    mop_11 - mop
                    bucket_13 - grabbable
                    glass_15 - grabbable
                )
                (:init
                    (at robot kitchen_2)
                    (at mop_11 kitchen_2)
                    (at bucket_13 kitchen_2)
                    (at glass_15 living_room_1)
                    (dirty-floor kitchen_2)
                    (dirty-floor living_room_1)
                )
                (:goal
                    (and
                        (clean-floor kitchen_2)
                        (clean-floor living_room_1)
                    )
                )
            ) 
            """
                
    user_prompt = f"""
        The goal for the robot is the following:
        <planning_goal> 
        {task}
        </planning_goal>

    """
    
    user_prompt += f"""
        <PDDL_planning_domain>
        {domain}
        </PDDL_planning_domain>

        <environment>
        {environment}
        </environment>

    """

    user_prompt += f"<initial_robot_location> {initial_robot_location} </initial_robot_location>"
    #print(user_prompt)

    problem = llm_call(system_prompt, user_prompt, model=model)
    
    # Save prompts and response
    _save_prompt_response(
        prompt=f"{system_prompt}\n\n{user_prompt}",
        response=problem,
        prefix="problem_generation",
        suffix=str(workflow_iteration),
        output_dir=logs_dir
    )

    problem = problem.replace("`", "").replace("pddl", "").replace("lisp", "")
    with open(problem_file_path, "w") as file:
        file.write(problem)
    
    return problem




# PROBLEM REFINEMENT #

def refine_problem(
    planner_output_file_path,
    domain_file_path,
    problem_file_path,
    scene_graph,
    task,
    logs_dir,
    workflow_iteration,
    refinement_iteration,
    pddlenv_error_log = None,
    planner_error_log = None,
    VAL_validation_log = None,
    VAL_grounding_log = None,
    scene_graph_grounding_log = None,
    model = "gpt-4o"
):

    # 1) Read the relevant files
    with open(problem_file_path, "r") as file:
        problem_pddl = file.read()

    # Open the domain
    with open(domain_file_path, "r") as file:
        domain_pddl = file.read()

    # ----------------------------------------------------------
    # STEP A: Ask LLM for the reason of failure (Diagnosis)
    # ----------------------------------------------------------
    reason_system_prompt = (
        "You are a helpful planning assistant. You have access to the domain PDDL, problem PDDL, planner output, and scene. Your job is to figure out why planning might have failed, but do NOT rewrite the problem yet."
    )
    reason_user_prompt = f"""
    Below is the domain PDDL file:
    ```
    {domain_pddl}
    ```
    Below is the problem PDDL file:
    ```
    {problem_pddl}
    ```

    The original task is: {task}
    The scene is: {scene_graph}

    """

    if pddlenv_error_log is not None:
        reason_user_prompt += f"""
        The PDDLGym PDDLEnv simulation environment returned the following error:
        ```
        {pddlenv_error_log}
        ```
    """
    
    if planner_error_log is not None:
        reason_user_prompt += f"""
        The FD planner of the PDDLGym library returned the following error:
        ```
        {planner_error_log}
        ```
    """

    if VAL_validation_log is not None:
        reason_user_prompt += f"""
        An attempt to validate the plan using the VAL PDDL validation tool returned the following error:
        ```
        {VAL_validation_log}
        ```
    """

    if VAL_grounding_log is not None:
        reason_user_prompt += f"""
        An attempt to generate the plan and ground it in the given domain and problem using the VAL PDDL grounding tool returned the following error:
        ```
        {VAL_grounding_log}
        ```
    """


    if scene_graph_grounding_log is not None:
        reason_user_prompt += f"""
        An attempt to ground the previously generatred plan into the scene graph returned the following error:
        ```
        {scene_graph_grounding_log}
        ```
    """

    reason_user_prompt += f"""
    Please provide a clear explanation of the possible reason(s) for the planning failure.\
    At the end provide detailed suggestions to solve the issues you found.\
    Focus on diagnosing the issue. Do NOT rewrite the PDDL yet; just explain the error.\
    Verify if predicates are written correctly, if the objects are correctly defined.
    """

    # Call the LLM for diagnosis
    reason_of_failure = llm_call(reason_system_prompt, reason_user_prompt)

    _save_prompt_response(
        prompt=reason_system_prompt + "\n\n" + reason_user_prompt,
        response=reason_of_failure,
        prefix="problem_diagnosis",
        suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
        output_dir=logs_dir
    )

    # ----------------------------------------------------------
    # STEP B: Ask LLM to correct the problem (Correction)
    # ----------------------------------------------------------
    correction_system_prompt = (
        "You are a helpful planning assistant. You have access to the domain PDDL, "
        "the old problem, and a reason of failure from a previous analysis. "
        "Now your job is to rewrite or fix the problem PDDL to address that failure"
        "according to the suggestions provided by the expert."
    )
    correction_user_prompt = f"""
    Below is the old problem PDDL file:
    ```
    {problem_pddl}
    ```
    Reason of failure (expert suggest to you):
    ```
    {reason_of_failure}
    ```
    Now please:
    1. Make sure to fix any domain-problem inconsistencies.
    2. Make sure all objects/types/predicates in the problem match what's in the domain.
    3. Make sure the goal is achievable from the initial state.
    4. Output your changes with the following format:

    <PROBLEM>
    The new corrected PDDL problem here.
    </PROBLEM>

    Use always the tokens otherwise the entire system fails.
    """

    # Call the LLM for correction
    llm_response = llm_call(correction_system_prompt, correction_user_prompt, model=model)

    # Save prompt & response if you like
    _save_prompt_response(
        prompt=correction_system_prompt + "\n\n" + correction_user_prompt,
        response=llm_response,
        prefix="problem_refinement",
        suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
        output_dir=logs_dir
    )

    # ----------------------------------------------------------
    # Extract the new problem from LLM response
    # ----------------------------------------------------------
    try:
        new_problem = llm_response.split("<PROBLEM>")[1].split("</PROBLEM>")[0]
    except (IndexError, ValueError):
        # If the format is not as expected, fallback to entire response
        llm_response = llm_call(correction_system_prompt, correction_user_prompt)
        _save_prompt_response(
            prompt=correction_system_prompt + "\n\n" + correction_user_prompt,
            response=llm_response,
            prefix="problem_refinement",
            suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
            output_dir=logs_dir
        )
        new_problem = llm_response.split("<PROBLEM>")[1].split("</PROBLEM>")[0]

    # Cleanup new problem string
    new_problem = new_problem.replace("`", "").replace("pddl", "").replace("lisp", "")

    # Optionally, log the new iteration
    iteration_log = f"Iteration #\nReason of failure:\n{reason_of_failure}\n\nLLM correction:\n{llm_response}"
    planning_log_file_path = os.path.join(logs_dir, f"refinement_{workflow_iteration}_{refinement_iteration}.txt")
    with open(planning_log_file_path, "a") as log_file:
        log_file.write(iteration_log)

    return new_problem


#########
# OTHER #
#########


def _save_prompt_response(prompt: str, response: str, output_dir: str, prefix: str, suffix: str) -> None:
    """Save prompt and response to separate files."""
        
    base_name = f"{prefix}_{suffix}"
    
    # Save prompt
    prompt_file = os.path.join(output_dir, f"{base_name}_prompt.log")
    with open(prompt_file, "w") as f:
        f.write(prompt)
        
    # Save response
    response_file = os.path.join(output_dir, f"{base_name}_response.log")
    with open(response_file, "w") as f:
        f.write(json.dumps(response, indent=4))

