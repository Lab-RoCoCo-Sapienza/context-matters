import json
import datetime
import os
from typing import Optional

from utils import print_to_planning_log
from agent import llm_call

##########
# DOMAIN #
##########

def generate_domain(
    task_file, 
    domain_description, 
    logs_dir=None, 
    model="gpt-4o"
):
    print_blue("Generating PDDL domain...")

    prompt = """
    Role: You are an excellent PDDL domain generator. Given a description of domain knowledge, you can generate a PDDL domain file.
    We support the following subset of PDDL1.2:
        STRIPS
        Typing (including hierarchical)
        Quantifiers (forall, exists)
        Disjunctions (or)
        Equality
        Constants
        Derived predicates

    Act as if rooms are all interconnected. Make sure all types are present when using them.

    Example: 
    A robot in a household environment can perform the following actions on various objects.
    For instance, consider the action "mop floor":
    - Natural Language Description:
        For mopping the floor, the agent must be in the room and have the mop in hand.
        The mop must be clean and the floor must not be clean.
        After performing the action, the floor becomes clean, but the mop becomes dirty and the agent’s battery is no longer full.

    - Corresponding PDDL Definition:
    
    (define (domain house_cleaning)
      (:requirements 
        :strips
    )

    (:predicates
        (agent-at ?r)
        (has ?agent ?object)
        (clean ?object)
        (floor-clean ?r)
        (battery-full)
      )
    
    (:action mop_floor
        :parameters (?a - agent ?i - item ?r - room)
        :precondition 
        (and
            (agent_at ?a ?r)
            (item_is_mop ?i)
            (item_pickable ?i)
            (agent_has_item ?a ?i)
            (mop_clean ?i)
            (not(floor_clean ?r))
        )

        :effect 
        (and
            (floor_clean ?r)
            (not(mop_clean ?i))
            (not(battery_full ?a))
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

    task_description = Path(task_file).read_text()
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

    return domaind_pddl



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
    model = "gpt-4o"
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
        - [OPTIONALLY] the initial location of the robot in the environment (contained within the <initial_robot_location> XML tags)
        """

    system_prompt += f"""
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

        Make sure that all all the objects in the :init block are initialized in the :objects block (do not forget anything).
        Make sure that all rooms and the initial robot location are present in the :objects block and that the initial robot location is correctly initialized.

        ANSWER WITH ONLY THE PDDL AND NOTHING ELSE. Any code other than the PDDL problem will mean that the system will fail. 
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
    val_validation_log = None,
    val_grounding_log = None,
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
        "You are a helpful planning assistant. You have access to the domain PDDL, "
        "problem PDDL, planner output, and scene. Your job is to figure out why planning "
        "might have failed, but do NOT rewrite the problem yet."
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

    if val_validation_log is not None:
        reason_user_prompt += f"""
        An attempt to validate the plan using the VAL PDDL validation tool returned the following error:
        ```
        {val_validation_log}
        ```
    """

    if val_grounding_log is not None:
        reason_user_prompt += f"""
        An attempt to generate the plan and ground it in the given domain and problem using the VAL PDDL grounding tool returned the following error:
        ```
        {val_grounding_log}
        ```
    """


    if scene_graph_grounding_log is not None:
        reason_user_prompt += f"""
        An attempt to ground the previously generatred plan into the scene graph returned the following error:
        ```
        {grounding_error_log}
        ```
    """

    reason_user_prompt += f"""
    Please provide a clear explanation of the possible reason(s) for the planning failure.
    at the end provide the suggestion to solve the issues with high details.
    Focus on diagnosing the issue. Do NOT rewrite the PDDL yet; just explain the error.
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

