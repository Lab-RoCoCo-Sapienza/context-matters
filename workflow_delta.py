import os

from utils import *
from pathlib import Path
import json

from pprint import pprint


from agent import llm_call
from planner import plan_with_output
from pddlgym.core import PDDLEnv
from pddl_verification import verify_groundability_in_scene_graph, convert_JSON_to_locations_dictionary, VAL_validate, VAL_parse, VAL_ground

from pddl_generation import _save_prompt_response


def generate_pddl_domain(task_file, domain_description, logs_dir=None):
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

    question = question.replace("<domain_description>", domain_description)

    answer = llm_call(prompt, question)

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


def prune_scene_graph(scene_graph_path, goal_description_path, initial_robot_location=None, logs_dir=None):
    print_blue("Pruning Scene Graph...")
    
    prompt = """
    Role: You are an excellent assistant in pruning SGs with a list of SG items and a goal description.

    Example: A SG can be programmed as a nested Python dictionary such as 
    {
        "room": {
            '1': {
                    'floor_area': 7.309859732351093, 
                    'floor_number': 'A', 
                    'id': 1, 
                    'location': [-0.7197105, -1.610225, 1.1466205], 
                    'scene_category': 'dining_room', 
                    'size': [2.5, 1.3, 0.8], 
                    'volume': 7.856456590462219, 
                    'parent_building': 70
                }
            },
            '2': {
                    'floor_area': 7.309859732351093, 
                    'floor_number': 'B', 
                    'id': 2, 
                    'location': [-1.7197105, -2.610225, 1.255566205], 
                    'scene_category': 'kitchen', 
                    'size': [1.836419, 1.91739, 2.326879], 
                    'volume': 5,21398719237, 
                    'parent_building': 70
                }
            },
            '3': {
                    'floor_area': 7.309859732351093, 
                    'floor_number': 'A', 
                    'id': 3, 
                    'location': [-0.8297505, -1.717225, 1.77466205], 
                    'scene_category': 'bathroom', 
                    'size': [1.836419, 1.91739, 2.326879], 
                    'volume': 4.948372943, 
                    'parent_building': 70
                }
            },
    "object": {
        '11': {
                'action_affordance': ['wash', 'clean', 'clean with'], 
                'floor_area': 1.7741857552886344, 
                'surface_coverage': 1.0771061806828595, 
                'class_': 'sink', 
                'id': 11, 
                'location': [-0.77102688, -1.56634777, 0.45304914], 
                'material': None, 
                'tactile_texture': None, 
                'visual_texture': None, 
                'volume': 0.13050794184989123, 
                'parent_room': 2, 
                'description': 'A metallic sink.'
            },
        '15': {
                'action_affordance': ['wash', 'clean', 'break', 'move'], 
                'floor_area': 1.7741857552886344, 
                'surface_coverage': 0.02222, 
                'class_': 'glass', 
                'id': 15, 
                'material': 'glass', 
                'tactile_texture': None, 
                'visual_texture': None, 
                'volume': 0.13050794184989123, 
                'parent_room': 1, 
                'description': 'A drinking glass.'
            },
        '17': {
                'action_affordance': ['wash', 'clean', 'clean with'], 
                'floor_area': 1.7741857552886344, 
                'surface_coverage': 0.02222, 
                'class_': 'sink', 
                'id': 17, 
                'material': 'ceramic', 
                'tactile_texture': None, 
                'visual_texture': None, 
                'volume': 0.13050794184989123, 
                'parent_room': 3, 
                'description': 'A ceramic sink.'
            },

    For accomplishing the goal "put the glass into the sink", the relevant items are the glass in the dining_room and the sink in the kitchen and the relevant rooms are the dining_room and the kitchen.
    """

    # Carica il Scene Graph originale
    scene_graph = read_graph_from_path(Path(scene_graph_path))
    filtered_sg = filter_graph(scene_graph, ["room", "object"])
    prompt = prompt.replace("<filtered_graph>", str(filtered_sg))

    question = """Instruction: Given a new scene graph and a goal, please prune the SG by keeping the relevant items. 
    """

    if initial_robot_location is not None:
        question += f"""
        Keep in mind that the robot is initially located in '{initial_robot_location}', therefore this room should be included in the pruned scene graph."
        """

    question += """
    Write only the pruned graph respecting the original structure.
    The graph should be readily parsable in JSON.

    GOAL:
    <GOAL>

    NEW GRAPH:
    <new_graph>
    """

    # Carica la descrizione del task
    goal_description = Path(goal_description_path).read_text()

    # Carica il nuovo Scene Graph da filtrare
    new_sg = read_graph_from_path(Path(scene_graph_path))
    filtered_new_sg = filter_graph(new_sg, ["room", "object"])
    
    question = question.replace("<new_graph>", str(filtered_new_sg)).replace("<GOAL>", goal_description)


    # Esegui il pruning
    pruned_sg = llm_call(prompt, question)

    # Remove ```json and ``` from the response
    pruned_sg = pruned_sg.replace("`", "").replace("json", "").replace("lisp", "")

    pprint(pruned_sg)

    # Parse the graph into json
    pruned_sg = json.loads(pruned_sg)

    # Save prompts and response
    _save_prompt_response(
        prompt=f"{prompt}\n\n{question}",
        response=pruned_sg,
        prefix="DELTA_pruned_scene_graph",
        suffix="",
        output_dir=logs_dir
    )
    
    return pruned_sg

def generate_pddl_problem(pruned_scene_graph, goal_description_path, domain_pddl_path, initial_robot_location=None, logs_dir=None):
    print_blue("Generating PDDL problem...")

    prompt = """
    Role: You are an excellent problem generator. Given a Scene Graph (SG) and a desired goal,
    you can generate a PDDL problem file.
    We support the following subset of PDDL1.2:
        STRIPS
        Typing (excluding hierarchical)
        Quantifiers (forall, exists)
        Disjunctions (or)
        Equality
        Constants
        Derived predicates

    """


    prompt += """
    Example:
    Given an EXAMPLE_GRAPH, an EXAMPLE_GOAL, and using the predicates defined in EXAMPLE_DOMAIN, a corresponding PDDL problem file looks like GENERATED_PROBLEM.

    EXAMPLE_GRAPH:
        {
            "room": {
                '1': {
                        'floor_area': 7.309859732351093, 
                        'floor_number': 'A', 
                        'id': 1, 
                        'location': [-0.7197105, -1.610225, 1.1466205], 
                        'scene_category': 'living_room', 
                        'size': [2.5, 1.3, 0.8], 
                        'volume': 7.856456590462219, 
                        'parent_building': 70
                    }
                },
                '2': {
                        'floor_area': 7.309859732351093, 
                        'floor_number': 'B', 
                        'id': 2, 
                        'location': [-1.7197105, -2.610225, 1.255566205], 
                        'scene_category': 'kitchen', 
                        'size': [1.836419, 1.91739, 2.326879], 
                        'volume': 5,21398719237, 
                        'parent_building': 70
                }
        },
        "object": {
            '11': {
                    'action_affordance': ['wash', 'clean', 'clean with'], 
                    'floor_area': 1.7741857552886344, 
                    'surface_coverage': 1.0771061806828595, 
                    'class_': 'mop', 
                    'id': 11, 
                    'location': [-0.77102688, -1.56634777, 0.45304914], 
                    'material': None, 
                    'tactile_texture': None, 
                    'visual_texture': None, 
                    'volume': 0.13050794184989123, 
                    'parent_room': 2, 
                    'description': 'A mop.'
            },
            '13': {
                    'action_affordance': ['wash', 'clean', 'clean with'], 
                    'floor_area': 1.7741857552886344, 
                    'surface_coverage': 1.0771061806828595, 
                    'class_': 'bucket', 
                    'id': 13, 
                    'location': [-0.77102688, -1.56634777, 0.45304914], 
                    'material': None, 
                    'tactile_texture': None, 
                    'visual_texture': None, 
                    'volume': 0.13050794184989123, 
                    'parent_room': 2, 
                    'description': 'A metallic bucket.'
            },
            '15': {
                    'action_affordance': ['wash', 'clean', 'break', 'move'], 
                    'floor_area': 1.7741857552886344, 
                    'surface_coverage': 0.02222, 
                    'class_': 'glass', 
                    'id': 15, 
                    'material': 'glass', 
                    'tactile_texture': None, 
                    'visual_texture': None, 
                    'volume': 0.13050794184989123, 
                    'parent_room': 1, 
                    'description': 'A drinking glass.'
            }
        }
    }

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
                robot - agent
                kitchen_2 living_room_1 - room
                mop_11 - item
                bucket_13 - item
                glass_15 - item
            )
            (:init
                (at robot kitchen)
                (at mop_11 kitchen)
                (at bucket_13 kitchen)
                (at glass_15 living_room)
                (dirty-floor kitchen)
                (dirty-floor living_room)
            )
            (:goal
                (and
                    (clean-floor kitchen)
                    (clean-floor living_room)
                )
            )
        )
    """

    question = """

    Instruction:
    Given a new Scene Graph SCENE_GRAPH, a new goal description GOAL_DESCRIPTION, and the predicates defined in DOMAIN_PDDL, generate a new PDDL problem file.
    Write only the PDDL problem and nothing more.

    SCENE_GRAPH:
    <scene_graph>

    GOAL_DESCRIPTION:
    <goal_description>

    DOMAIN_PDDL:
    <domain_pddl>
    """

    if initial_robot_location is not None:
        question += f"""
        The robot is initially located in '{initial_robot_location}', therefore make sure that the problem you are generating includes '{initial_robot_location}'.
        """

    # Carica il Scene Graph prunato
    #filtered_sg = read_graph_from_path(Path(scene_graph_path))
    #filtered_sg = filter_graph(scene_graph, ["room", "object"])


    # Carica la descrizione del goal
    goal_description = Path(goal_description_path).read_text()

    # Carica il dominio PDDL
    domain_pddl = Path(domain_pddl_path).read_text()

    # Personalizza il prompt
    question = question.replace("<scene_graph>", str(pruned_scene_graph))
    question = question.replace("<goal_description>", goal_description)
    question = question.replace("<domain_pddl>", domain_pddl)

    # Esegui la generazione del PDDL del problema
    answer = llm_call(prompt, question)

    # Save prompts and response
    _save_prompt_response(
        prompt=f"{prompt}\n\n{question}",
        response=answer,
        prefix="DELTA_generate_pddl_problem",
        suffix="",
        output_dir=logs_dir
    )

    problem_pddl = answer.replace("`", "").replace("pddl", "").replace("lisp", "")

    return problem_pddl

def decompose_pddl_goal(problem_pddl_path, domain_pddl_path, initial_robot_location=None, logs_dir=None):
    print_blue("Decomposing PDDL goal...")
    """
    Scompone gli obiettivi di un file PDDL del problema in una sequenza di sotto-obiettivi (sub-goals).
    """
    prompt = """
    Role: You are an excellent assistant in decomposing long-term goals. Given a PDDL problem file, 
    you can decompose the goal states into a sequence of sub-goals.

    Example:
    Given an example_problem.pddl, the goal states can be decomposed into a sequence of example sub-goals. Using the predicates defined in example_domain.pddl, the example subgoals can be formulated as: sub-goal_1.pddl, ..., sub-goal_n.pddl.
    


    example_problem.pddl:
        (define (problem house_cleaning)
        (:domain cleaning)
            (:objects
                robot - agent
                kitchen living_room - room
                mop - item
                sink - item
                bucket - item
                water - liquid
            )
            (:init
                (agent_at robot kitchen)
                (has bucket water)
                (mop_dry mop)
                (floor_dirty kitchen)
                (floor_dirty living_room)
            )
            (:goal
                (and
                    (floor_clean kitchen)
                    (floor_clean living_room)
                )
            )
        )



    example_domain.pddl:
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



    sub-goal_1.pddl:
        (define (problem sub-goal-1)
            (:domain cleaning)
            (:objects
                robot - agent
                room - room
                mop - item
                bucket - container
                water - liquid
            )
            (:init
                (agent_at robot kitchen)
                (has bucket water)
                (mop_dry mop)
                (floor_dirty room)
            )
            (:goal
                (and
                    (mop_wet mop)
                )
            )
        )


    sub-goal_2.pddl:
        (define (problem sub-goal-2)
            (:domain cleaning)
            (:objects
                robot - agent
                room - room
                mop - item
                bucket - container
                water - liquid
            )
            (:init
                (agent_at room)
                (has bucket water)
                (mop_wet mop)
                (floor_dirty room)
            )
            (:goal
                (and
                    (floor_clean room)
                )
            )
        )
"""
    # Carica il file PDDL del problema
    problem_pddl = Path(problem_pddl_path).read_text()

    # Carica il file PDDL del dominio
    domain_pddl = Path(domain_pddl_path).read_text()

    # Personalizza il prompt con i file PDDL reali
    question = f"""
    Given the following problem PDDL:
    {problem_pddl}

    And the following domain PDDL:
    {domain_pddl}

    Decompose the goal into sub-goals in pddl. 
    """
    if initial_robot_location is not None:
        prompt += f"""
        The robot is initially located in robot location '{initial_robot_location}', therefore the first generated subgoal should reflect this.
        """

    prompt += """    
    The subsequent subgoals should reflect the intermediate states of the goal.
    Your answer should be a collection of readily usable as a PDDL problems.
    Write only the PDDL problems, separated ONLY by the <subgoal> tag, WITHOUT ANY ADDITIONAL TEXT.
    """

    # Esegui la generazione dei sub-goal PDDL
    answer = llm_call(prompt, question)

    # Save prompts and response
    _save_prompt_response(
        prompt=f"{prompt}\n\n{question}",
        response=answer,
        prefix="DELTA_decompose_pddl_goal",
        suffix="",
        output_dir=logs_dir
    )


    sub_goals_pddl = answer.replace("`", "").replace("pddl", "").replace("lisp", "").strip()
    sub_goals_pddl = sub_goals_pddl.split("<subgoal>")

    # Remove empty strings
    sub_goals_pddl = [sub_goal.strip() for sub_goal in sub_goals_pddl if sub_goal.strip() != ""]

    # Filter remove subgoals that do not contain the word "(problem"
    sub_goals_pddl = [sub_goal for sub_goal in sub_goals_pddl if "(problem" in sub_goal]

    print(sub_goals_pddl)

    return sub_goals_pddl


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
    GROUND_IN_SCENE_GRAPH = False
):
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    initial_robot_location = open(initial_location_file_path, "r").read()


    if domain_file_path is None:
        
        assert domain_description is not None, "Provide the domain or the domain description"

        # Run the pipeline
        print_green("\n######################\n# GENERATING PDDL DOMAIN #\n######################")
        domain_pddl = generate_pddl_domain(goal_file_path, domain_description, logs_dir=logs_dir)

        # Save the generated PDDL domain
        domain_file_path = os.path.join(results_dir, "domain.pddl")
        with open(domain_file_path, "w") as f:
            f.write(domain_pddl)
    


    # CHECK #1
    # Use VAL_parse to check if the domain is valid
    val_parse_success, val_parse_log = VAL_parse(domain_file_path)
    if not val_parse_success:
        return domain_file_path, None, None, None, False, False, None, "DOMAIN_GENERATION", val_parse_log



    print_cyan("\n######################\n# PRUNING SCENE GRAPH #\n######################")
    pruned_sg = prune_scene_graph(scene_graph_file_path, goal_file_path, initial_robot_location, logs_dir=logs_dir)

    # Save the pruned scene graph
    pruned_sg_path_readable = os.path.join(results_dir, "pruned_sg.txt")
    with open(pruned_sg_path_readable, "w") as f:
        f.write(json.dumps(pruned_sg, indent=4))

    pruned_sg_path_npz = os.path.join(results_dir, "pruned_sg.npz")
    save_graph(pruned_sg, pruned_sg_path_npz)

    

    print_yellow("\n######################\n# GENERATING PDDL PROBLEM #\n######################")
    problem_pddl = generate_pddl_problem(pruned_sg_path_npz, goal_file_path, domain_file_path, initial_robot_location, logs_dir=logs_dir)
    
    # Save the generated PDDL problem
    problem_pddl_path = os.path.join(results_dir, "problem.pddl")
    with open(problem_pddl_path, "w") as f:
        f.write(problem_pddl)



    # CHECK #2
    # Use VAL_parse to check if the problem is valid
    val_parse_success, val_parse_log = VAL_parse(domain_file_path)
    if not val_parse_success:
        return domain_file_path, pruned_sg, problem_pddl_path, None, False, False, None, "PROBLEM_GENERATION", val_parse_log
        



    print_magenta("\n######################\n# GENERATING PDDL SUB-GOALS #\n######################")
    sub_goals_pddl = decompose_pddl_goal(problem_pddl_path, domain_file_path, initial_robot_location, logs_dir=logs_dir)



    print_blue("\n######################\n# PLANNING AND GROUNDING AUTOREGRESSIVE #\n######################")
    
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
    # 1) Try grounding each subgoal
    # 2) Verify if its initial conditions coincide with the final conditions of the previous subgoal
    for i, sub_goal in enumerate(sub_goals_pddl):
        grounding_succesful = False
        planning_succesful = False

        sub_goal_dir = os.path.join(results_dir, f"sub_goal_{i+1}")
        sub_goal_file = os.path.join(sub_goal_dir, f"sub_goal_{i+1}.pddl")
        # Initialize env
        new_pddl_env = PDDLEnv(domain_file_path, sub_goal_dir, operators_as_actions=True)

        # Compare envs to check that their initial conditions coincide
        if old_pddl_env is not None:
            envs_coincide = new_pddl_env.initial_state == old_pddl_env.initial_state
        
            # Return with both planning successful and grounding successful set to False
            if not envs_coincide:
                break
        
        output_plan_file_path = os.path.join(sub_goal_dir, f"plan_{i+1}.txt")

        # Compute plan on env
        plan, pddlenv_error_log, planner_error_log = plan_with_output(domain_file_path, sub_goal_dir, output_plan_file_path, env=new_pddl_env)
        
        # Convert the scene graph into a format readable by the grounder
        extracted_locations_dictionary = convert_JSON_to_locations_dictionary(pruned_sg)

        if plan is not None:       
            print_cyan("\nGrounding started...")
            
            # Use VAL to just validate the plan
            val_succesful, val_log = VAL_validate(domain_file_path, sub_goal_dir, output_plan_file_path)

            # If this experiment requires it, try grounding the plan in the real scene graph
            if GROUND_IN_SCENE_GRAPH:
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

                # If we achieved 100% grounding success, we can break the loop as we correctly achieved the original goal
                if not grounding_succesful:
                    plans.append(plan)
                    return domain_file_path, pruned_sg, problem_pddl_path, sub_goals_file_paths, False, False, None, "PLANNING", planner_error_log
                    break
        else:
            return domain_file_path, pruned_sg, problem_pddl_path, sub_goals_file_paths, False, False, None, "PLANNING", planner_error_log
                

        plans.append(plan)

        new_pddl_env = old_pddl_env

    return domain_file_path, pruned_sg, problem_pddl_path, sub_goals_file_paths, planning_succesful, grounding_succesful, plans, reason_for_failure
