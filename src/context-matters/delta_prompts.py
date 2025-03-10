import json
import os
from pathlib import Path
from pprint import pprint


from agent import llm_call
from utils import print_blue, read_graph_from_path, filter_graph
from pddl_generation import _save_prompt_response

def generate_pddl_domain(task_file, domain_description, logs_dir=None, model="gpt-4o"):
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


def prune_scene_graph(scene_graph_path, goal_description_path, initial_robot_location=None, logs_dir=None, model="gpt-4o"):
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
                    'volume': 5.21398719237, 
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
    pruned_sg = llm_call(prompt, question, model=model)

    # Remove ```json and ``` from the response
    pruned_sg = pruned_sg.replace("`", "").replace("json", "").replace("lisp", "")

    #pprint(pruned_sg)

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

def generate_pddl_problem(pruned_scene_graph, goal_description_path, domain_pddl_path, initial_robot_location=None, logs_dir=None, model="gpt-4o"):
    print_blue("Generating PDDL problem...")

    prompt = """
    Role: You are an excellent problem generator. Given a Scene Graph (SG) and a desired goal,
    you can generate a PDDL problem file.
    We support the following subset of PDDL1.2:
        STRIPS
        Typing
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
                        'volume': 5.21398719237, 
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
                robot - robot
                kitchen_2 living_room_1 - room
                mop_11 - mop
                bucket_13 - grabbable
                glass_15 - grabbable
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
    answer = llm_call(prompt, question, model=model)

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

def decompose_pddl_goal(problem_pddl_path, domain_pddl_path, initial_robot_location=None, logs_dir=None, model="gpt-4o"):
    print_blue("Decomposing PDDL goal...")
    """
    Scompone gli obiettivi di un file PDDL del problema in una sequenza di sotto-obiettivi (sub-goals).
    """
    prompt = """
    Role: You are an excellent assistant in decomposing long-term goals. Given a PDDL problem file, 
    you can decompose the goal states into a sequence of sub-goals.

    Example:
    Given an example_problem.pddl, the goal states can be decomposed into a sequence of EXAMPLE_SUB_GOALS. Using the predicates defined in example_domain.pddl, the example subgoals can be formulated as: sub-goal_1.pddl, ..., sub-goal_n.pddl.
    
    
    example_problem.pddl:
        (define (problem house_cleaning)
        (:domain cleaning)
            (:objects
                robot - agent
                kitchen living_room - room
                mop - item
            )
            (:init
                (at robot kitchen)
                (dirty-floor kitchen)
                (dirty-floor living_room)
            )
            (:goal
                (and
                    (clean-floor kitchen)
                    (clean-floor living_room)
                    (is-clean mop)
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
                robot grabbable - locatable
                mop - grabbable
            )

            (:predicates
                (at ?something - locatable ?where - room)
                (is-holding ?who - robot ?something - grabbable)
                (is-free ?who - robot)
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



    EXAMPLE_SUB_GOALS:
    <subgoal>
    (:goal
        (and
            (floor-clean kitchen)
            (is-clean mop)
        )
    )
    <subgoal>
    (:goal
        (and
            (floor-clean kitchen)
            (is-clean mop)
        )
    )
    


    sub-goal_1.pddl:
        (define (problem sub-goal-1)
            (:domain cleaning)
            (:objects
                robot - agent
                kitchen living_room - room
                mop - item
            )
            (:init
                (at robot kitchen)
                (dirty-floor kitchen)
            )
            (:goal
                (and
                    (floor-clean kitchen)
                    (is-clean mop)
                )
            )
        )


    sub-goal_2.pddl:
        (define (problem sub-goal-2)
            (:domain cleaning)
            (:objects
                robot - agent
                kitchen living_room - room
                mop - item
            )
            (:init
                (at robot kitchen)
                (is-clean mop)
                (dirty-floor living_room)
            )
            (:goal
                (and
                    (dirty-floor living_room)
                    (is-clean mop)
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
    Your answer should be a collection of readily usable as a PDDL problems.
    Write only the PDDL problems, separated ONLY by the <subgoal> tag, WITHOUT ANY ADDITIONAL TEXT.
    The sub-goals should be readily parsable by a PDDL validator, without errors.
    """

    # Esegui la generazione dei sub-goal PDDL
    answer = llm_call(prompt, question, model=model)

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
