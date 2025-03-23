import os 
from pathlib import Path
import subprocess
import re

from pddlgym.core import InvalidAction


from .planner import run_planner_FD, execute_action, initialize_pddl_environment
from .utils import read_graph_from_path, get_verbose_scene_graph

def convert_JSON_to_verbose_SG(scene_graph):
    locations = {}
    for node, data in scene_graph["nodes"].items():
        if "attributes" not in data.keys():
            continue
        assert "location" in data["attributes"]
        locations[node] = data["attributes"]["location"]
    
    return locations

def convert_JSON_to_locations_dictionary(scene_graph):
    locations = {}
    location_names = {}
    
    for room_id, room_data in scene_graph["room"].items():
        location_names[str(room_id)] = room_data["scene_category"]+"_"+str(room_id)
    
    for obj_id, obj_data in scene_graph["object"].items():
        if str(obj_data["parent_room"]) not in location_names.keys():
            print("Could not find location "+str(obj_data["parent_room"])+" for object "+obj_data["class_"]+"_"+str(obj_id))
            continue
        locations[obj_data["class_"]+"_"+str(obj_id)] = location_names[str(obj_data["parent_room"])]

    return locations

# Grounds a subplan in a specific room
def verify_subplan_groundability(pddlgym_environment, graph, locations_dictionary, subplan, location):
    """
    Grounds a subplan in a specific room.

    Args:
        pddlgym_environment: The PDDL gym environment.
        locations_dictionary: Dictionary mapping objects to locations.
        subplan: The subplan to verify.
        location: The current location.

    Returns:
        Tuple containing the number of successful actions, the hallucinated object (if any), and the hallucinated location (if any).
    """
    successful_actions = 0

    # Find the move action in the subplan
    move_action = subplan["move_action"]
    if move_action:
    
        # Find the action arguments
        from_location, to_location = extract_locations_from_movement_action(move_action)
        print(f"Moving from {from_location} to {to_location}")

        # Find the from_location of the current subplan
        if from_location not in graph.keys():
            print(f"Error: Location {from_location} not found in scene graph")
            return 0, f"Location {from_location} not found in scene graph"
        
        #Find the to_location of the current subplan
        if to_location not in graph.keys():
            print(f"Error: Location {to_location} not found in scene graph")
            return 0, f"Location {to_location} not found in scene graph"
        
        # Perform a PDDL gym step to move the robot
        pddlgym_environment, obs = execute_action(pddlgym_environment, move_action)

        # Update the locations_dictionary (useful if the robot was carrying something with him while moving)
        locations_dictionary, hallucinated_obj, hallucinated_loc = update_locations_dictionary(graph, locations_dictionary, obs)

        if hallucinated_obj or hallucinated_loc:
            return 0, f"Hallucinated object: {hallucinated_obj}, Hallucinated location: {hallucinated_loc}"
        
        successful_actions += 1



    for action in subplan["actions"]:

        try:
            pddlgym_environment, obs = execute_action(pddlgym_environment, action)
        except InvalidAction as e:
            print("Grounding failed for action "+action+" because: '"+str(e)+"'")
            return successful_actions, f"Grounding failed for action {action} because: '{str(e)}'"
            
        successful_actions += 1

    return successful_actions, ""



def extract_locations_from_movement_action(action):
    """
    Extracts the from and to locations from a movement action.

    Args:
        action: The movement action.

    Returns:
        Tuple containing the from location and the to location.
    """
    if action is not str:
        action = str(action)

    action_args = action.split("(")[1].split(",")

    from_location = action_args[1].split(":")[0]
    to_location = action_args[2].split(":")[0]

    return from_location, to_location



# Given a scene graph, extract a dictionary object -> location
def extract_locations_dictionary(graph):
    locations = {}

    for location, objects in graph.items():
        for obj in objects:
            obj_name = obj[0].replace(" ", "_")
            location = location.replace(" ", "_")
            locations[obj_name] = location
    
    return locations



def update_locations_dictionary(graph, locations_dictionary, new_environment_state, location_relation_str = "at", location_variable_type = "room"):
    """
    Updates the locations dictionary based on the new environment state.

    Args:
        locations_dictionary: The current locations dictionary.
        new_environment_state: The new environment state.
        location_relation_str: The string representing the location relation.
        location_variable_type: The type of the location variable.

    Returns:
        Tuple containing the updated locations dictionary, the hallucinated object (if any), and the hallucinated location (if any).
    """
    new_locations_dictionary = locations_dictionary.copy()

    # Extract all relations in the State that contain the location_relation_str
    for literal in new_environment_state.literals:

        predicate_name = literal.predicate.name

        # Determine if the literal is a location literal
        if location_relation_str == predicate_name or\
            "_"+location_relation_str in predicate_name or\
                location_relation_str+"_" in predicate_name:
            
            # Make sure that the location is the second variable in the location relation
            assert location_variable_type in str(literal.variables[1])

            pddl_object = str(literal.variables[0]).split(":")[0]
            new_location = str(literal.variables[1]).split(":")[0]

            # Check that the PDDLgym state objects are present in the locations dictionary, 
            # otherwise they are the result of hallucinations in the problem generation step
            if pddl_object not in new_locations_dictionary.keys():
                return new_locations_dictionary, pddl_object, ""

            # Check that the PDDLgym state locations are present in the locations dictionary, 
            # otherwise they are the result of hallucinations in the problem generation step
            if new_location not in graph.keys():
                return new_locations_dictionary, "", new_location
                
            new_locations_dictionary[pddl_object] = new_location

    return new_locations_dictionary, "", ""



def find_robot_location(obs, location_relation_str, location_type_str="room", robot_type_str="robot"):
    """
    Finds the initial robot location from the initial observation.

    Args:
        obs: The initial observation.
        location_relation_str: The string representing the location relation.
        location_type_str: The type of the location variable.

    Returns:
        The initial robot location.
    """
    # Find initial robot location from initial observation
    for literal in obs.literals:
        predicate_name = str(literal.predicate)

        # If the location_relation_str is in the predicate name
        if location_relation_str == predicate_name or\
            "_"+location_relation_str in predicate_name or\
                location_relation_str+"_" in predicate_name:
            
            # Iterate over the predicate arguments, verifying that the location predicate contains
            # a robot argument and the robot robot location
            location = ""
            is_robot_location_predicate = False
            for variable in literal.variables:
                var_components = str(variable).split(":")

                if location_type_str == var_components[1]:
                    location = var_components[0]
                    
                if robot_type_str == var_components[1]:
                    robot_name = var_components[0]
                    is_robot_location_predicate = True
                
                if location:
                    if is_robot_location_predicate:
                        assert location, "There should be a location in a predicate expressing the robot location"
                        return location, robot_name

    return None, None



def verify_groundability_in_scene_graph(
    plan, 
    graph, 
    domain_file_path, 
    problem_dir, 
    move_action_str, 
    location_relation_str, 
    location_type_str, 
    initial_robot_location,
    pddlgym_environment=None,
    locations_dictionary=None
):
    """
    Performs grounding of a plan.

    Args:
        plan: The plan to ground.
        graph: The scene graph.
        domain_file_path: The path to the domain file.
        problem_dir: The directory containing the problem files.
        move_action_str: The string representing the move action.
        location_relation_str: The string representing the location relation.
        location_type_str: The type of the location variable.

    Returns:
        Tuple containing the grounding success percentage, the failure object (if any), and the failure room (if any).
    """
    # Divide plan (list of strings each representing an action like (find_table(robot_1:robot,table_1:furniture,dining:location)) into subplans, separated by a move_action
    subplans = []

    if pddlgym_environment is None:
        # Initialize PDDLgym environment and obtain the first observation
        pddlgym_environment, initial_observation = initialize_pddl_environment(domain_file_path, problem_dir)
    else:
        problem_index = 0
        # Use only first problem in directory
        pddlgym_environment.fix_problem_index(problem_index)

        # Produce initial observation
        initial_observation, _ = pddlgym_environment.reset()

    # Find initial robot location from initial observation of the PDDLGym environment (the initial location of the robot in the PDDL problem)
    initial_PDDL_robot_location, robot_name = find_robot_location(initial_observation, location_relation_str)
    
    # Check that the initial robot location is specified in the PDDL problem
    if initial_PDDL_robot_location is None:
        print(f"Grounding failed because the initial robot location is not specified in the PDDL problem")
        return 0, f"Grounding failed because the initial robot location is not specified in the PDDL problem"
    
    # Check that the PDDL problem contains the correct robot location
    if initial_PDDL_robot_location != initial_robot_location:
        print(f"Grounding failed because the robot location in the PDDL problem is not the requested one (desired: {initial_robot_location}, found: {initial_PDDL_robot_location})")
        return 0, f"Grounding failed because the robot location in the PDDL problem is not the requested one (desired: {initial_robot_location}, found: {initial_PDDL_robot_location})"

    # Initialize the first empty subplan
    current_subplan = {
        "move_action": "",
        "location": initial_robot_location,
        "actions": []
        }

    # Extract a "object -> location" map
    if locations_dictionary is None:
        locations_dictionary = extract_locations_dictionary(graph)

    # Explicitly add the robot to the locations_dictionary at its initial_robot_location
    if robot_name not in locations_dictionary:
        locations_dictionary[robot_name] = initial_robot_location

    print(locations_dictionary)
    
    # [Problem hallucination checks]

    # Check that the initial location exists in the locations dictionary
    if not initial_robot_location in locations_dictionary.values():
        return 0, f"Grounding failed because initial robot location in the PDDL problem was different from the location in the knowledge graph."
    # Check that the initial location in the initial PDDLgym state coincides with the initial location in the knowledge graph
    # Look for the robot
    for obj, loc in locations_dictionary.items():
        if "robot" in obj:
            if loc != initial_robot_location:
                print("Grounding failed because initial robot location in the PDDL problem was different from the location in the knowledge graph.")
                return 0, f"Grounding failed because initial robot location in the PDDL problem was different from the location in the knowledge graph."

    # Check that all the objects in the initial PDDLgym state are present in the knowledge graph
    # (to make sure we're not introducing any hallucination in the planning process)
    for literal in initial_observation.literals:
        for variable in literal.variables:

            var_components = str(variable).split(":")
            # In case variable is a room, check that it is in the keys of the location -> object dictionary
            if var_components[1] == location_type_str and str(var_components[0]) not in locations_dictionary.values():
                print("Grounding failed because object "+str(var_components[0])+" was not found in the knowledge graph.")
                return 0, f"Grounding failed because object {str(var_components[0])} was not found in the knowledge graph."
            
            # In case variable is not a room, check that it is in the values of the location -> object dictionary
            if var_components[1] != location_type_str and str(var_components[0]) not in locations_dictionary.keys():
                print("Grounding failed because object "+str(var_components[0])+" was not found in the knowledge graph.")
                return 0, f"Grounding failed because object {str(var_components[0])} was not found in the knowledge graph."
            

    

    # Subdivide plan into subplans, using movement actions as splitting criterion
    for action in plan:
        if move_action_str in str(action):
            if current_subplan["actions"]:
                subplans.append(current_subplan)

            _, to_location = extract_locations_from_movement_action(action)

            current_subplan = {
                "move_action": action,
                "location": to_location,
                "actions": []
            }
        else:
            current_subplan["actions"].append(action)
    
    # Append final subplan
    if current_subplan["actions"]:
        subplans.append(current_subplan)
    

    # Ground each subplan, accumulate the success percentage, if any fail, return the failure object and room
    total_successful_actions = 0
    grounding_percentage = 0.0
    current_location = initial_robot_location
    for subplan in subplans:
        
        print("Verifying subplan: "+str(subplan))

        if subplan["move_action"]:
            current_location = subplan["location"]

        # Attempt grounding for the current subplan (the part of a plan happening in a single room)    
        successful_actions, reason_for_failure = verify_subplan_groundability(pddlgym_environment, graph, locations_dictionary, subplan, current_location)

        if not successful_actions:
            return grounding_percentage, reason_for_failure
        else:
            total_successful_actions += successful_actions
            grounding_percentage = total_successful_actions / len(plan)

            if successful_actions < len(subplan["actions"]) + 1 if subplan["move_action"] else 0:
                return grounding_percentage, reason_for_failure


    return grounding_percentage, ""


def check_state_consistency(final_state, initial_state):
    """
    Checks that the initial state of the next PDDL sub-problem and the final state of the previous one coincide.

    Args:
        final_state: The final state of the previous sub-problem.
        initial_state: The initial state of the next sub-problem.

    Returns:
        Boolean indicating whether the states are consistent.
    """
    return final_state.literals == initial_state.literals


def translate_plan(input_file, output_file=None):
    # Step 1: Read the input file
    with open(input_file, 'r') as file:
        plan_content = file.read().strip()

    # Step 2: Extract actions using regex
    pattern = re.compile(r'(\w+)\((.*?)\)')
    matches = pattern.findall(plan_content)

    translated_plan = []
    for action, params in matches:
        # Extract arguments and strip types (remove everything after ':')
        args = [arg.split(':')[0] for arg in params.split(',')]
        translated_action = f"({action} {' '.join(args)})"
        translated_plan.append(translated_action)

    if output_file is not None:
        # Step 3: Write to the output file
        with open(output_file, 'w') as file:
            file.write('\n'.join(translated_plan))

    print(f"Plan successfully translated and written to {output_file}")

    return translate_plan

def VAL_validate(domain_file_path, problem_file_path=None, plan_path=None):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    validate_path = os.path.join(BASE_DIR, "third-party", "VAL", "build", "linux64", "Release", "bin", "Validate")
    command = [validate_path, "-v", domain_file_path]
    if problem_file_path:
        command.append(problem_file_path)
    if plan_path:
        command.append(plan_path)
    
    result = subprocess.run(command, capture_output=True, text=True)
    output_text = result.stdout

    # Parse the validation output
    validation_successful = False

    # Check for a successful plan validation
    if plan_path is not None:
        if "Plan valid" in output_text:
            validation_successful = True
            goal_satisfied = True  # If the plan is valid, the goal is satisfied
    else:
        if "fail" not in output_text:
            validation_successful = True

    return validation_successful, output_text


def VAL_ground(domain_file_path, problem_file_path):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    validate_path = os.path.join(BASE_DIR, "third-party", "VAL", "build", "linux64", "Release", "bin", "Instantiate")
    command = [validate_path, domain_file_path, problem_file_path]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    output_text = result.stdout

    if "Errors Encountered" in output_text:
        return False, output_text
    else:
        return True, output_text

def VAL_parse(domain_file_path, problem_file_path=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    validate_path = os.path.join(BASE_DIR, "third-party", "VAL", "build", "linux64", "Release", "bin", "Parser")
    command = [validate_path, domain_file_path]
    if problem_file_path:
        command.append(problem_file_path)
    
    result = subprocess.run(command, capture_output=True, text=True)
    result_str = result.stdout.strip().split("\n")

    if "Errors: 0" in result_str[-1]:
        return True, ""
    else:    
        return False, result_str[-1]