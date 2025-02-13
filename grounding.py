import os 
import json
from pprint import pprint
from pathlib import Path

from pddlgym.core import InvalidAction


from planner import run_planner_FD, execute_plan, execute_action, initialize_pddl_environment
from utils import load_knowledge_graph, read_graph_from_path, get_verbose_scene_graph



# Grounds a subplan in a specific room
def verify_subplan_groundability(pddlgym_environment, locations_dictionary, subplan, location):
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
        #print(f"Move action: {move_action}")
    
        # Find the action arguments
        from_location, to_location = extract_locations_from_movement_action(move_action)
        print(f"Moving from {from_location} to {to_location}")

        # Find the from_location of the current subplan
        if from_location not in locations_dictionary.values():
            print(f"Error: Location {from_location} not found in scene graph")
            return 0, "", from_location
        
        #Find the to_location of the current subplan
        if to_location not in locations_dictionary.values():
            print(f"Error: Location {to_location} not found in scene graph")
            return 0, "", to_location
        
        # Perform a PDDL gym step to move the robot
        pddlgym_environment, obs = execute_action(pddlgym_environment, move_action)

        # Update the locations_dictionary (useful if the robot was carrying something with him while moving)
        locations_dictionary, hallucinated_obj, hallucinated_loc = update_locations_dictionary(locations_dictionary, obs)

        if hallucinated_obj or hallucinated_loc:
            return 0, hallucinated_obj, hallucinated_loc
        
        location = to_location
        successful_actions += 1



    # Collect all objects of locations_dictionary that have the value location
    #objects_in_room = []
    #for obj, loc in locations_dictionary.items():
    #    if loc == location:
    #        objects_in_room.append(obj)
    #print(f"Objects in room: {objects_in_room}")

    objects_to_find = []
    for action in subplan["actions"]:
        action = str(action)
        #action_args = [arg.split(":")[0] for arg in action.split("(")[1].split(",")]
        #print(f"Action arguments: {action_args}")

        #for arg in action_args:
        #    if arg!=location and not arg in objects_in_room:
        #        print("Grounding failed for action "+action+" because "+arg+" could not be found in "+subplan["location"]+".")
        #        return successful_actions, arg, subplan["location"]

        try:
            pddlgym_environment, obs = execute_action(pddlgym_environment, move_action)
        except InvalidAction as e:
            print("Grounding failed for action "+action+" because: '"+str(e)+"'")
            return successful_actions, "", ""
            
        successful_actions += 1


    #print("\tGrounding successful")
    return successful_actions, "", ""



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

    #print(action_args)

    from_location = action_args[1].split(":")[0]
    to_location = action_args[2].split(":")[0]

    return from_location, to_location



# Given a scene graph, extract a dictionary object -> location
def extract_locations_dictionary(graph):
    locations = {}

    for location, objects in graph.items():
        for obj in objects:
            obj_name = obj[0]
            locations[obj_name] = location
    
    return locations



def update_locations_dictionary(locations_dictionary, new_environment_state, location_relation_str = "at", location_variable_type = "room"):
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

            #print("PDDL obj: " + str(pddl_object))
            #print("New location: " + str(new_location))

            # Check that the PDDLgym state objects are present in the locations dictionary, 
            # otherwise they are the result of hallucinations in the problem generation step
            if pddl_object not in new_locations_dictionary.keys():
                return new_locations_dictionary, hallucinated_obj, ""

            # Check that the PDDLgym state locations are present in the locations dictionary, 
            # otherwise they are the result of hallucinations in the problem generation step
            if new_location not in new_locations_dictionary.values():
                return new_locations_dictionary, "", hallucinated_loc
                
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
    #print(obs.literals)
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



def verify_groundability(plan, graph, domain_file_path, problem_dir, move_action_str, location_relation_str, location_type_str, initial_robot_location):
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


    # Initialize PDDLgym environment and obtain the first observation
    pddlgym_environment, initial_observation = initialize_pddl_environment(domain_file_path, problem_dir)

    #pprint(initial_observation.literals)

    # Find initial robot location from initial observation of the PDDLGym environment (the initial location of the robot in the PDDL problem)
    initial_PDDL_robot_location, robot_name = find_robot_location(initial_observation, location_relation_str)
    #print(str(initial_PDDL_robot_location), " ", str(robot_name))
    
    # Check that the initial robot location is specified in the PDDL problem
    if initial_PDDL_robot_location is None:
        print(f"Grounding failed because the initial robot location is not specified in the PDDL problem")
        return 0, robot_name, ""
    
    # Check that the PDDL problem contains the correct robot location
    if initial_PDDL_robot_location != initial_robot_location:
        print(f"Grounding failed because the robot location in the PDDL problem is not the requested one (desired: {initial_robot_location}, found: {initial_PDDL_robot_location})")
        return 0, robot_name, initial_robot_location

    # Initialize the first empty subplan
    current_subplan = {
        "move_action": "",
        "location": initial_robot_location,
        "actions": []
        }

    # Extract a "object -> location" map
    locations_dictionary = extract_locations_dictionary(graph)

    # Explicitly add the robot to the locations_dictionary at its initial_robot_location
    if robot_name not in locations_dictionary:
        locations_dictionary[robot_name] = initial_robot_location

    #pprint(locations_dictionary)
    
    # [Problem hallucination checks]

    # Check that the initial location exists in the locations dictionary
    if not initial_robot_location in locations_dictionary.values():
        return 0, "", initial_robot_location

    # Check that the initial location in the initial PDDLgym state coincides with the initial location in the knowledge graph
    # Look for the robot
    for obj, loc in locations_dictionary.items():
        if "robot" in obj:
            if loc != initial_robot_location:
                print("Grounding failed because initial robot location in the PDDL problem was different from the location in the knowledge graph.")
                return 0, obj, loc


    #pprint(plan)
    

    # Subdivide plan into subplans, using movement actions as splitting criterion
    for action in plan:
        if move_action_str in str(action):
            if current_subplan["actions"]:
                subplans.append(current_subplan)

            from_location, to_location = extract_locations_from_movement_action(action)

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
        successful_actions, failure_object, failure_room = verify_subplan_groundability(pddlgym_environment, locations_dictionary, subplan, current_location)

        print("successful_actions: "+str(successful_actions))
        print("failure_object: "+str(failure_object))
        print("failure_room: "+str(failure_room))

        if not successful_actions:
            return grounding_percentage, failure_object, failure_room
        else:
            total_successful_actions += successful_actions
            grounding_percentage = total_successful_actions / len(plan)

            if successful_actions < len(subplan["actions"]) + 1 if subplan["move_action"] else 0:
                return grounding_percentage, failure_object, failure_room


    return grounding_percentage, "", ""


def plan_and_ground(problem_dir):

    knowledge_graph = load_knowledge_graph(os.path.join(problem_dir, "kg.json"))

    initial_location = open(os.path.join(problem_dir, "init_loc.txt", "r").read())

    # Generate plan
    plan = run_planner_FD(DOMAIN_FILE_PATH, problem_dir)

    pprint(plan)

    if plan is None:
        print("Could not generate plan")
        return

    # Perform grounding
    grounding_success_percentage, failure_object, failure_room = verify_groundability(
        plan, 
        knowledge_graph, 
        domain_file_path=DOMAIN_FILE_PATH, 
        problem_dir=problem_dir, 
        move_action_str="walk",
        initial_robot_location=initial_location
    )

    return grounding_success_percentage, failure_object, failure_room, plan



if __name__ == "__main__":


    def perform_test(problem_dir):

        #knowledge_graph = load_knowledge_graph(os.path.join(problem_dir, "kg.json"))

        # Look for a file with extension .npz, get its path
        path_graph = ""
        for file in os.listdir(problem_dir):
            if file.endswith(".npz"):
                path_graph = os.path.join(problem_dir, file)
                break

        scene_graph = read_graph_from_path(Path(path_graph))
        knowledge_graph = get_verbose_scene_graph(scene_graph, as_string=False)
        #pprint(knowledge_graph)

        # Generate plan
        plan = run_planner_FD(DOMAIN_FILE_PATH, problem_dir)

        pprint(plan)

        if plan is None:
            print("Could not generate plan")
            return

        # Perform grounding
        grounding_success_percentage, failure_object, failure_room = verify_groundability(
            plan, 
            knowledge_graph, 
            domain_file_path=DOMAIN_FILE_PATH, 
            problem_dir=problem_dir, 
            move_action_str="walk", 
            location_relation_str="at",
            location_type_str="room"
        )

        print(grounding_success_percentage, failure_object, failure_room)

    TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    GROUNDING_TEST_DIR = os.path.join(TEST_DIR, "grounding")
    DOMAIN_FILE_PATH = os.path.join(GROUNDING_TEST_DIR, "domain.pddl")



    # First test: robot and vacuum in dining room
    perform_test(os.path.join(GROUNDING_TEST_DIR, "problem_1"))

    # Second test: robot in kitchen room and vacuum in dining room (task in dining room)
    perform_test(os.path.join(GROUNDING_TEST_DIR, "problem_2"))

    # Third test: robot in dining room and vacuum in kitchen (task in dining room)
    perform_test(os.path.join(GROUNDING_TEST_DIR, "problem_3"))

    # Fourth test: robot in dining room and vacuum in kitchen, missing objects in scene graph but not in problem(task in dining room)
    #perform_test(os.path.join(GROUNDING_TEST_DIR, "problem_4"))
