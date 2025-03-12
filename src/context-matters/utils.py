import re
import json
import os
import numpy as np
import traceback

from typing import Dict, Set, Optional
from collections import defaultdict
from pathlib import Path


def filter_graph(graph: Dict, labels: Set[str]) -> Dict:
    """
    Filter the 3DSG graph to only include objects with labels in the 'labels' set

    :param graph: Dictionary containing the 3DSG
    :param labels: Set of labels (strings) to consider, e.g. {"room", "object"}
    :return new_graph: Dictionary containing the filtered 3DSG
    """
    new_graph = {}
    for key, item in graph.items():
        if key in labels:
            new_graph[key] = item
    return new_graph


def save_graph(graph: Dict, path: str):
    """
    Save the 3DSG graph to a file

    :param graph: Dictionary containing the 3DSG
    :param path: Path to the file
    """
    np.savez(path, output=graph)


def read_graph_from_path(path: Path) -> Dict:
    """
    Read 3DSG from file (.npz) and returns it stored in a dictionary

    :param path: Path to the .npz file
    :return: Dictionary containing the 3DSG
    """

    assert isinstance(path, Path), "Input file is not a Path"
    assert str(path).endswith(".npz"), "Input file is not .npz object"

    graph = np.load(path, allow_pickle=True)["output"].item()

    keeps = set(["object", "room"])
    graph = filter_graph(graph, keeps)

    return graph


def get_verbose_scene_graph(graph: Dict, as_string=True) -> str:
    """
    Given a 3DSG, return a verbose description of the scene graph with meaningful information,
    including room names, objects, and their descriptions.

    :param graph: Dictionary containing the 3DSG
    :return: String with the verbose scene graph
    """

    rooms = graph.get("room", {})
    objects = graph.get("object", {})

    # 1. Create a label for each room: always append "_roomId".
    room_id_to_label = {
        r_id: f"{room_info.get('scene_category', 'UnnamedRoom')}_{r_id}"
        for r_id, room_info in rooms.items()
    }

    # 2. Create a label for each object: always append "_objId".
    obj_id_to_label = {
        o_id: f"{obj_info.get('class_', 'UnnamedObject')}_{o_id}"
        for o_id, obj_info in objects.items()
    }

    # 3. Group objects by their roomId.
    room_to_objects = defaultdict(list)
    for o_id, obj_info in objects.items():
        r_id = obj_info.get("parent_room")
        if r_id in rooms:
            obj_label = obj_id_to_label[o_id].replace("  ", " ").replace(" ", "_")
            obj_description = obj_info.get("description", "No description available")
            room_to_objects[r_id].append((obj_label, obj_description))

    # 4. Construct the output string.
    if as_string:
        lines = []
        for r_id in rooms:
            room_label = room_id_to_label[r_id]
            objs_in_room = room_to_objects.get(r_id, [])
            obj_lines = []
            if objs_in_room:
                for obj in objs_in_room:
                    object_name = obj[0].replace(" ", "_")
                    object_description = obj[1]
                    obj_lines.append(f"{object_name} - {object_description}")

                objs_str = "\n  - " + "\n  - ".join(obj_lines)
            else:
                objs_str = "  - No objects"
            lines.append(f"\n\n{room_label}:{objs_str}")

        return "\n".join(lines)
    else:
        output = {}
        for r_id in rooms:
            room_label = room_id_to_label[r_id]
            objs_in_room = room_to_objects.get(r_id, [])
            output[room_label] = objs_in_room
        return output


def copy_file(src: str, dst: str):
    """
    Copy a file from src to dst

    :param src: Source file path
    :param dst: Destination file path
    """
    with open(src, "r") as f:
        data = f.read()

    with open(dst, "w") as f:
        f.write(data)


def save_file(data: str, path: str):
    """
    Save a string to a file

    :param data: String to save
    :param path: Path to the file
    """
    with open(path, "w") as f:
        f.write(data)


def save_statistics(
    phase,
    dir,
    workflow_iteration,
    pddl_refinement_iteration=None,
    plan_successful=None,
    pddlenv_error_log=None,
    planner_error_log=None,
    planner_statistics=None,
    VAL_validation_log=None,
    VAL_grounding_log=None,
    scene_graph_grounding_log=None,
    grounding_success_percentage=None,
    exception=None,
):

    data = {
        "plan_successful": plan_successful,
        "pddlenv_error_log": pddlenv_error_log,
        "planner_error_log": planner_error_log,
        "planner_statistics": planner_statistics,
        "VAL_validation_log": VAL_validation_log,
        "VAL_grounding_log": VAL_grounding_log,
        "scene_graph_grounding_log": scene_graph_grounding_log,
        "grounding_success_percentage": grounding_success_percentage,
    }

    if not os.path.exists(dir):
        os.makedirs(dir)

    stats_file = os.path.join(dir, "statistics.json")

    statistics = {}

    if not os.path.exists(stats_file):
        if phase is not None:
            statistics["statistics"] = {"0": {phase: {}}}

            if phase == "PDDL_REFINEMENT":
                statistics["statistics"]["0"][phase] = [data]
            else:
                statistics["statistics"]["0"][phase] = data
    else:
        with open(stats_file, "r") as f:
            statistics = json.load(f)

        if str(workflow_iteration) not in statistics["statistics"]:
            statistics["statistics"][str(workflow_iteration)] = {}

        if phase == "PDDL_REFINEMENT":
            if (
                "PDDL_REFINEMENT"
                not in statistics["statistics"][str(workflow_iteration)]
            ):
                statistics["statistics"][str(workflow_iteration)][phase] = []
            else:
                statistics["statistics"][str(workflow_iteration)][phase].append(data)
        else:
            statistics["statistics"][str(workflow_iteration)][phase] = data

    if exception is not None:
        statistics["exception"] = {
            "reason": str(exception),
            "traceback": "".join(traceback.format_tb(exception.__traceback__)),
            "workflow_iteration": workflow_iteration,
            "refinement_iteration": pddl_refinement_iteration,
            "phase": phase,
        }

    with open(stats_file, "w") as f:
        json.dump(statistics, f, indent=4)