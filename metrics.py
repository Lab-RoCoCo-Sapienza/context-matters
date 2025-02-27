import csv
import json
from collections import defaultdict
from plots import plot_metrics
import os
import re

from pprint import pprint
from tables import generate_global_metrics_table

# TODO: recover exceptions and termination reasons from failed plans
# (for each False planning_successful, get the reason either from logs/error.txt or from the plan_N.out of the latest refinement of the latest iteration)

# TODO: planner statistics: In load_raw_data find the last "planner_statistics" entry of each "statistics" section of each problem in the raw_data for which plannin_successful and grounding_successful are True and use those statistics to compute the average total planning time, average expanded nodes and average plan length

def load_csv(filepath):
    """
    Load data from a CSV file and return it as a list of dictionaries.
    """
    data = []
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file, delimiter='|')
        for row in reader:
            data.append(row)
    return data

def find_last_planner_statistics(obj):
    # Recursively search for last instance of "planner_statistics"
    last = None
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "planner_statistics":
                last = value
            else:
                candidate = find_last_planner_statistics(value)
                if candidate is not None:
                    last = candidate
    elif isinstance(obj, list):
        for item in obj:
            candidate = find_last_planner_statistics(item)
            if candidate is not None:
                last = candidate
    return last

def load_raw_data(base_dir, model_id, models, tasks, add_whole_statistics=False):

    model_dir = os.path.join(base_dir, model_id)
    #print(model_dir)

    data = {}
    if not os.path.isdir(model_dir):
        print(f"{model_dir} not a directory")

    model_id = os.path.basename(model_dir)
    if model_id in models.keys():

        model_name = models[model_id] 
        data = {}



        # Iterate over tasks in the model directory
        for task_id in os.listdir(model_dir):

            if task_id not in tasks.keys():
                continue

            task_name = tasks[task_id]

            data[task_id] = {}

            # Load CSV data
            results_csv_path = os.path.join(model_dir, task_id + ".csv")
            if not os.path.exists(results_csv_path):
                print("CSV for model "+model_name+", for task "+task_id+" does not exits!")
                continue

            results_csv_data = load_csv(results_csv_path)
            # Save the data line-by-line
            for line in results_csv_data:
                assert line["Task"] == task_id

                if line["Scene"] not in data[task_id].keys():
                    data[task_id][line["Scene"]] = {}


                data[task_id][line["Scene"]][line["Problem"]] = line


    # Collect planning statistics
    for task_id, task_data in data.items():
        for scene_id, scene_data in task_data.items():
            for problem_id, problem_data in scene_data.items():
                
                
                statistics_path = os.path.join(base_dir, model_id, task_id, scene_id, problem_id,"statistics.json")

                statistics = None
                if os.path.exists(statistics_path):
                    with open(statistics_path, "r") as f:
                        statistics = f.read()
                    if add_whole_statistics:
                        try:
                            data[task_id][scene_id][problem_id]["statistics"] = statistics
                        except Exception as e:
                            print(model_id)
                            print(task_id)
                            print(scene_id)
                            print(problem_id)
                            raise


                    # Load planner statistics data
                    grounding_ok = (problem_data["Grounding Successful"].strip() == "True")
                    planning_ok = (problem_data["Planning Succesful"].strip() == "True")
                    if planning_ok and grounding_ok:
                        with open(statistics_path, "r") as f:
                            try:
                                stats_json = json.load(f)
                                # For delta models, aggregate over keys matching SUBGOAL_N:VAL:GROUNDING
                                if model_id.startswith("delta_"):
                                    aggregated = {"plan_length": 0, "total_time": 0, "num_node_expansions": 0}
                                    for key, value in stats_json.items():
                                        if re.match(r"SUBGOAL_\d+:VAL:GROUNDING", key):
                                            aggregated["plan_length"] += value.get("plan_length", 0)
                                            aggregated["total_time"] += value.get("total_time", 0)
                                            aggregated["num_node_expansions"] += value.get("num_node_expansions", 0)
                                    data[task_id][scene_id][problem_id]["planner_statistics"] = aggregated
                                else:
                                    print(re.findall(r"planner_statistics", str(stats_json)))
                                    planner_stats_value = find_last_planner_statistics(stats_json)
                                    if planner_stats_value is not None:
                                        data[task_id][scene_id][problem_id]["planner_statistics"] = planner_stats_value
                            except Exception as e:
                                print(f"Error parsing {statistics_path}: {e}")
                                raise

    print("\n\n\n")
    for task_id, task_data in data.items():
        for scene_id, scene_data in task_data.items():
            for problem_id, problem_data in scene_data.items():
                if "planner_statistics" in problem_data.keys():
                    print(data[task_id][scene_id][problem_id]["planner_statistics"])
    
    
    possibility_data = {}
    # Load possibility data
    possibility_csv_path = os.path.join(base_dir, "possibility.csv")
    possibility_csv_data = load_csv(possibility_csv_path)
    for line in possibility_csv_data:
        if line["Task"] not in possibility_data.keys():
            possibility_data[line["Task"]] = {}
        if line["Scene"] not in possibility_data[line["Task"]].keys():
            possibility_data[line["Task"]][line["Scene"]] = {}
        if line["Problem"] not in possibility_data[line["Task"]][line["Scene"]].keys():
            possibility_data[line["Task"]][line["Scene"]][line["Problem"]] = {}

        try:
            possibility_data[line["Task"]][line["Scene"]][line["Problem"]]["Task possible"] = line["Task possible"]
            #possibility_data[line["Task"]][line["Scene"]][line["Problem"]]["Possibility explanation"] = line["Possibility explanation"]
        except Exception as e:
            print(line)
            raise

    return data, possibility_data
                        
                        



def save_metrics(filepath, metrics):
    """
    Save the computed metrics to a JSON file.
    """
    with open(filepath, mode='w') as file:
        json.dump(metrics, file, indent=4)

def compute_global_metrics(raw_data, metrics, tasks, possibility_data=None):
    per_model_metrics = {}
    for model, model_data in raw_data.items():
        total_problems = 0
        planning_success_count = 0
        grounding_success_count = 0
        success_including_impossible = 0
        total_relaxations = 0.0
        total_refinements = 0.0
        count_CM = 0
        total_planning_time = 0.0
        total_expanded_nodes = 0
        total_plan_length = 0
        planning_stats_count = 0
        sum_refinements_per_iter = [0.0, 0.0, 0.0, 0.0]
        count_refinements_per_iter = [0, 0, 0, 0]
        possible_total = 0
        possible_success_count = 0

        for task, task_data in model_data.items():
            for scene, scene_data in task_data.items():
                for problem, problem_data in scene_data.items():
                    total_problems += 1
                    planning_ok = (problem_data.get("Planning Succesful", "").strip() == "True")
                    grounding_ok = (problem_data.get("Grounding Successful", "").strip() == "True")
                    if planning_ok:
                        planning_success_count += 1
                        if grounding_ok:
                            grounding_success_count += 1
                    if model.startswith("CM"):
                        relaxations = problem_data.get("Relaxations", "").strip()
                        refinements = problem_data.get("Refinements per iteration", "").strip()
                        if relaxations != "":
                            try:
                                total_relaxations += float(relaxations)
                            except ValueError:
                                pass
                        if refinements != "":
                            tokens = [tok.strip() for tok in refinements.split(";") if tok.strip() != ""]
                            for i, token in enumerate(tokens):
                                if i < 4:
                                    try:
                                        value = float(token)
                                        sum_refinements_per_iter[i] += value
                                        count_refinements_per_iter[i] += 1
                                    except ValueError:
                                        pass
                            try:
                                total_refinements += float(tokens[-1])
                            except (ValueError, IndexError):
                                pass
                        count_CM += 1

                    # Aggregate planning_statistics only for problems with successful planning and grounding
                    if planning_ok and grounding_ok and "planner_statistics" in problem_data:
                        stats = problem_data["planner_statistics"]
                        if isinstance(stats, dict):
                            total_expanded_nodes += stats.get("num_node_expansions", 0)
                            total_planning_time += stats.get("total_time", 0)
                            total_plan_length += stats.get("plan_length", 0)
                            planning_stats_count += 1

                    if possibility_data is not None:
                        poss = possibility_data[model][task][scene][problem]["Task possible"].strip()
                        if poss == "True":
                            possible_total += 1
                            if planning_ok and grounding_ok:
                                possible_success_count += 1
                    poss_status = (poss if possibility_data is not None else "True")
                    if poss_status == "True" and planning_ok and grounding_ok:
                        success_including_impossible += 1

        global_success_score = round((grounding_success_count / total_problems) * 100, 2) if total_problems > 0 else 0
        global_planning_success_score = round((planning_success_count / total_problems) * 100, 2) if total_problems > 0 else 0
        global_grounding_success_score = round((grounding_success_count / planning_success_count) * 100, 2) if planning_success_count > 0 else 0
        global_average_relaxations = round(total_relaxations / count_CM, 2) if count_CM > 0 else None
        global_average_refinements = round(total_refinements / count_CM, 2) if count_CM > 0 else None
        planning_success_but_grounding_failure = (round(((planning_success_count - grounding_success_count) / planning_success_count) * 100, 2) if planning_success_count > 0 else 0)
        success_rate_only_for_possible_problems = (round((possible_success_count / possible_total) * 100, 2) if possibility_data is not None and possible_total else 0)
        success_rate_with_impossible = round((success_including_impossible / total_problems) * 100, 2) if total_problems > 0 else 0

        average_refinements_per_relaxation_iteration = [
            round(sum_refinements_per_iter[i] / count_refinements_per_iter[i], 2) if count_refinements_per_iter[i] > 0 else None
            for i in range(4)
        ]
        avg_planning_time = round(total_planning_time / planning_stats_count, 4) if planning_stats_count > 0 else None
        avg_expanded_nodes = round(total_expanded_nodes / planning_stats_count, 2) if planning_stats_count > 0 else None
        avg_plan_length = round(total_plan_length / planning_stats_count, 2) if planning_stats_count > 0 else None

        per_model_metrics[model] = {
            "success_score": global_success_score,
            "planning_success_score": global_planning_success_score,
            "grounding_success_score": global_grounding_success_score,
            "average_relaxations": global_average_relaxations,
            "average_refinements": global_average_refinements,
            "planning_success_but_grounding_failure": planning_success_but_grounding_failure,
            "average_refinements_per_relaxation_iteration": average_refinements_per_relaxation_iteration,
            "success_rate_only_for_possible_problems": success_rate_only_for_possible_problems,
            "success_rate_with_impossible_problems_as_failed": success_rate_with_impossible,
            "average_planning_time": avg_planning_time,
            "average_expanded_nodes": avg_expanded_nodes,
            "average_plan_length": avg_plan_length
        }
    return per_model_metrics

def compute_per_model_per_task_metrics(raw_data, metrics, tasks, models, possibility_data=None):
    per_model_per_task = {}
    for model, model_data in raw_data.items():
        per_model_per_task[model] = {}
        for task, task_name in tasks.items():
            total_problems = 0
            planning_success_count = 0
            grounding_success_count = 0
            total_relaxations = 0.0
            total_refinements = 0.0
            count_CM = 0
            total_planning_time = 0.0
            total_expanded_nodes = 0
            total_plan_length = 0
            planning_stats_count = 0
            sum_refinements_per_iter = [0.0, 0.0, 0.0, 0.0]
            count_refinements_per_iter = [0, 0, 0, 0]
            possible_total = 0
            possible_success_count = 0
            success_including_impossible = 0

            if task in model_data:
                for scene, scene_data in model_data[task].items():
                    for problem, problem_data in scene_data.items():
                        total_problems += 1
                        planning_ok = (problem_data.get("Planning Succesful", "").strip() == "True")
                        grounding_ok = (problem_data.get("Grounding Successful", "").strip() == "True")
                        if planning_ok:
                            planning_success_count += 1
                            if grounding_ok:
                                grounding_success_count += 1
                        if model.startswith("CM"):
                            relaxations = problem_data.get("Relaxations", "").strip()
                            refinements = problem_data.get("Refinements per iteration", "").strip()
                            if relaxations:
                                try:
                                    total_relaxations += float(relaxations)
                                except ValueError:
                                    pass
                            if refinements:
                                tokens = [tok.strip() for tok in refinements.split(";") if tok.strip()]
                                for i, token in enumerate(tokens):
                                    if i < 4:
                                        try:
                                            value = float(token)
                                            sum_refinements_per_iter[i] += value
                                            count_refinements_per_iter[i] += 1
                                        except ValueError:
                                            pass
                                try:
                                    total_refinements += float(tokens[-1])
                                except (ValueError, IndexError):
                                    pass
                            count_CM += 1
                        # Accumulate planning statistics only if both planning and grounding succeed
                        if planning_ok and grounding_ok and "planner_statistics" in problem_data:
                            stats = problem_data["planner_statistics"]
                            if isinstance(stats, dict):
                                total_expanded_nodes += stats.get("num_node_expansions", 0)
                                total_planning_time += stats.get("total_time", 0)
                                total_plan_length += stats.get("plan_length", 0)
                                planning_stats_count += 1
                        if possibility_data is not None:
                            poss = possibility_data[model][task][scene][problem]["Task possible"].strip()
                            if poss == "True":
                                possible_total += 1
                                if planning_ok and grounding_ok:
                                    possible_success_count += 1
                        poss_status = (poss if possibility_data is not None else "True")
                        if poss_status == "True" and planning_ok and grounding_ok:
                            success_including_impossible += 1

            success_score = round((grounding_success_count / total_problems) * 100, 2) if total_problems else 0
            planning_success_score = round((planning_success_count / total_problems) * 100, 2) if total_problems else 0
            grounding_success_score = round((grounding_success_count / planning_success_count) * 100, 2) if planning_success_count else 0
            average_relaxations = round(total_relaxations / count_CM, 2) if count_CM else None
            average_refinements = round(total_refinements / count_CM, 2) if count_CM else None
            planning_success_but_grounding_failure = (round(((planning_success_count - grounding_success_count) / planning_success_count * 100), 2) if planning_success_count else 0)
            average_refinements_per_relaxation_iteration = [
                round(sum_refinements_per_iter[i] / count_refinements_per_iter[i], 2) if count_refinements_per_iter[i] else None
                for i in range(4)
            ]
            success_rate_only_for_possible_problems = (round((possible_success_count / possible_total) * 100, 2) if possibility_data is not None and possible_total else 0)
            success_rate_with_impossible = round((success_including_impossible / total_problems) * 100, 2) if total_problems > 0 else 0
            avg_planning_time = round(total_planning_time / planning_stats_count, 4) if planning_stats_count > 0 else None
            avg_expanded_nodes = round(total_expanded_nodes / planning_stats_count, 2) if planning_stats_count > 0 else None
            avg_plan_length = round(total_plan_length / planning_stats_count, 2) if planning_stats_count > 0 else None

            per_model_per_task[model][task] = {
                "success_score": success_score,
                "planning_success_score": planning_success_score,
                "grounding_success_score": grounding_success_score,
                "average_relaxations": average_relaxations,
                "average_refinements": average_refinements,
                "planning_success_but_grounding_failure": planning_success_but_grounding_failure,
                "average_refinements_per_relaxation_iteration": average_refinements_per_relaxation_iteration,
                "success_rate_only_for_possible_problems": success_rate_only_for_possible_problems,
                "success_rate_with_impossible_problems_as_failed": success_rate_with_impossible,
                "average_planning_time": avg_planning_time,
                "average_expanded_nodes": avg_expanded_nodes,
                "average_plan_length": avg_plan_length
            }
    return per_model_per_task

def compute_per_model_per_scene_metrics_ignore_tasks(raw_data, metrics, tasks, models, possibility_data=None):
    per_model_per_scene = {}
    for model, model_data in raw_data.items():
        scene_accum = {}
        scene_possible_total = {}
        scene_possible_success = {}
        for task, task_data in model_data.items():
            for scene, scene_data in task_data.items():
                if scene not in scene_accum:
                    scene_accum[scene] = {
                        "total_problems": 0,
                        "planning_success_count": 0,
                        "grounding_success_count": 0,
                        "total_relaxations": 0.0,
                        "total_refinements": 0.0,
                        "count_CM": 0,
                        "total_planning_time": 0.0,
                        "total_expanded_nodes": 0,
                        "total_plan_length": 0,
                        "planning_stats_count": 0
                    }
                    scene_possible_total[scene] = 0
                    scene_possible_success[scene] = 0
                for problem, problem_data in scene_data.items():
                    scene_accum[scene]["total_problems"] += 1
                    if problem_data.get("Planning Succesful", "").strip() == "True":
                        scene_accum[scene]["planning_success_count"] += 1
                        if problem_data.get("Grounding Successful", "").strip() == "True":
                            scene_accum[scene]["grounding_success_count"] += 1
                    if model.startswith("CM"):
                        relaxations = problem_data.get("Relaxations", "").strip()
                        refinements = problem_data.get("Refinements per iteration", "").strip()
                        if relaxations:
                            try:
                                scene_accum[scene]["total_relaxations"] += float(relaxations)
                            except ValueError:
                                pass
                        if refinements:
                            try:
                                scene_accum[scene]["total_refinements"] += float(refinements)
                            except ValueError:
                                pass
                        scene_accum[scene]["count_CM"] += 1
                    p_ok = (problem_data.get("Planning Succesful", "").strip() == "True")
                    g_ok = (problem_data.get("Grounding Successful", "").strip() == "True")
                    if p_ok and g_ok and "planner_statistics" in problem_data:
                        stats = problem_data["planner_statistics"]
                        if isinstance(stats, dict):
                            scene_accum[scene]["total_expanded_nodes"] += stats.get("num_node_expansions", 0)
                            scene_accum[scene]["total_planning_time"] += stats.get("total_time", 0)
                            scene_accum[scene]["total_plan_length"] += stats.get("plan_length", 0)
                            scene_accum[scene]["planning_stats_count"] += 1
                    if possibility_data is not None:
                        for t in scene_data:
                            poss = possibility_data.get(t, {}).get(scene, {}).get(problem, {}).get("Task possible", "True").strip()
                            if poss == "True":
                                scene_possible_total[scene] += 1
                                if (problem_data.get("Planning Succesful", "").strip() == "True" and 
                                    problem_data.get("Grounding Successful", "").strip() == "True"):
                                    scene_possible_success[scene] += 1
        per_model_per_scene[model] = {}
        for scene, counters in scene_accum.items():
            total = counters["total_problems"]
            planning = counters["planning_success_count"]
            grounding = counters["grounding_success_count"]
            count_CM = counters["count_CM"]
            success_score = round((grounding / total) * 100, 2) if total else 0
            planning_success_score = round((planning / total) * 100, 2) if total else 0
            grounding_success_score = round((grounding / planning) * 100, 2) if planning else 0
            average_relaxations = round(counters["total_relaxations"] / count_CM, 2) if count_CM else None
            average_refinements = round(counters["total_refinements"] / count_CM, 2) if count_CM else None
            planning_success_but_grounding_failure = (round(((planning - grounding) / planning) * 100, 2) if planning else 0)
            average_refinements_per_relaxation_iteration = None
            possible_total = scene_possible_total.get(scene, 0)
            possible_success = scene_possible_success.get(scene, 0)
            success_rate_only_for_possible_problems = (round((possible_success / possible_total) * 100, 2) if possibility_data is not None and possible_total else 0)
            ps_count = counters["planning_stats_count"]
            avg_planning_time = round(counters["total_planning_time"] / ps_count, 6) if ps_count > 0 else None
            avg_expanded_nodes = round(counters["total_expanded_nodes"] / ps_count, 2) if ps_count > 0 else None
            avg_plan_length = round(counters["total_plan_length"] / ps_count, 2) if ps_count > 0 else None

            per_model_per_scene[model][scene] = {
                "success_score": success_score,
                "planning_success_score": planning_success_score,
                "grounding_success_score": grounding_success_score,
                "average_relaxations": average_relaxations,
                "average_refinements": average_refinements,
                "planning_success_but_grounding_failure": planning_success_but_grounding_failure,
                "average_refinements_per_relaxation_iteration": average_refinements_per_relaxation_iteration,
                "success_rate_only_for_possible_problems": success_rate_only_for_possible_problems,
                "average_planning_time": avg_planning_time,
                "average_expanded_nodes": avg_expanded_nodes,
                "average_plan_length": avg_plan_length
            }
    return per_model_per_scene

if __name__ == "__main__":
    BASE_PATH = "/DATA/frozen_results"

    # Define models and their architectures
    models = {
        "CM_gpt-4o_gendomain_groundsg" : "CM + DG + SG",   # CM architecture example
        "CM_gpt-4o_gendomain_NOgroundsg" : "CM + DG",
        "CM_gpt-4o_NOgendomain_groundsg" : "CM + SG",
        "CM_gpt-4o_NOgendomain_NOgroundsg" : "CM",
        "delta_gpt-4o_gendomain_groundsg" : "DELTA + DG + SG",  # DELTA architecture example
        "delta_gpt-4o_gendomain_NOgroundsg" : "DELTA + DG",
        "delta_gpt-4o_NOgendomain_groundsg" : "DELTA + SG",
        "delta_gpt-4o_NOgendomain_NOgroundsg": "DELTA"
    }

    # Global metrics definitions:
    metrics = {
        "success_score": "SR",                                                                          # (Planning and Grounding Success Rate)
        "planning_success_score": "Planning SR",                                                        # % of problems with successful planning
        "grounding_success_score": "Scene Grounding SR",                                                # % of grounding successes for planned problems
        "average_relaxations": "Avg Rel.",                                                              # Average relaxations (CM only)
        "average_refinements": "Avg Ref.",                                                              # Average refinements (CM only)
        "planning_success_but_grounding_failure": "Planning SR but Grounding Failure",                  # % of problems with successful planning but grounding failure
        "average_refinements_per_relaxation_iteration": "Avg Ref. per Relax. Iter.",                    # Average refinements per relaxation iteration (CM only)
        "success_rate_only_for_possible_problems": "SR only for possible problems",                     # Success rate only for problems marked as possible
        "success_rate_with_impossible_problems_as_failed": "SR with impossible problems as failed",      # Success rate where impossible problems are considered failed
        "average_planning_time": "Avg Planning Time",                                                   # Average planning time
        "average_expanded_nodes": "Avg Expanded Nodes",                                                 # Average expanded nodes
        "average_plan_length": "Avg Plan Length"                                                        # Average plan length
    }

    all_tasks = {
                    "dining_setup": "DS",
                    "house_cleaning": "HC",
                    "office_setup": "OS",
                    "laundry": "LA",
                    "pc_assembly": "PC"
                }

    raw_data = {}
    possibility_raw_data = {}

    for model in models.keys():

        model_dir = os.path.join(BASE_PATH, model)

        if not os.path.exists(model_dir):
            print(f"Path not found: {model_dir}")
            continue
    
        # Compute metrics for the selected dataset splits
        raw_data[model], possibility_raw_data[model] = load_raw_data(BASE_PATH, model, models, tasks=all_tasks)

    #print(raw_data)
    #print(len(re.findall(r"planner_statistics", str(raw_data))))
    #raise

    # Compute global metrics (one metric spanning all models in raw_data and all selected_tasks and selected_scenes)
    global_metrics = compute_global_metrics(
        raw_data, 
        metrics = metrics, 
        tasks = all_tasks,
        possibility_data = possibility_raw_data
    )
    # Save metrics to JSON
    save_metrics(os.path.join(BASE_PATH,'global_metrics.json'), global_metrics)

    pprint(global_metrics)
    

    ##########################
    # PER-TASK/SCENE METRICS #
    ##########################


    # Compute per-task metrics (tasks on the rows, metrics on the columns)
    #per_task_metrics = compute_per_task_metrics(
    #    raw_data,
    #    metrics = average_metrics,
    #    tasks = all_tasks
    #)

    # Compute per-model per-task metrics (models on the rows, tasks on the columns, for each column, a column for each metric)
    per_model_per_task_metrics = compute_per_model_per_task_metrics(
        raw_data,
        metrics = metrics,
        tasks = all_tasks,
        models = models,
        possibility_data = possibility_raw_data
    )
    save_metrics(os.path.join(BASE_PATH,'per_model_per_task_metrics.json'), per_model_per_task_metrics)


    pprint(per_model_per_task_metrics)
    

    # Compute per-model per-scene metrics, ignoring tasks
    per_model_per_scene_metrics_ignore_tasks = compute_per_model_per_scene_metrics_ignore_tasks(
        raw_data,
        metrics = metrics,
        tasks = all_tasks,
        models = models,
        possibility_data = possibility_raw_data
    )
    save_metrics(os.path.join(BASE_PATH,'per_model_per_scene_metrics_ignore_tasks.json'), per_model_per_scene_metrics_ignore_tasks)

    #pprint(per_model_per_scene_metrics_ignore_tasks)

    # Compute per-model per-task metrics (models on the rows, tasks on the columns, for each column, a column for each metric)
    per_model_per_task_metrics = compute_per_model_per_task_metrics(
        raw_data,
        metrics = metrics,
        tasks = all_tasks,
        models = models,
        possibility_data = possibility_raw_data
    )
    save_metrics(os.path.join(BASE_PATH,'per_model_per_task_metrics.json'), per_model_per_task_metrics)

    pprint(per_model_per_task_metrics)
    

    # Compute per-model per-scene metrics, ignoring tasks
    per_model_per_scene_metrics_ignore_tasks = compute_per_model_per_scene_metrics_ignore_tasks(
        raw_data,
        metrics = metrics,
        tasks = all_tasks,
        models = models,
        possibility_data = possibility_raw_data
    )
    save_metrics(os.path.join(BASE_PATH,'per_model_per_scene_metrics_ignore_tasks.json'), per_model_per_scene_metrics_ignore_tasks)

    #pprint(per_model_per_scene_metrics_ignore_tasks)
