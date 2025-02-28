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
                                    #print(re.findall(r"planner_statistics", str(stats_json)))
                                    planner_stats_value = find_last_planner_statistics(stats_json)
                                    if planner_stats_value is not None:
                                        data[task_id][scene_id][problem_id]["planner_statistics"] = planner_stats_value
                            except Exception as e:
                                print(f"Error parsing {statistics_path}: {e}")
                                raise

    #print("\n\n\n")
    #for task_id, task_data in data.items():
    #    for scene_id, scene_data in task_data.items():
    #        for problem_id, problem_data in scene_data.items():
    #            if "planner_statistics" in problem_data.keys():
    #                print(data[task_id][scene_id][problem_id]["planner_statistics"])
    
    
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

# NEW FUNCTION: build intermediate count structure: MODEL -> TASK -> SCENE
def count_instances(raw_data, possibility_data=None):
    counts = {}
    for model, model_data in raw_data.items():
        counts[model] = {}
        for task, task_data in model_data.items():
            counts[model].setdefault(task, {})
            for scene, scene_data in task_data.items():
                # initialize counts per scene
                counts[model][task][scene] = {
                    "total_problems": 0,
                    "planning_success_count": 0,
                    "grounding_success_count": 0,
                    "total_relaxations": 0.0,
                    "total_refinements": 0.0,
                    "count_CM": 0,
                    "sum_refinements_per_iter": [0.0, 0.0, 0.0, 0.0],
                    "count_refinements_per_iter": [0, 0, 0, 0],
                    "total_planning_time": 0.0,
                    "total_expanded_nodes": 0,
                    "total_plan_length": 0,
                    "planning_stats_count": 0,
                    "possible_total": 0,
                    "possible_success_count": 0,
                    "success_including_impossible": 0,
                }
                for problem, problem_data in scene_data.items():
                    counts[model][task][scene]["total_problems"] += 1
                    planning_ok = (problem_data.get("Planning Succesful", "").strip() == "True")
                    grounding_ok = (problem_data.get("Grounding Successful", "").strip() == "True")
                    if planning_ok:
                        counts[model][task][scene]["planning_success_count"] += 1
                        if grounding_ok:
                            counts[model][task][scene]["grounding_success_count"] += 1
                    # For CM models: accumulate relaxations and refinements per iteration
                    if model.startswith("CM"):
                        relaxations = problem_data.get("Relaxations", "").strip()
                        refinements = problem_data.get("Refinements per iteration", "").strip()
                        if relaxations:
                            try:
                                counts[model][task][scene]["total_relaxations"] += float(relaxations)
                            except ValueError:
                                pass
                        if refinements:
                            tokens = [tok.strip() for tok in refinements.split(";") if tok.strip()]
                            for i, token in enumerate(tokens):
                                if i < 4:
                                    try:
                                        value = float(token)
                                        counts[model][task][scene]["sum_refinements_per_iter"][i] += value
                                        counts[model][task][scene]["count_refinements_per_iter"][i] += 1
                                    except ValueError:
                                        pass
                            try:
                                counts[model][task][scene]["total_refinements"] += float(tokens[-1])
                            except (ValueError, IndexError):
                                pass
                        counts[model][task][scene]["count_CM"] += 1
                    # Accumulate planner statistics if available
                    if planning_ok and grounding_ok and "planner_statistics" in problem_data:
                        stats = problem_data["planner_statistics"]
                        if isinstance(stats, dict):
                            counts[model][task][scene]["total_expanded_nodes"] += stats.get("num_node_expansions", 0)
                            counts[model][task][scene]["total_planning_time"] += stats.get("total_time", 0)
                            counts[model][task][scene]["total_plan_length"] += stats.get("plan_length", 0)
                            counts[model][task][scene]["planning_stats_count"] += 1
                    # Use possibility data if provided
                    if possibility_data is not None:
                        try:
                            poss = possibility_data[model][task][scene][problem]["Task possible"].strip()
                        except KeyError:
                            poss = "True"
                        if poss == "True":
                            counts[model][task][scene]["possible_total"] += 1
                            if planning_ok and grounding_ok:
                                counts[model][task][scene]["possible_success_count"] += 1
                        poss_status = poss if possibility_data is not None else "True"
                        if poss_status=="True" and planning_ok and grounding_ok:
                            counts[model][task][scene]["success_including_impossible"] += 1
                    else:
                        # Without possibility data, treat as possible by default.
                        if planning_ok and grounding_ok:
                            counts[model][task][scene]["success_including_impossible"] += 1
    return counts


# MODIFIED: Compute global metrics using count structure
def compute_global_metrics(count, metrics, tasks, possibility_data=None):

    per_model_metrics = {}
    for model, tasks_counts in counts.items():
        total_problems = 0
        grounding_success_count = 0
        planning_success_count = 0
        total_relaxations = 0.0
        total_refinements = 0.0
        count_CM = 0
        sum_refinements_per_iter = [0.0, 0.0, 0.0, 0.0]
        count_refinements_per_iter = [0, 0, 0, 0]
        total_planning_time = 0.0
        total_expanded_nodes = 0
        total_plan_length = 0
        planning_stats_count = 0
        possible_total = 0
        possible_success_count = 0
        success_including_impossible = 0

        for task in tasks_counts.keys():
            for scene, scene_counts in tasks_counts[task].items():
                total_problems += scene_counts["total_problems"]
                planning_success_count += scene_counts["planning_success_count"]
                grounding_success_count += scene_counts["grounding_success_count"]
                total_relaxations += scene_counts["total_relaxations"]
                total_refinements += scene_counts["total_refinements"]
                count_CM += scene_counts["count_CM"]
                for i in range(4):
                    sum_refinements_per_iter[i] += scene_counts["sum_refinements_per_iter"][i]
                    count_refinements_per_iter[i] += scene_counts["count_refinements_per_iter"][i]
                total_planning_time += scene_counts["total_planning_time"]
                total_expanded_nodes += scene_counts["total_expanded_nodes"]
                total_plan_length += scene_counts["total_plan_length"]
                planning_stats_count += scene_counts["planning_stats_count"]
                possible_total += scene_counts["possible_total"]
                possible_success_count += scene_counts["possible_success_count"]
                success_including_impossible += scene_counts["success_including_impossible"]

        print(total_problems)
        print(grounding_success_count)
        global_success_score = (grounding_success_count / total_problems)*100 if total_problems else 0
        global_planning_success_score = round((planning_success_count / total_problems)*100, 2) if total_problems else 0
        global_grounding_success_score = round((grounding_success_count / planning_success_count)*100, 2) if planning_success_count else 0
        global_average_relaxations = round(total_relaxations / count_CM, 2) if count_CM else None
        global_average_refinements = round(total_refinements / count_CM, 2) if count_CM else None
        planning_success_but_grounding_failure = round(((planning_success_count - grounding_success_count) / planning_success_count)*100, 2) if planning_success_count else 0
        success_rate_only_for_possible_problems = round((possible_success_count / possible_total)*100, 2) if possibility_data is not None and possible_total else 0
        success_rate_with_impossible = round((success_including_impossible / total_problems)*100, 2) if total_problems else 0
        average_refinements_per_relaxation_iteration = [
            round(sum_refinements_per_iter[i]/count_refinements_per_iter[i],2) if count_refinements_per_iter[i] else None
            for i in range(4)
        ]
        avg_planning_time = round(total_planning_time / planning_stats_count, 4) if planning_stats_count else None
        avg_expanded_nodes = round(total_expanded_nodes / planning_stats_count, 2) if planning_stats_count else None
        avg_plan_length = round(total_plan_length / planning_stats_count, 2) if planning_stats_count else None

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


# MODIFIED: Compute per-model per-task metrics using the intermediate count structure
def compute_per_model_per_task_metrics(count, metrics, tasks, models, possibility_data=None):

    per_model_per_task = {}
    for model, tasks_counts in counts.items():
        per_model_per_task[model] = {}
        for task in tasks.keys():
            # initialize task-level counts
            task_counts = {
                "total_problems": 0,
                "planning_success_count": 0,
                "grounding_success_count": 0,
                "total_relaxations": 0.0,
                "total_refinements": 0.0,
                "count_CM": 0,
                "sum_refinements_per_iter": [0.0, 0.0, 0.0, 0.0],
                "count_refinements_per_iter": [0, 0, 0, 0],
                "total_planning_time": 0.0,
                "total_expanded_nodes": 0,
                "total_plan_length": 0,
                "planning_stats_count": 0,
                "possible_total": 0,
                "possible_success_count": 0,
                "success_including_impossible": 0,
            }
            if task in tasks_counts:
                for scene, scene_counts in tasks_counts[task].items():
                    task_counts["total_problems"] += scene_counts["total_problems"]
                    task_counts["planning_success_count"] += scene_counts["planning_success_count"]
                    task_counts["grounding_success_count"] += scene_counts["grounding_success_count"]
                    task_counts["total_relaxations"] += scene_counts["total_relaxations"]
                    task_counts["total_refinements"] += scene_counts["total_refinements"]
                    task_counts["count_CM"] += scene_counts["count_CM"]
                    for i in range(4):
                        task_counts["sum_refinements_per_iter"][i] += scene_counts["sum_refinements_per_iter"][i]
                        task_counts["count_refinements_per_iter"][i] += scene_counts["count_refinements_per_iter"][i]
                    task_counts["total_planning_time"] += scene_counts["total_planning_time"]
                    task_counts["total_expanded_nodes"] += scene_counts["total_expanded_nodes"]
                    task_counts["total_plan_length"] += scene_counts["total_plan_length"]
                    task_counts["planning_stats_count"] += scene_counts["planning_stats_count"]
                    task_counts["possible_total"] += scene_counts["possible_total"]
                    task_counts["possible_success_count"] += scene_counts["possible_success_count"]
                    task_counts["success_including_impossible"] += scene_counts["success_including_impossible"]

            total_problems = task_counts["total_problems"]
            planning = task_counts["planning_success_count"]
            grounding = task_counts["grounding_success_count"]
            success_score = (grounding/total_problems)*100 if total_problems else 0
            planning_success_score = round((planning/total_problems)*100,2) if total_problems else 0
            grounding_success_score = round((grounding/planning)*100,2) if planning else 0
            average_relaxations = round(task_counts["total_relaxations"]/task_counts["count_CM"],2) if task_counts["count_CM"] else None
            average_refinements = round(task_counts["total_refinements"]/task_counts["count_CM"],2) if task_counts["count_CM"] else None
            planning_success_but_grounding_failure = round(((planning-grounding)/planning)*100,2) if planning else 0
            average_refinements_per_relaxation_iteration = [
                round(task_counts["sum_refinements_per_iter"][i]/task_counts["count_refinements_per_iter"][i],2) if task_counts["count_refinements_per_iter"][i] else None
                for i in range(4)
            ]
            possible_total = task_counts["possible_total"]
            possible_success = task_counts["possible_success_count"]
            success_rate_only_for_possible_problems = round((possible_success/possible_total)*100,2) if possibility_data is not None and possible_total else 0
            success_rate_with_impossible = round((task_counts["success_including_impossible"]/total_problems)*100,2) if total_problems else 0
            avg_planning_time = round(task_counts["total_planning_time"]/task_counts["planning_stats_count"],4) if task_counts["planning_stats_count"] else None
            avg_expanded_nodes = round(task_counts["total_expanded_nodes"]/task_counts["planning_stats_count"],2) if task_counts["planning_stats_count"] else None
            avg_plan_length = round(task_counts["total_plan_length"]/task_counts["planning_stats_count"],2) if task_counts["planning_stats_count"] else None

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
                "average_plan_length": avg_plan_length,
                "total_problems": total_problems
            }
    return per_model_per_task


# MODIFIED: Compute per-model per-scene metrics ignoring tasks using count structure
def compute_per_model_per_scene_metrics_ignore_tasks(count, metrics, tasks, models, possibility_data=None):
    
    per_model_per_scene = {}
    for model, tasks_counts in counts.items():
        per_model_per_scene[model] = {}
        # merge counts across tasks for each scene
        scene_merged = {}
        for task_counts in tasks_counts.values():
            for scene, scene_counts in task_counts.items():
                if scene not in scene_merged:
                    scene_merged[scene] = {key: scene_counts[key] if not isinstance(scene_counts[key], list) else scene_counts[key][:]
                                       for key in scene_counts}
                else:
                    scene_merged[scene]["total_problems"] += scene_counts["total_problems"]
                    scene_merged[scene]["planning_success_count"] += scene_counts["planning_success_count"]
                    scene_merged[scene]["grounding_success_count"] += scene_counts["grounding_success_count"]
                    scene_merged[scene]["total_relaxations"] += scene_counts["total_relaxations"]
                    scene_merged[scene]["total_refinements"] += scene_counts["total_refinements"]
                    scene_merged[scene]["count_CM"] += scene_counts["count_CM"]
                    for i in range(4):
                        scene_merged[scene]["sum_refinements_per_iter"][i] += scene_counts["sum_refinements_per_iter"][i]
                        scene_merged[scene]["count_refinements_per_iter"][i] += scene_counts["count_refinements_per_iter"][i]
                    scene_merged[scene]["total_planning_time"] += scene_counts["total_planning_time"]
                    scene_merged[scene]["total_expanded_nodes"] += scene_counts["total_expanded_nodes"]
                    scene_merged[scene]["total_plan_length"] += scene_counts["total_plan_length"]
                    scene_merged[scene]["planning_stats_count"] += scene_counts["planning_stats_count"]
                    scene_merged[scene]["possible_total"] += scene_counts["possible_total"]
                    scene_merged[scene]["possible_success_count"] += scene_counts["possible_success_count"]
                    scene_merged[scene]["success_including_impossible"] += scene_counts["success_including_impossible"]

        for scene, sc in scene_merged.items():
            total = sc["total_problems"]
            planning = sc["planning_success_count"]
            grounding = sc["grounding_success_count"]
            success_score = round((grounding/total)*100,2) if total else 0
            planning_success_score = round((planning/total)*100,2) if total else 0
            grounding_success_score = round((grounding/planning)*100,2) if planning else 0
            average_relaxations = round(sc["total_relaxations"]/sc["count_CM"],2) if sc["count_CM"] else None
            average_refinements = round(sc["total_refinements"]/sc["count_CM"],2) if sc["count_CM"] else None
            planning_success_but_grounding_failure = round(((planning-grounding)/planning)*100,2) if planning else 0
            average_refinements_per_relaxation_iteration = None  # unchanged as before
            possible_total = sc["possible_total"]
            possible_success = sc["possible_success_count"]
            success_rate_only_for_possible_problems = round((possible_success/possible_total)*100,2) if possibility_data is not None and possible_total else 0
            ps_count = sc["planning_stats_count"]
            avg_planning_time = round(sc["total_planning_time"]/ps_count,6) if ps_count else None
            avg_expanded_nodes = round(sc["total_expanded_nodes"]/ps_count,2) if ps_count else None
            avg_plan_length = round(sc["total_plan_length"]/ps_count,2) if ps_count else None

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

# NEW FUNCTION: Assert each CSV file (per task) has uniform scene line counts.
def check_csv_scene_line_counts(model_dir, tasks):
    for task_id in os.listdir(model_dir):
        if task_id not in tasks:
            continue
        csv_path = os.path.join(model_dir, task_id + ".csv")
        if not os.path.exists(csv_path):
            continue
        scene_counts = {}
        with open(csv_path, mode='r') as file:
            reader = csv.DictReader(file, delimiter='|')
            for row in reader:
                scene = row["Scene"]
                scene_counts[scene] = scene_counts.get(scene, 0) + 1
        if scene_counts:
            counts = set(scene_counts.values())
            if(len(counts) != 1): 
                print(f"CSV {csv_path} has uneven line counts per scene: {scene_counts}")

# Replace the old check_total_problems_across_models function with the following implementation:
def check_total_problems_across_models(raw_data, tasks):
    task_totals = {}
    for model, model_data in raw_data.items():
        task_totals[model] = {}
        for task in tasks.keys():
            if task in model_data:
                total = sum(len(problems) for scene, problems in model_data[task].items())
                task_totals[model][task] = total
    
    # Verify that each task has the same number of problems across all models
    for task in tasks.keys():
        totals = {model: task_totals[model].get(task, 0) for model in raw_data.keys()}
        if len(set(totals.values())) > 1:
            print(f"Task {task} has different problem counts across models:")
            print(totals)

    #print(task_totals)

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
        
        # NEW: Assert each task CSV for this model has uniform scene line counts 
        check_csv_scene_line_counts(model_dir, all_tasks)

        # Compute metrics for the selected dataset splits
        raw_data[model], possibility_raw_data[model] = load_raw_data(BASE_PATH, model, models, tasks=all_tasks)

    # Count instances
    counts = count_instances(raw_data, possibility_data=possibility_raw_data)

    # NEW: Check total problems consistency across models using raw_data.
    check_total_problems_across_models(raw_data, all_tasks)

    # Compute global metrics (one metric spanning all models in raw_data and all selected_tasks and selected_scenes)
    global_metrics = compute_global_metrics(
        counts, 
        metrics = metrics, 
        tasks = all_tasks,
        possibility_data = possibility_raw_data
    )
    # Save metrics to JSON
    save_metrics(os.path.join(BASE_PATH,'global_metrics.json'), global_metrics)

    pprint(global_metrics)
    

    print("\n\n\nTASK/SCENE")
    ##########################
    # PER-TASK/SCENE METRICS #
    ##########################

    # Compute per-model per-task metrics (models on the rows, tasks on the columns, for each column, a column for each metric)
    per_model_per_task_metrics = compute_per_model_per_task_metrics(
        counts,
        metrics = metrics,
        tasks = all_tasks,
        models = models,
        possibility_data = possibility_raw_data
    )
    save_metrics(os.path.join(BASE_PATH,'per_model_per_task_metrics.json'), per_model_per_task_metrics)


    pprint(per_model_per_task_metrics)
    

    # Verify that the average of ht

