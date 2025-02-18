import csv
import json
from collections import defaultdict
from plots import plot_metrics
import os

from pprint import pprint

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

def compute_metrics(data):
    """
    Compute various metrics from the data.
    """
    # Initialize metrics dictionary with default values
    metrics = {
        "global_success_score": 0,
        "success_score_per_scene": defaultdict(int),
        "success_score_per_problem_id": defaultdict(int),
        "success_score_per_task": defaultdict(int),
        "scene_success_score_per_task": defaultdict(lambda: defaultdict(int)),
        "average_relaxations_per_task": defaultdict(float),
        "average_relaxations_per_scene": defaultdict(float),
        "average_relaxations_per_problem_id": defaultdict(float),
        "average_refinements_per_task": defaultdict(float),
        "average_refinements_per_scene": defaultdict(float),
        "average_plan_length_per_task": defaultdict(float),
        "average_plan_length_per_scene": defaultdict(float),
        "global_average_relaxations": 0,
        "global_average_refinements": 0,
        "planning_success_but_grounding_failure": 0,
        "weighted_success_score_per_task": defaultdict(float),
        "weighted_success_score_per_scene": defaultdict(float),
        "weighted_success_score_per_problem_id": defaultdict(float),
        "global_weighted_success_score": 0
    }
    
    total_success = 0
    total_relaxations = 0
    total_refinements = 0
    total_plan_length = 0
    total_entries = len(data)
    total_weighted_success = 0
    
    # Count the number of entries per scene and per task
    scene_counts = defaultdict(int)
    task_counts = defaultdict(int)
    problem_counts = defaultdict(int)
    for entry in data:
        scene_counts[entry['Scene']] += 1
        task_counts[entry['Task']] += 1
        problem_counts[entry['Problem']] += 1
    
    # Iterate over each entry in the data
    for entry in data:
        task = entry['Task']
        scene = entry['Scene']
        problem_id = entry['Problem']
        planning_successful = entry['Planning Succesful'] == 'True'
        grounding_successful = entry['Grounding Successful'] == 'True'
        success = planning_successful and grounding_successful
        relaxations = int(entry['Relaxations'])
        refinements = sum(map(int, entry['Refinements per iteration'].split(';')))
        plan_length = int(entry['Plan Length'])
        
        # Calculate inverse path length
        inverse_path_length = 1 / plan_length if plan_length > 0 else 0
        
        # Update success-related metrics
        if success:
            total_success += 1
            metrics["success_score_per_scene"][scene] += 1
            metrics["success_score_per_problem_id"][problem_id] += 1
            metrics["success_score_per_task"][task] += 1
            metrics["scene_success_score_per_task"][scene][task] += 1
            metrics["weighted_success_score_per_task"][task] += inverse_path_length
            metrics["weighted_success_score_per_scene"][scene] += inverse_path_length
            metrics["weighted_success_score_per_problem_id"][problem_id] += inverse_path_length
            total_weighted_success += inverse_path_length
        
        # Update relaxation, refinement, and plan length-related metrics
        total_relaxations += relaxations
        total_refinements += refinements
        total_plan_length += plan_length
        metrics["average_relaxations_per_task"][task] += relaxations
        metrics["average_relaxations_per_scene"][scene] += relaxations
        metrics["average_relaxations_per_problem_id"][problem_id] += relaxations
        metrics["average_refinements_per_task"][task] += refinements
        metrics["average_refinements_per_scene"][scene] += refinements
        metrics["average_plan_length_per_task"][task] += plan_length
        metrics["average_plan_length_per_scene"][scene] += plan_length
        
        # Update planning success but grounding failure metric
        if planning_successful and not grounding_successful:
            metrics["planning_success_but_grounding_failure"] += 1
    
    # Calculate global success score as a percentage
    metrics["global_success_score"] = (total_success / total_entries) * 100
    metrics["global_average_relaxations"] = total_relaxations / total_entries
    metrics["global_average_refinements"] = total_refinements / total_entries
    metrics["global_weighted_success_score"] = (total_weighted_success / total_entries) * 100
    
    # Calculate success scores as percentages relative to the number of scenes or tasks
    for key in metrics["success_score_per_scene"]:
        metrics["success_score_per_scene"][key] = (metrics["success_score_per_scene"][key] / scene_counts[key]) * 100
    for key in metrics["success_score_per_problem_id"]:
        metrics["success_score_per_problem_id"][key] = (metrics["success_score_per_problem_id"][key] / problem_counts[key]) * 100
    for key in metrics["success_score_per_task"]:
        metrics["success_score_per_task"][key] = (metrics["success_score_per_task"][key] / task_counts[key]) * 100
    for scene in metrics["scene_success_score_per_task"]:
        for task in metrics["scene_success_score_per_task"][scene]:
            metrics["scene_success_score_per_task"][scene][task] = (metrics["scene_success_score_per_task"][scene][task] / scene_counts[scene]) * 100
    
    # Calculate average relaxations, refinements, and plan lengths
    for key in metrics["average_relaxations_per_task"]:
        metrics["average_relaxations_per_task"][key] /= task_counts[key]
    for key in metrics["average_relaxations_per_scene"]:
        metrics["average_relaxations_per_scene"][key] /= scene_counts[key]
    for key in metrics["average_relaxations_per_problem_id"]:
        metrics["average_relaxations_per_problem_id"][key] /= problem_counts[key]
    for key in metrics["average_refinements_per_task"]:
        metrics["average_refinements_per_task"][key] /= task_counts[key]
    for key in metrics["average_refinements_per_scene"]:
        metrics["average_refinements_per_scene"][key] /= scene_counts[key]
    for key in metrics["average_plan_length_per_task"]:
        metrics["average_plan_length_per_task"][key] /= task_counts[key]
    for key in metrics["average_plan_length_per_scene"]:
        metrics["average_plan_length_per_scene"][key] /= scene_counts[key]
    
    # Calculate weighted success scores as percentages
    for key in metrics["weighted_success_score_per_task"]:
        metrics["weighted_success_score_per_task"][key] = (metrics["weighted_success_score_per_task"][key] / task_counts[key]) * 100
    for key in metrics["weighted_success_score_per_scene"]:
        metrics["weighted_success_score_per_scene"][key] = (metrics["weighted_success_score_per_scene"][key] / scene_counts[key]) * 100
    for key in metrics["weighted_success_score_per_problem_id"]:
        metrics["weighted_success_score_per_problem_id"][key] = (metrics["weighted_success_score_per_problem_id"][key] / problem_counts[key]) * 100
    
    return metrics

def save_metrics(filepath, metrics):
    """
    Save the computed metrics to a JSON file.
    """
    with open(filepath, mode='w') as file:
        json.dump(metrics, file, indent=4)

def compute_metrics_for_splits(selected_dataset_splits, model, timestamp=None):
    """
    Compute metrics for selected dataset splits.
    """
    if timestamp is None:
        # Retrieve the latest timestamp in /DATA/context-matters/results/{model}
        timestamps = os.listdir(f'/DATA/context-matters/results/{model}')
        timestamp = sorted(timestamps)[-1]

    all_data = []
    for split in selected_dataset_splits:
        filepath = f'/DATA/context-matters/results/{model}/{timestamp}/{split}.csv'
        if os.path.exists(filepath):
            data = load_csv(filepath)
            all_data.extend(data)
        else:
            print(f"File not found: {filepath}")
    
    pprint(all_data)

    if all_data:
        return compute_metrics(all_data)
    else:
        raise Exception("No data found for the selected dataset splits")

if __name__ == "__main__":
    # Define the dataset splits
    DATASET_SPLITS = [
        "dining_setup",
        "house_cleaning",
        "laundry"
    ]
    
    # Compute metrics for the selected dataset splits
    metrics = compute_metrics_for_splits(DATASET_SPLITS, model="CM")
    
    # Save metrics to JSON
    save_metrics('/DATA/context-matters/frozen_results/metrics.json', metrics)
    
    # Define the path to save plots
    save_path = '/DATA/context-matters/frozen_results'
    
    # Plot metrics and save plots
    plot_metrics(metrics, save_path)
    
    # Plot weighted success scores and save plots
    plot_metrics(metrics, save_path, metric="weighted_success_score_per_task")
    plot_metrics(metrics, save_path, metric="weighted_success_score_per_scene")
    plot_metrics(metrics, save_path, metric="weighted_success_score_per_problem_id")
    plot_metrics(metrics, save_path, metric="global_weighted_success_score")
    
    # Plot other metrics
    plot_metrics(metrics, save_path, metric="success_score_per_scene")
    plot_metrics(metrics, save_path, metric="success_score_per_task")
    plot_metrics(metrics, save_path, metric="success_score_per_problem_id")
    plot_metrics(metrics, save_path, metric="global_success_score")
    plot_metrics(metrics, save_path, metric="average_relaxations_per_task")
    plot_metrics(metrics, save_path, metric="average_relaxations_per_scene")
    plot_metrics(metrics, save_path, metric="average_relaxations_per_problem_id")
    plot_metrics(metrics, save_path, metric="average_refinements_per_task")
    plot_metrics(metrics, save_path, metric="average_refinements_per_scene")
    plot_metrics(metrics, save_path, metric="average_plan_length_per_task")
    plot_metrics(metrics, save_path, metric="average_plan_length_per_scene")
    plot_metrics(metrics, save_path, metric="global_average_relaxations")
    plot_metrics(metrics, save_path, metric="global_average_refinements")
    plot_metrics(metrics, save_path, metric="planning_success_but_grounding_failure")
