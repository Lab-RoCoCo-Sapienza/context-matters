import os
import pandas as pd
import json
import argparse
from datetime import datetime
from typing import List

def get_latest_timestamp_dir(model_dir: str) -> str:
    """Get the subdirectory with the latest timestamp."""
    subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    timestamp_dirs = [d for d in subdirs if len(d) == 15 and d[8] == '-']
    if not timestamp_dirs:
        return model_dir
    latest = max(timestamp_dirs, key=lambda x: datetime.strptime(x, '%Y%m%d-%H%M%S'))
    return os.path.join(model_dir, latest)

def process_directory_argument(directory: str) -> List[str]:
    """Process a directory containing model subdirectories."""
    model_dirs = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            model_dirs.append(get_latest_timestamp_dir(full_path))
    return model_dirs

def aggregate_experiment_csvs(experiment_path: str) -> pd.DataFrame:
    """Aggregate all CSV files in an experiment directory excluding possibility.csv."""
    all_data = []
    for root, _, files in os.walk(experiment_path):
        for file in files:
            if file.endswith(".csv") and "possibility.csv" != file:
                path = os.path.join(root, file)
                df = pd.read_csv(path, sep='|')
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def get_possible_tasks(filepath: str) -> tuple[dict, bool]:
    """Get possible tasks from CSV file and return dict and availability flag."""
    dict_possible_tasks = {}
    has_possibility_data = False
    try:
        possible_tasks = pd.read_csv(filepath, sep='|')
        print(f"Found possibility.csv at: {filepath}")
        
        # Define expected columns and their potential variations
        column_mappings = {
            'Task': ['Task', 'task', 'TASK'],
            'Scene': ['Scene', 'scene', 'SCENE'],
            'Problem': ['Problem', 'problem', 'PROBLEM'],
            'Task possible': ['Task possible', 'task_possible', 'possible', 'Possible']
        }
        
        # Standardize column names
        for standard_name, variations in column_mappings.items():
            for var in variations:
                if var in possible_tasks.columns:
                    possible_tasks = possible_tasks.rename(columns={var: standard_name})
                    break
        
        # Verify required columns exist
        required_columns = ['Task', 'Scene', 'Problem', 'Task possible']
        missing_columns = [col for col in required_columns if col not in possible_tasks.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns in possibility.csv: {missing_columns}")
            print("Available columns:", possible_tasks.columns.tolist())
            return dict_possible_tasks, False
            
        for index, row in possible_tasks.iterrows():
            if row["Task"] not in dict_possible_tasks:
                dict_possible_tasks[row["Task"]] = []
            dict_possible_tasks[row["Task"]].append({
                "scene": row["Scene"],
                "problem": row["Problem"],
                "possible": row["Task possible"]
            })
        has_possibility_data = True
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: No possibility.csv found at {filepath}")
        print("Possibility-dependent metrics will be ignored")
    except Exception as e:
        print(f"Error processing possibility.csv: {str(e)}")
        print("Possibility-dependent metrics will be ignored")
    return dict_possible_tasks, has_possibility_data

def find_key(d, key):
    found_values = []  # Lista per raccogliere tutti i valori trovati
    
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                found_values.append(v)  # Se troviamo la chiave, aggiungiamo il valore alla lista
            elif isinstance(v, (dict, list)):
                found_values.extend(find_key(v, key))  # Se è un dizionario o una lista, cerchiamo ricorsivamente

    elif isinstance(d, list):
        for item in d:
            found_values.extend(find_key(item, key))  # Iteriamo sugli elementi della lista
    
    return found_values

def create_results_directory(directory: str) -> str:
    """Create and return path to results directory."""
    directory_parent = os.path.dirname(directory)
    results_dir = os.path.join(directory_parent, "metrics")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def process_results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert results dictionary to DataFrame."""
    all_results = []
    for key, value in results.items():
        for item in value:
            item["Model"] = key
            all_results.append(item)
    
    df_all_results = pd.DataFrame(all_results)
    
    # Define base score columns
    base_score_columns = [
        "Planning score", 
        "Success Score", 
        "Grounding Score", 
        "Total Average Plan Length", 
        "Succesfull Average Plan Length", 
        "Mean Relaxation"
    ]
    
    # Get available columns
    available_columns = set()
    for result in all_results:
        available_columns.update(result.keys())
    
    # Create dynamic score columns list
    score_columns = base_score_columns + [
        col for col in ["Not feasible", "SR feasible", "Mean Num Nodes", "Mean Time"] 
        if col in available_columns
    ]
    
    # Calculate mean scores
    mean_scores = df_all_results.groupby("Model")[score_columns].mean().reset_index()
    
    # Reorder columns
    columns = ["Model"] + [col for col in df_all_results.columns if col != "Model"]
    df_all_results = df_all_results[columns]
    
    # Merge with mean scores
    df_all_results = df_all_results.merge(mean_scores, on="Model", suffixes=("", "_score_mean"))
    
    return df_all_results

def calculate_cm_averages(results: dict) -> dict:
    """Calculate averages for CM architectures."""
    cm_averages = {}
    metrics_to_average = [
        "Success Score", "Succesfull Average Plan Length", "Mean Time", 
        "Mean Num Nodes", "Mean Relaxation"
    ]

    total_cm_architectures = sum(1 for key in results.keys() if key.startswith("CM_"))

    for key, value in results.items():
        if key.startswith("CM_"):
            df = pd.DataFrame(value)
            df['Split'] = df['Task'].apply(lambda x: 
                ''.join(word[0].upper() for word in x.replace('.csv', '').split('_')) 
                if '_' in x else x.replace('.csv', '')[:2].upper()
            )
            
            split_means = df.groupby('Split')[metrics_to_average].mean().round(2).to_dict('index')
            for split, metrics in split_means.items():
                if split not in cm_averages:
                    cm_averages[split] = {}
                    for metric in metrics_to_average:
                        cm_averages[split][metric] = []
                for metric, value in metrics.items():
                    cm_averages[split][metric].append(value)

    for split in cm_averages:
        for metric in metrics_to_average:
            values = cm_averages[split][metric]
            cm_averages[split][metric] = round(sum(values) / total_cm_architectures, 2)

    return cm_averages

def save_results(results: dict, results_dir: str) -> None:
    """Save results to CSV and JSON files."""
    # Process results to DataFrame
    df_all_results = process_results_to_dataframe(results)
    
    # Save DataFrame to CSV
    df_all_results.to_csv(os.path.join(results_dir, "all_results.csv"), sep='|', index=False)
    
    # Prepare JSON results
    results_json = dict(results)
    
    # Calculate and add task averages
    task_averages = {}
    for key, value in results.items():
        task_df = pd.DataFrame(value)
        task_mean = task_df.mean(numeric_only=True).to_dict()
        task_averages[key] = task_mean
    results_json["task_averages"] = task_averages
    
    # Add CM architecture averages
    results_json["cm_split_averages"] = calculate_cm_averages(results)
    
    # Save to JSON
    with open(os.path.join(results_dir, "complete_results.json"), "w") as json_file:
        json.dump(results_json, json_file, indent=4)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process metrics from experiment results.')
    parser.add_argument('-d', '--directory', 
                        help='Directory containing model subdirectories', 
                        required=True)
    return parser.parse_args()

def main(args):
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory = os.path.join(BASE_DIR, args.directory)
    experiment_paths = process_directory_argument(directory)

    # Process possibility data
    global_possibility_path = os.path.join(directory, "possibility.csv")
    dict_possible_tasks, has_global_possibility = get_possible_tasks(global_possibility_path)

    # Process results
    results = {}
    for experiments_path in experiment_paths:
        # Only check for experiment-specific possibility.csv if no global data
        if not has_global_possibility:
            possibility_path = os.path.join(experiments_path, "possibility.csv")
            dict_possible_tasks, has_possibility_data = get_possible_tasks(possibility_path)
        else:
            has_possibility_data = has_global_possibility
        
        # Aggregate all CSVs in the experiment directory
        aggregated_df = aggregate_experiment_csvs(experiments_path)
        
        for root, dirs, files in os.walk(experiments_path):
            for file in files:
                if file.endswith(".csv") and "possibility.csv" != file:
                    # Determine if path contains timestamp (length 15 and contains hyphen at pos 8)
                    path_parts = root.split("/")
                    model_idx = -2 if (len(path_parts[-1]) == 15 and path_parts[-1][8] == '-') else -1
                    model = path_parts[model_idx]
                    if model not in results.keys():
                        results[model] = []
                    path = os.path.join(root, file)
                    df = pd.read_csv(path, sep='|')
                    
                    for index, row in df.iterrows():
                        df.at[index, "Planning Successful"] = str(row["Planning Successful"]).strip().lower() == "true"
                        df.at[index, "Grounding Successful"] = str(row["Grounding Successful"]).strip().lower() == "true"
                    
                    for index, row in df.iterrows():
                        task = row["Task"]
                        task_found = False
                        
                        # First try possibility.csv if available
                        if has_possibility_data and task in dict_possible_tasks:
                            for task_dict in dict_possible_tasks[task]:
                                if task_dict["scene"] == row["Scene"] and task_dict["problem"] == row["Problem"]:
                                    task_found = True
                                    df.loc[index, "Possible"] = task_dict["possible"]
                                    break
                        
                        # If not found in possibility.csv, check aggregated data
                        if not task_found:
                            task_match = aggregated_df[
                                (aggregated_df["Task"] == task) & 
                                (aggregated_df["Scene"] == row["Scene"]) & 
                                (aggregated_df["Problem"] == row["Problem"])
                            ]
                            if not task_match.empty:
                                task_found = True
                                df.loc[index, "Possible"] = True  # Default to possible
                        
                        if not task_found:
                            df.loc[index, "Possible"] = True  # Default to possible if task not found anywhere
                        
                        if task_found:
                            domain = file.replace(".csv", "")
                            path_to_json = os.path.join(experiments_path, task, row["Scene"], row["Problem"], "statistics.json")
                            if os.path.exists(path_to_json):
                                with open(path_to_json, "r") as json_file:
                                    data = json.load(json_file)
                                    num_node = find_key(data, "num_node_expansions")
                                    if num_node:
                                        df.loc[index, "numnodes"] = num_node[0]
                                    else:
                                        df.loc[index, "numnodes"] = 0
                                    total_time = find_key(data, "total_time")
                                    if total_time:
                                        df.loc[index, "total_time"] = total_time[0]
                                    else:
                                        df.loc[index, "total_time"] = 0

                    media_planning_succesfull = df[df["Planning Successful"] == True].shape[0] / df.shape[0] * 100
                    success_score = df[df["Grounding Successful"] == True].shape[0] / df.shape[0] * 100
                    if "delta" not in model:
                        mean_relaxation = df["Relaxations"].sum() / df.shape[0]
                    else:
                        mean_relaxation = 0
                    df_with_plan_true = df[df["Planning Successful"] == True]
                    succesfull_average_plan_length = df_with_plan_true["Plan Length"].sum() / df_with_plan_true.shape[0]
                    average_plan_length = df["Plan Length"].sum() / df["Plan Length"].shape[0]
                    if df_with_plan_true.shape[0] != 0:
                        grounding_score = df_with_plan_true[df_with_plan_true["Grounding Successful"] == True].shape[0] / df_with_plan_true.shape[0] * 100
                    else:
                        grounding_score = 0
                    
                    # Initialize possibility-dependent fields with defaults
                    df["Possible"] = True  # Default all tasks to possible
                    
                    if has_possibility_data:
                        for index, row in df.iterrows():
                            task = row["Task"]
                            if task in dict_possible_tasks:
                                for task_dict in dict_possible_tasks[task]:
                                    if task_dict["scene"] == row["Scene"] and task_dict["problem"] == row["Problem"]:
                                        df.loc[index, "Possible"] = task_dict["possible"]

                    # Calculate base metrics first
                    model_results = {
                        "Task": file.replace(".csv", ""),
                        "Planning score": round(media_planning_succesfull, 2),
                        "Success Score": round(success_score, 2),
                        "Grounding Score": round(grounding_score, 2),
                        "Total Average Plan Length": round(average_plan_length, 2),
                        "Succesfull Average Plan Length": round(succesfull_average_plan_length, 2),
                        "Mean Relaxation": round(mean_relaxation, 2)
                    }

                    # Add node and time metrics only if they are available in the dataframe
                    if "numnodes" in df.columns and df["numnodes"].notna().any():
                        mean_num_nodes = df["numnodes"].sum() / df.shape[0]
                        model_results["Mean Num Nodes"] = round(mean_num_nodes, 2)
                    
                    if "total_time" in df.columns and df["total_time"].notna().any():
                        mean_time = df["total_time"].sum() / df.shape[0]
                        model_results["Mean Time"] = mean_time

                    # Only add possibility-dependent metrics if data is available
                    if has_possibility_data:
                        not_feasible = df_with_plan_true[df_with_plan_true["Possible"] == True].shape[0] / df_with_plan_true.shape[0] * 100 if df_with_plan_true.shape[0] != 0 else 0
                        model_results.update({
                            "Not feasible": round(not_feasible, 2),
                            "SR feasible": round(media_planning_succesfull, 2) * (round(not_feasible, 2) / 100)
                        })

                    results[model].append(model_results)

    # Create results directory and save results
    results_dir = create_results_directory(directory)
    save_results(results, results_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)


