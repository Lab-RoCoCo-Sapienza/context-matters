import os
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
experiments_path = os.path.join(BASE_DIR, "experiments")

possible_tasks_path = os.path.join(experiments_path, "possibility.csv")
possible_tasks = pd.read_csv(possible_tasks_path, sep='|')
print(possible_tasks)
dict_possible_tasks = {}
for index, row in possible_tasks.iterrows():
    if row["Task"] not in dict_possible_tasks:
        dict_possible_tasks[row["Task"]] = []
    dict_possible_tasks[row["Task"]] .append({
        "scene": row["Scene"],
        "problem": row["Problem"],
        "possible": row["Task possible"]
    })

results = {}

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


for root, dirs, files in os.walk(experiments_path):
    for file in files:
        if file.endswith(".csv") and "possibility.csv" !=file:
            print(file)
            print(root.split("/")[-1])
            model = root.split("/")[-1]
            if model not in results.keys():
                results[model] = []
            path = os.path.join(root, file)
            df = pd.read_csv(path, sep='|')
            
            for index, row in df.iterrows():
                df.at[index, "Planning Successful"] = str(row["Planning Successful"]).strip().lower() == "true"
                df.at[index, "Grounding Successful"] = str(row["Grounding Successful"]).strip().lower() == "true"
            for index, row in df.iterrows():
                task = row["Task"]
                for task_dict in dict_possible_tasks[task]:
                    if task_dict["scene"] == row["Scene"] and task_dict["problem"] == row["Problem"]:
                        print(model, task_dict["scene"],task, task_dict["problem"])
                        domain = file.replace(".csv", "")
                        #construct path
                        path_to_json = os.path.join(experiments_path,model,task,task_dict['scene'],task_dict['problem'],"statistics.json")
                        if os.path.exists(path_to_json):
                            with open(path_to_json, "r") as json_file:
                                data = json.load(json_file)
                                num_node = find_key(data, "num_node_expansions")
                                print(num_node)
                                if num_node:
                                    df.loc[index, "numnodes"] = num_node[0]
                                else:
                                    df.loc[index, "numnodes"] = 0
                                total_time = find_key(data, "total_time")
                                if total_time:
                                    df.loc[index, "total_time"] = total_time[0]
                                else:
                                    df.loc[index, "total_time"] = 0
                        print(path_to_json)
                        df.loc[index, "Possible"] = task_dict["possible"]

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
            #Devo contare i true di planning succesfull dove però il problema non era possibile
            mean_num_nodes = df["numnodes"].sum() / df.shape[0]
            mean_time = df["total_time"].sum() / df.shape[0]
            """
            print("I piani totali sono ", df.shape[0])
            print("I piani fattibili ereano ", df[df["Possible"] == True].shape[0])
            print("I piani generati con successo erano ", df_with_plan_true.shape[0])
            print("I piani generati con successo e fattibili erano ", df_with_plan_true[df_with_plan_true["Possible"] == True].shape[0])
            """
            if df_with_plan_true.shape[0] != 0:
                not_faisable = df_with_plan_true[df_with_plan_true["Possible"] == True].shape[0] / df_with_plan_true.shape[0] * 100
            else:
                not_faisable = 0

            results[model].append({
                "Task": file.replace(".csv", ""),
                "Planning score": round(media_planning_succesfull, 2),
                "Success Score": round(success_score, 2),
                "Grounding Score": round(grounding_score, 2),
                "Total Average Plan Length": round(average_plan_length, 2),
                "Succesfull Average Plan Length": round(succesfull_average_plan_length, 2),
                "Not Faisable": round(not_faisable, 2),
                "SR faisable": round(media_planning_succesfull, 2) * (round(not_faisable, 2) / 100),
                "Mean Num Nodes": round(mean_num_nodes, 2),
                "Mean Time": mean_time,
                "Mean Relaxation": round(mean_relaxation, 2)
            })
print(results)

#TEST 
'''
path_test = "/home/michele/Downloads/Brave/frozen_results-20250228T144616Z-001/frozen_results/experiments/CM_gpt-4o_NOgendomain_groundsg/dining_setup.csv"
df_test = pd.read_csv(path_test, sep='|')
for index, row in df_test.iterrows():
    df_test.at[index, "Planning Succesful"] = str(row["Planning Succesful"]).strip().lower() == "true"
    df_test.at[index, "Grounding Successful"] = str(row["Grounding Successful"]).strip().lower() == "true"
for index, row in df_test.iterrows():
    task = row["Task"]
    for task_dict in dict_possible_tasks[task]:
        if task_dict["scene"] == row["Scene"] and task_dict["problem"] == row["Problem"]:
            df_test.loc[index, "Possible"] = task_dict["possible"]

media_planning_succesfull = df_test[df_test["Planning Succesful"] == True].shape[0] / df_test.shape[0] * 100
success_score = df_test[df_test["Grounding Successful"] == True].shape[0] / df_test.shape[0] * 100
if "delta" not in path_test:
    mean_relaxation = df_test["Relaxations"].sum() / df_test.shape[0]
else:
    mean_relaxation = 0
df_with_plan_true = df_test[df_test["Planning Succesful"] == True]
succesfull_average_plan_length = df_with_plan_true["Plan Length"].sum() / df_with_plan_true.shape[0]
average_plan_length = df_test["Plan Length"].sum() / df_test["Plan Length"].shape[0]
if df_with_plan_true.shape[0] != 0:
    grounding_score = df_with_plan_true[df_with_plan_true["Grounding Successful"] == True].shape[0] / df_with_plan_true.shape[0] * 100
else:
    grounding_score = 0
if df_with_plan_true.shape[0] != 0:
    not_faisable = df_with_plan_true[df_with_plan_true["Possible"] == True].shape[0] / df_with_plan_true.shape[0] * 100
else:
    not_faisable = 0

expected_results = {
    "Task": path_test.split("/")[-1].replace(".csv", ""),
    "Planning score": 88.89,
    "Success Score": 88.89,
    "Grounding Score": 100,
    "Total Average Plan Length": 21.83,
    "Succesfull Average Plan Length": 23.56,
    "Not Faisable": 93.75, # 18 piani , 17 sono fattibili , 16 quelli correttamente generati di cui uno però non era possibile quindi 16/17
    "SR faisable": 88.89 * (93.75 / 100), # 88.89 / 93.75 * 100
    "Mean Relaxation": 0.67
}

test_results = {
    "Task": path_test.split("/")[-1].replace(".csv", ""),
    "Planning score": round(media_planning_succesfull, 2),
    "Success Score": round(success_score, 2),
    "Grounding Score": round(grounding_score, 2),
    "Total Average Plan Length": round(average_plan_length, 2),
    "Succesfull Average Plan Length": round(succesfull_average_plan_length, 2),
    "Not Faisable": round(not_faisable, 2),
    "SR faisable": round(media_planning_succesfull, 2) * (round(not_faisable, 2) / 100),
    "Mean Relaxation": round(mean_relaxation, 2)
}

print("Expected results: ", expected_results)
print("Test results: ", test_results)
print("Test passed: ", expected_results == test_results)
'''


os.makedirs("results", exist_ok=True)
#results to csv
for key in results.keys():
    df = pd.DataFrame(results[key])
    df.to_csv(f"results/{key}.csv", sep='|', index=False)
    # Create a unique CSV where the key is the element on the row
    all_results = []
for key, value in results.items():
    for item in value:
        item["Model"] = key
        all_results.append(item)

df_all_results = pd.DataFrame(all_results)
numeric_columns = df_all_results.select_dtypes(include='number').columns
mean_results = df_all_results.groupby("Model")[numeric_columns].mean().reset_index()

columns = ["Model"] + [col for col in df_all_results.columns if col != "Model"]
df_all_results = df_all_results[columns]

score_columns = ["Planning score", "Success Score", "Grounding Score", "Total Average Plan Length", "Succesfull Average Plan Length", "Not Faisable", "Mean Relaxation"]
mean_scores = df_all_results.groupby("Model")[score_columns].mean().reset_index()

df_all_results = df_all_results.merge(mean_scores, on="Model", suffixes=("", "_score_mean"))

df_all_results.to_csv("results/all_results.csv", sep='|', index=False)
# Save results to JSON
results_json = {}
for key in results.keys():
    results_json[key] = results[key]

# Calculate and add averages per task
task_averages = {}
for key, value in results.items():
    task_df = pd.DataFrame(value)
    task_mean = task_df.mean(numeric_only=True).to_dict()
    task_averages[key] = task_mean

results_json["task_averages"] = task_averages
# Save the entire results dictionary to a single JSON file
with open("results/complete_results.json", "w") as json_file:
    json.dump(results_json, json_file, indent=4)


