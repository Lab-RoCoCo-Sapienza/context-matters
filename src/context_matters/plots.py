import matplotlib.pyplot as plt
import os

def plot_success_score_per_scene(metrics, save_path):
    """
    Plot the success score per scene as a bar graph and save it.
    """
    # Extract scenes and their success scores
    scenes = list(metrics["success_score_per_scene"].keys())
    success_scores_scene = list(metrics["success_score_per_scene"].values())
    
    # Create bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(scenes, success_scores_scene, color='blue')
    plt.xlabel('Scenes')
    plt.ylabel('Success Score (%)')
    plt.title('Success Score per Scene')
    plt.savefig(os.path.join(save_path, 'success_score_per_scene.png'))
    plt.close()

def plot_success_score_per_task(metrics, save_path):
    """
    Plot the success score per task as a bar graph and save it.
    """
    # Extract tasks and their success scores
    tasks = list(metrics["success_score_per_task"].keys())
    success_scores_task = list(metrics["success_score_per_task"].values())
    
    # Create bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(tasks, success_scores_task, color='green')
    plt.xlabel('Tasks')
    plt.ylabel('Success Score (%)')
    plt.title('Success Score per Task')
    plt.savefig(os.path.join(save_path, 'success_score_per_task.png'))
    plt.close()

def plot_task_success_per_scene(metrics, save_path):
    """
    Plot the success score per task for each scene as bar graphs and save them.
    """
    # Iterate over each scene and plot success scores for tasks
    for scene in metrics["scene_success_score_per_task"]:
        tasks = list(metrics["scene_success_score_per_task"][scene].keys())
        success_scores = list(metrics["scene_success_score_per_task"][scene].values())
        
        # Create bar plot
        plt.figure(figsize=(10, 5))
        plt.bar(tasks, success_scores, color='purple')
        plt.xlabel('Tasks')
        plt.ylabel('Success Score (%)')
        plt.title(f'Success Score per Task in Scene {scene}')
        plt.savefig(os.path.join(save_path, f'success_score_per_task_in_scene_{scene}.png'))
        plt.close()

def plot_relaxations_and_refinements_per_task(metrics, save_path):
    """
    Plot the average number of relaxations and refinements per task as a bar graph and save it.
    """
    # Extract tasks, relaxations, and refinements
    tasks = list(metrics["average_relaxations_per_task"].keys())
    relaxations = list(metrics["average_relaxations_per_task"].values())
    refinements = list(metrics["average_refinements_per_task"].values())
    
    x = range(len(tasks))
    
    # Create bar plot with side-by-side bars for relaxations and refinements
    plt.figure(figsize=(10, 5))
    plt.bar(x, relaxations, width=0.4, label='Relaxations', align='center', color='red')
    plt.bar(x, refinements, width=0.4, label='Refinements', align='edge', color='orange')
    plt.xlabel('Tasks')
    plt.ylabel('Average Count')
    plt.title('Average Relaxations and Refinements per Task')
    plt.xticks(x, tasks)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'relaxations_and_refinements_per_task.png'))
    plt.close()

def plot_average_plan_length_per_task(metrics, save_path):
    """
    Plot the average plan length per task as a bar graph and save it.
    """
    # Extract tasks and their average plan lengths
    tasks = list(metrics["average_plan_length_per_task"].keys())
    plan_lengths = list(metrics["average_plan_length_per_task"].values())
    
    # Create bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(tasks, plan_lengths, color='cyan')
    plt.xlabel('Tasks')
    plt.ylabel('Average Plan Length')
    plt.title('Average Plan Length per Task')
    plt.savefig(os.path.join(save_path, 'average_plan_length_per_task.png'))
    plt.close()

def plot_average_plan_length_per_scene(metrics, save_path):
    """
    Plot the average plan length per scene as a bar graph and save it.
    """
    # Extract scenes and their average plan lengths
    scenes = list(metrics["average_plan_length_per_scene"].keys())
    plan_lengths = list(metrics["average_plan_length_per_scene"].values())
    
    # Create bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(scenes, plan_lengths, color='magenta')
    plt.xlabel('Scenes')
    plt.ylabel('Average Plan Length')
    plt.title('Average Plan Length per Scene')
    plt.savefig(os.path.join(save_path, 'average_plan_length_per_scene.png'))
    plt.close()

def plot_metrics(metrics, save_path, metric=None):
    """
    Plot the computed metrics and save the plots.
    """
    import matplotlib.pyplot as plt
    
    if metric:
        data = metrics[metric]
        plt.figure()
        if isinstance(data, dict):
            plt.bar(data.keys(), data.values())
            plt.xlabel('Keys')
            plt.ylabel('Values')
        else:
            plt.bar([metric], [data])
            plt.xlabel('Metric')
            plt.ylabel('Value')
        plt.title(f'{metric} Plot')
        plt.savefig(os.path.join(save_path, f'{metric}.png'))
        plt.close()
    else:
        plot_success_score_per_scene(metrics, save_path)
        plot_success_score_per_task(metrics, save_path)
        plot_task_success_per_scene(metrics, save_path)
        plot_relaxations_and_refinements_per_task(metrics, save_path)
        plot_average_plan_length_per_task(metrics, save_path)
        plot_average_plan_length_per_scene(metrics, save_path)
