from agent import *
from utils import *

from pddl_generation import _save_prompt_response

    
def get_graph_and_task(problem_dir):
    for file in os.listdir(problem_dir):
        if file.endswith(".npz"):
            graph =  read_graph_from_path(Path(os.path.join(problem_dir, file)))
    task = open(os.path.join(problem_dir, "task.txt"),"r").read()
    return graph, task


def dict_replaceable_objects(graph, task, workflow_iteration=None, logs_dir=None):

    system_prompt = open(os.path.join("prompt", "replace_objects.txt"),"r").read()
    user_prompt = "SCENE: \n" + graph + "\n\nTASK: \n" + task

    answer = llm_call(system_prompt, user_prompt, temperature=0.1, top_p=1)
    
    if logs_dir is not None:
        _save_prompt_response(
            prompt=f"System:\n"+system_prompt+"\n\nUser:\n"+user_prompt,
            response=answer,
            prefix="object_replacement",
            suffix=workflow_iteration,
            output_dir=logs_dir
        )
    
    pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.findall(pattern, answer)

    new_goal = answer.split("<NEW_GOAL>")[1]

    return match, new_goal

def relax_goal(objects, goal, workflow_iteration=None, logs_dir=None):
    system_prompt = open(os.path.join("prompt", "relax_goal.txt"),"r").read()
    user_prompt = "Objects:\n" + str(objects) + "\nGoal:" + goal
    answer = llm_call(system_prompt, user_prompt, temperature=0.1, top_p=1)

    if logs_dir is not None:
        _save_prompt_response(
            prompt=f"System:\n"+system_prompt+"\n\nUser:\n"+user_prompt,
            response=answer,
            prefix="goal_relaxation",
            suffix=workflow_iteration,
            output_dir=logs_dir
        )

    new_goal = answer.split("<NEW_GOAL>")[1]
    return new_goal


if __name__ == "__main__":
    goal_relaxation = GoalRelaxation()
    path_problem = "/DATA/context-matters/dataset/dining_setup/Allensville/problem_4"

    graph, task = get_graph_and_task(path_problem)
    alternatives, new_goal = dict_replaceable_objects(graph, task)
    print(alternatives)
    print(new_goal)

    print(relax_goal(get_verbose_scene_graph(graph), new_goal))