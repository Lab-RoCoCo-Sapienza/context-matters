from agent import *
from utils import *
from pprint import pprint

class GoalRelaxation:

    def __init__(self, left_relaxations = 3, right_relaxations=3):
        self.left_relaxations = left_relaxations
        self.right_relaxations = right_relaxations
    
    def get_graph_and_task(self, problem_dir):
        for file in os.listdir(problem_dir):
            if file.endswith(".npz"):
                graph =  read_graph_from_path(Path(os.path.join(problem_dir, file)))
        task = open(os.path.join(problem_dir, "task.txt"),"r").read()
        return graph, task

    def dict_replacable_objects(self, problem_dir):
        graph, task = self.get_graph_and_task(problem_dir)
        objects_in_graph = get_verbose_scene_graph(graph)
        print(objects_in_graph)

        print("\n\n______________________\n\n")
        print(task)

        answer = llm_call(open(os.path.join("prompt", "replace_objects.txt"),"r").read(), "SCENE: \n" + objects_in_graph + "\n\nTASK: \n" + task)
        print(answer)
        pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.findall(pattern, answer)

        new_goal = answer.split("<NEW_GOAL>")[1]

        return match, new_goal
    
    def relaxate_goal(self,objects,goal):
        answer = llm_call(open(os.path.join("prompt", "relaxate_goal.txt"),"r").read(),"Objects:\n" + str(objects) + "\nGoal:" + goal)
        return answer

goal_relaxation = GoalRelaxation()
path_problem = "/home/michele/Desktop/IROS_2025/context-matters/dataset/Allensville/problem_2"

graph, task = goal_relaxation.get_graph_and_task(path_problem)
alternativies, new_goal = goal_relaxation.dict_replacable_objects(path_problem)
print(alternativies)
print(new_goal)

print(goal_relaxation.relaxate_goal(get_verbose_scene_graph(graph),new_goal))