from agent import *
from utils import *

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
        alternatives = {}
        graph, task = self.get_graph_and_task(problem_dir)
        objects_in_graph = get_verbose_scene_graph(graph)
        print(objects_in_graph)

        print("\n\n______________________\n\n")
        print(task)

        return alternatives
    
    def relaxate_goal(self):
        goal = ""

        return goal

goal_relaxation = GoalRelaxation()
path_problem = "/home/michele/Desktop/IROS_2025/context-matters/dataset/Allensville/problem_1"
print(goal_relaxation.dict_replacable_objects(path_problem))
