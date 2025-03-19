import os
import json
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from context_matters.utils import copy_file

class BasePipeline(ABC):
    def __init__(self, **kwargs):
        self.base_dir: str  = kwargs["base_dir"]
        self.data_dir: str = kwargs["data_dir"]
        self.results_dir: str = kwargs["results_dir"]
        self.splits: List[str]  = kwargs["splits"]
        self.generate_domain: bool = kwargs["generate_domain"]
        self.ground_in_sg: bool = kwargs["ground_in_sg"]
        self.model: str = kwargs["model"]
        
        self._name: str = "BasePipeline"
        
        self.api_key: str = os.getenv("API_KEY")
        
        self.scenes_per_task: Dict = {
            "dining_setup": ["Allensville", "Parole", "Shelbiana"],
            "house_cleaning": ["Allensville", "Parole", "Shelbiana"],
            "laundry": ["Kemblesville"],
            "office_setup": ["Allensville", "Parole", "Shelbiana"],
            "other_1": [
                "Beechwood",
                "Benevolence",
                "Coffeen",
                "Collierville",
                "Corozal",
                "Cosmos"
            ],
            "other_2": [
                "McDade",
                "Merom",
                "Mifflinburg", 
                "Muleshoe",
                "Newfields",
                "Noxapater"
            ],
            "pc_assembly": ["Allensville", "Parole", "Shelbiana"],
        }
        self.problems_per_task: Dict = {
            "dining_setup": 6,
            "house_cleaning": 6,
            "laundry": 6,
            "office_setup": 8,
            "other_1": 10,
            "other_2": 9,
            "pc_assembly": 3,
        }
        
        self.experiment_name: str = self._construct_experiment_name()
        self.current_phase: Optional[str] = None
    
    @property
    def name(self):
        return self._name
    
    def _construct_experiment_name(self):
        name = f"{self._name}_{self.model}"
        if self.generate_domain:
            name += "_gendomain"
        if self.ground_in_sg:
            name += "_ground"
        return name
    
    
    def _process_task(self, task_name, results_dir):
        task_dir = os.path.join(self.data_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        csv_filepath = os.path.join(results_dir, f"{task_name}.csv")
        self._initialize_csv(csv_filepath)

        task_file_path = os.path.join(self.data_dir, f"{task_name}.json")
        with open(task_file_path) as f:
            task_description = json.load(f)

        domain_file_path, domain_description = self._setup_domain(task_description, task_name, task_dir, results_dir)

        for scene_name in self.scenes_per_task[task_name]:
            # Loop over the Gibson scenes of the task
            self._process_scene(
                task_name, scene_name, task_dir, results_dir,
                domain_file_path, domain_description, csv_filepath
            )

    def _setup_domain(self, task_description, task_name, task_dir, results_dir):
        # Set up domain files based on the pipeline configuration
        if self.generate_domain:
            return None, task_description["domain"]

        domain_name = task_name.replace("_", "-") + ".pddl"
        domain_file_path = os.path.join(task_dir, domain_name)
        copy_file(domain_file_path, os.path.join(results_dir, task_name, "domain.pddl"))

        return domain_file_path, None

    def _process_scene(self, task_name, scene_name, task_dir, results_dir,
                domain_file_path, domain_description, csv_filepath):

        scene_dir = os.path.join(task_dir, scene_name)

        for problem_id in range(1, self.problems_per_task[task_name] + 1):
            # Loop over all the problems of the scene for that task
            problem_dir = os.path.join(scene_dir, f"problem_{problem_id}")
            results_problem_dir = os.path.join(results_dir, task_name, scene_name, f"problem_{problem_id}")
            os.makedirs(results_problem_dir, exist_ok=True)
            os.makedirs(os.path.join(results_problem_dir, "logs"), exist_ok=True)

            scene_graph_file_path = os.path.join(problem_dir, f"{scene_name}.npz")

            # Copy required files
            for file_name in ["task.txt", "init_loc.txt", "description.txt"]:
                copy_file(os.path.join(problem_dir, file_name), os.path.join(results_problem_dir, file_name))

            self._run_and_log_pipeline(
                task_name, scene_name, f"problem_{problem_id}", results_problem_dir,
                domain_file_path, domain_description, scene_graph_file_path,
                csv_filepath
            )

    def run(self):
        results_dir = os.path.join(self.results_dir, self.experiment_name)
        os.makedirs(results_dir, exist_ok=True)
        
        for task_name in self.splits:
        # Loop over all the tasks that we want to consider
            self._process_task(task_name, results_dir)

    @abstractmethod
    def _run_and_log_pipeline(self, task_name, scene_name, problem_id, results_problem_dir,
                            domain_file_path, domain_description, scene_graph_file_path,
                            csv_filepath):
        pass
    
    @abstractmethod
    def _initialize_csv(self, csv_filepath):
        pass