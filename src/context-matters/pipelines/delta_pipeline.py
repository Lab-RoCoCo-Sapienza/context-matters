import os
import json
import csv
import traceback
from pathlib import Path
from typing import Optional

from .base_pipeline import BasePipeline
from utils import (
    copy_file, save_file, save_statistics,
    read_graph_from_path,
    get_verbose_scene_graph,
)

from logger_cfg import logging
from planner import plan_with_output
from pddl_generation import generate_domain, generate_problem, refine_problem, determine_problem_possibility
from pddl_verification import translate_plan, VAL_validate, VAL_ground, verify_groundability_in_scene_graph
from goal_relaxation import relax_goal, dict_replaceable_objects

class DeltaPipeline(BasePipeline):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name: str = "DELTA"
        
        self.experiment_name: str = self._construct_experiment_name()
        self.current_phase: Optional[str] = None
        
    @property
    def name(self):
        return self.name
    
    def _construct_experiment_name(self):
        name = f"DELTA_{self.model}"
        if self.generate_domain:
            name += "_gendomain"
        if self.ground_in_sg:
            name += "_ground"
        return name

    def run(self):
        results_dir = os.path.join(self.results_dir, self.experiment_name)
        os.makedirs(results_dir, exist_ok=True)
        
        for task_name in self.splits:
        # Loop over all the tasks that we want to consider
            self._process_task(task_name, results_dir)
        
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

    def _initialize_csv(self, csv_filepath):
        # Initialize CSV with headers
        header = ["Task", "Scene", "Problem", "Planning Successful", "Grounding Successful", 
                "Plan Length", "Number of subgoals", "Failure stage", "Failure Reason"]
        
        with open(csv_filepath, mode="w", newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(header)
            
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
            
    def _run_and_log_pipeline(self, task_name, scene_name, problem_id, results_problem_dir,
                            domain_file_path, domain_description, scene_graph_file_path,
                            csv_filepath):
        try:
            results = self.delta_planning(
                goal_file_path=os.path.join(results_problem_dir, "task.txt"),
                initial_location_file_path=os.path.join(results_problem_dir, "init_loc.txt"),
                scene_graph_file_path=scene_graph_file_path,
                domain_file_path=domain_file_path,
                results_dir=results_problem_dir,
                domain_description=domain_description,
                )
            
            (final_domain_file_path, planning_successful, grounding_successful, 
                subplans, failure_stage, failure_reason) = results
            
            plan_length = 0
            if planning_successful and grounding_successful:
                final_plan_file_path = os.path.join(results_problem_dir, "plan_final.txt")
                
                # Concatenate all subplans and write the resulting plan
                with open(final_plan_file_path, "w") as f:
                    for subplan in subplans:
                        for action in subplan:
                            f.write(str(action) + ",\n")
                            plan_length += 1
                            
                # Save final generated domain
                with open(final_domain_file_path, "r") as f:
                    final_generated_domain = f.read()
                save_file(final_generated_domain, os.path.join(results_problem_dir, "domain_final.pddl"))
                
                # Save results to CSV
                with open(csv_filepath, mode="a", newline='') as f:
                    writer = csv.writer(f, delimiter='|')
                    writer.writerow([
                        task_name, scene_name, problem_id, planning_successful, grounding_successful,
                        plan_length, len(subplans), "", ""
                    ])
                
            else:
                with open(csv_filepath, mode="a", newline='') as f:
                    writer = csv.writer(f, delimiter='|')
                    writer.writerow([
                        task_name, scene_name, problem_id, planning_successful, grounding_successful,
                        "", "", str(failure_stage).replace('\n', ' ').replace('\r', ''),
                        str(failure_reason).replace('\n', ' ').replace('\r', '')
                    ])
                    
            if planning_successful and grounding_successful:
                logging.info("DELTA pipelines successful")
                logging.debug(planning_successful)
                logging.debug(grounding_successful)
            else:
                logging.info("DELTA pipelines completed with issues")
                logging.debug(planning_successful)
                logging.debug(grounding_successful)
                
        except Exception as e:
            traceback.print_exc()
            exception_str = str(e).strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
            with open(csv_filepath, mode="a", newline='') as f:
                writer = csv.writer(f, delimiter='|')
                writer.writerow([task_name, scene_name, problem_id, False, False, "", "", "Exception", exception_str])
            
            # Save the exception to statistics.json
            save_statistics(
                dir=results_problem_dir,
                workflow_iteration=0,
                phase=self.current_phase,
                exception=e
            )
            
        return
    
    
    def delta_planning(self, **kwargs):
        
        goal_file_path = kwargs["goal_file_path"]
        initial_location_file_path = kwargs["initial_location_file_path"]
        scene_graph_file_path = kwargs["scene_graph_file_path"]
        domain_file_path = kwargs["domain_file_path"]
        results_dir = kwargs["results_dir"]
        domain_description = kwargs["domain_description"]