import os
import json
import csv
import traceback
from pathlib import Path

from .base_pipeline import BasePipeline
from utils import (
    copy_file, save_file, save_statistics,
    read_graph_from_path,
    get_verbose_scene_graph,
    print_red, print_green, print_blue,
    print_yellow, print_magenta, print_cyan,
)

from planner import plan_with_output
from pddl_generation import generate_domain, generate_problem, refine_problem, determine_problem_possibility
from pddl_verification import translate_plan, VAL_validate, VAL_ground, verify_groundability_in_scene_graph
from goal_relaxation import relax_goal, dict_replaceable_objects


class ContextMattersPipeline(BasePipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.determine_possibility: bool = kwargs["determine_possibility"]
        self.prevent_impossibility: bool = kwargs["prevent_impossibility"]
        self.workflow_iterations: int = kwargs["workflow_iterations"]
        self.pddl_gen_iterations: int = kwargs["pddl_generation_iterations"]
        self._name: str = "Context Matters"
        
        self.experiment_name: str = self._construct_experiment_name()

    @property
    def name(self):
        return self._name

    def _construct_experiment_name(self):
        name = f"CM_{self.model}"
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
                "Plan Length", "Relaxations", "Refinements per iteration", "Goal relaxations",
                "Failure Stage", "Failure Reason"]
        
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
            # Run the pipeline
            results = self.bidimensional_planning(
                goal_file_path=os.path.join(results_problem_dir, "task.txt"),
                initial_location_file_path=os.path.join(results_problem_dir, "init_loc.txt"),
                scene_graph_file_path=scene_graph_file_path,
                description_file_path=os.path.join(results_problem_dir, "description.txt"),
                domain_file_path=domain_file_path,
                scene_name=scene_name,
                problem_id=problem_id,
                results_dir=results_problem_dir,
                domain_description=domain_description,
            )
            
            (final_problem_file_path, final_plan_file_path, final_goal,
                planning_successful, grounding_successful, task_possible,
                possibility_explanation, n_relaxations, refinements_per_iteration,
                goal_relaxations, failure_stage, failure_reason, current_phase
            ) = results

            # Save the final problem and plan if successful
            plan_length = 0
            if planning_successful and grounding_successful:
                with open(final_problem_file_path, "r") as f:
                    save_file(f.read(), os.path.join(results_problem_dir, "problem_final.pddl"))

                with open(final_plan_file_path, "r") as f:
                    final_generated_plan = f.read()
                    save_file(final_generated_plan, os.path.join(results_problem_dir, "plan_final.txt"))
                    plan_length = len(final_generated_plan.split(", "))

            # Save the relaxed goal if any relaxations occurred
            if n_relaxations > 0:
                save_file(final_goal, os.path.join(results_problem_dir, "task_final.txt"))

            # Format refinements and goal relaxations
            refinements_per_iteration_str = ";".join(map(str, refinements_per_iteration))
            goal_relaxations_str = "; ".join(
                f"'{relax.strip().replace('\n', ' ').replace('\r', '')}'"
                for relax in goal_relaxations
            )

            # Save results to the main CSV file
            with open(csv_filepath, mode="a", newline='') as f:
                writer = csv.writer(f, delimiter='|')
                writer.writerow([
                    task_name, scene_name, problem_id, planning_successful, grounding_successful,
                    plan_length, n_relaxations, refinements_per_iteration_str,
                    goal_relaxations_str, failure_stage, failure_reason
                ])

            # Save the possibility result if enabled
            if self.determine_possibility:
                with open(os.path.join(self.results_dir, "possibility.csv"), mode="a", newline='') as f:
                    writer = csv.writer(f, delimiter='|')
                    writer.writerow([
                        task_name, scene_name, problem_id, task_possible,
                        possibility_explanation.strip().replace('\n', ' ').replace('\r', '')
                    ])

        except Exception as e:
            exception_str = str(e).strip().replace('\n', ' ').replace('\r', '')

            # Log the exception in the main CSV file
            with open(csv_filepath, mode="a", newline='') as f:
                writer = csv.writer(f, delimiter='|')
                writer.writerow([
                    task_name, scene_name, problem_id, f"Exception: {exception_str}",
                    "", "", "", "", "", "", ""
                ])

            # Log the possibility failure if enabled
            if self.determine_possibility:
                with open(os.path.join(self.results_dir, "possibility.csv"), mode="a", newline='') as f:
                    writer = csv.writer(f, delimiter='|')
                    writer.writerow([task_name, scene_name, problem_id, False, f"Exception: {exception_str}"])

            # Write the exception traceback to error.txt
            error_log_path = os.path.join(results_problem_dir, "logs", "error.txt")
            with open(error_log_path, "w") as error_log_file:
                traceback.print_exc(file=error_log_file)

            # Save exception details to statistics.json
            save_statistics(
                dir=results_problem_dir,
                workflow_iteration=0,
                phase=current_phase,
                exception=e
            )
            
        return
    
    def bidimensional_planning(**kwargs):
        
        goal_file_path = kwargs["goal_file_path"]
        initial_location_file_path = kwargs["initial_location_file_path"]
        scene_graph_file_path = kwargs["scene_graph_file_path"]
        description_file_path = kwargs["description_file_path"]
        domain_file_path = kwargs["domain_file_path"]
        scene_name = kwargs["scene_name"]
        problem_id = kwargs["problem_id"]
        results_dir = kwargs["results_dir"]
        domain_description = kwargs["domain_description"]
        
        
        