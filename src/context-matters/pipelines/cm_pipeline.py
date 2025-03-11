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

class ContextMattersPipeline(BasePipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.determine_possibility: bool = kwargs["determine_possibility"]
        self.prevent_impossibility: bool = kwargs["prevent_impossibility"]
        self.workflow_iterations: int = kwargs["workflow_iterations"]
        self.pddl_gen_iterations: int = kwargs["pddl_generation_iterations"]
        self._name: str = "Context Matters"
        
        self.experiment_name: str = self._construct_experiment_name()
        self.current_phase: Optional[str] = None

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
                domain_file_path=domain_file_path,
                results_dir=results_problem_dir,
                domain_description=domain_description,
            )
            
            (final_problem_file_path, final_plan_file_path, final_goal,
                planning_successful, grounding_successful, task_possible,
                possibility_explanation, n_relaxations, refinements_per_iteration,
                goal_relaxations, failure_stage, failure_reason) = results
            
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
                phase=self.current_phase,
                exception=e
            )
            
        return
    
    def bidimensional_planning(self, **kwargs):
        '''
        Algorithm 1 of the paper
        '''
        goal_file_path = kwargs["goal_file_path"]
        initial_location_file_path = kwargs["initial_location_file_path"]
        scene_graph_file_path = kwargs["scene_graph_file_path"]
        domain_file_path = kwargs["domain_file_path"]
        results_dir = kwargs["results_dir"]
        domain_description = kwargs["domain_description"]
        
        # Setup
        scene_graph = read_graph_from_path(Path(scene_graph_file_path))
        extracted_sg = get_verbose_scene_graph(scene_graph, as_string=False)
        extracted_sg_str = get_verbose_scene_graph(scene_graph, as_string=True)
        current_goal = open(goal_file_path, "r").read()
        initial_robot_location = open(initial_location_file_path, "r").read()
        
        # Write the verbose scene graph to file
        with open(os.path.join(results_dir, "extracted_scene_graph.txt"), "w") as file:
            file.write(extracted_sg_str)

        # Initialize workflow variables
        planning_successful = False
        grounding_successful = False
        
        scene_graph_grounding_log = None
        pddlenv_error_log = None
        planner_error_log = None
        VAL_validation_log = None
        planner_statistics = None
        
        task_possible = None
        possibility_explanation = None
        
        iteration = 0  # Iteration counter of the outer loop
        refinements_per_iteration = []  # Output data: number of inner loop iterations per outer loop iteration
        goals = [current_goal]  # Output data: all relaxed goals
        
        # Outer loop
        while iteration < self.workflow_iterations:
            # Reset VAL-related flags for this outer iteration
            VAL_grounding_successful = False
            
            logging.info(f"------------ Workflow iteration {iteration+1}/{self.workflow_iterations} ------------")
            
            iteration_dir = os.path.join(results_dir, f"iteration_{iteration}")
            os.makedirs(iteration_dir, exist_ok=True)
            
            problem_dir = os.path.join(iteration_dir, "refinement_0")
            os.makedirs(problem_dir, exist_ok=True)

            problem_file_path = os.path.join(problem_dir, "problem.pddl")
            
            if domain_file_path is None:
                domain_file_path = os.path.join(iteration_dir, "generated_domain.pddl")

            logs_dir = os.path.join(iteration_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            if self.determine_possibility:
                logging.info("Preventing impossible problems")
                self.current_phase = "IMPOSSIBLE_PROBLEM_PREVENTION"
                task_possible, possibility_explanation = determine_problem_possibility(
                    domain_file_path=domain_file_path,
                    initial_robot_location=initial_robot_location,
                    task=current_goal,
                    environment=extracted_sg_str,
                    model=self.model,
                    logs_dir=logs_dir
                )

                with open(os.path.join(results_dir, "logs", "explanation.txt"), "w") as file:
                    file.write(possibility_explanation)
                        
                if task_possible.lower() == "false":
                    logging.info("Task is impossible, stopping current workflow")
                    logging.debug(possibility_explanation)
                    if self.prevent_impossibility:
                        return (problem_file_path, None, current_goal,
                                False, False, False, possibility_explanation,
                                iteration, refinements_per_iteration, goals,
                                self.current_phase, "")
                else:
                    logging.info("Task is possible, continuing workflow")
                    logging.debug(possibility_explanation)
                        
            if domain_description:
                logging.info("Generating domain")
                domain_pddl = generate_domain(  #TODO ASK EMA
                    domain_file_path=domain_file_path,
                    goal_file_path=goal_file_path,
                    domain_description=domain_description,
                    logs_dir=logs_dir,
                    model=self.model
                )
                
                # Check domain with VAL using helper method
                VAL_successful, VAL_validation_log = self._check_val(domain_file_path)

                self.current_phase = "DOMAIN_GENERATION"
                save_statistics(
                    dir=results_dir,
                    workflow_iteration=iteration,
                    phase=self.current_phase,
                    VAL_validation_log=VAL_validation_log
                )
                
            logging.info("Generating problem")
            problem = generate_problem(  #TODO ASK EMA
                domain_file_path=domain_file_path,
                initial_robot_location=initial_robot_location,
                task=current_goal,
                environment=extracted_sg_str,
                problem_file_path=problem_file_path,
                logs_dir=logs_dir,
                workflow_iteration=iteration,
                model=self.model,
                ADD_PREDICATE_UNDERSCORE_EXAMPLE=(domain_description is not None)
            )
            # The underscore example is only used when generating the domain from a description
            
            # Check problem and domain with VAL using helper method
            VAL_successful, VAL_validation_log = self._check_val(domain_file_path, problem_file_path)
            
            plan_file_path = os.path.join(problem_dir, "plan_0.out")
            planner_output_file_path = os.path.join(problem_dir, "logs", "planner_output_0.log")
            
            plan, pddlenv_error_log, planner_error_log, planner_statistics = plan_with_output(
                domain_file_path, problem_dir, plan_file_path, search_flag='--search "astar(ff())"'
            )      
            planning_successful = plan is not None
            
            # Check planning with VAL (outer loop version) using helper method.
            # Use the iteration number as a suffix for the translated plan file.
            VAL_successful, VAL_validation_log, VAL_ground_successful, VAL_grounding_log = \
                self._check_val_planning(domain_file_path, problem_file_path, plan_file_path, 
                                        translation_suffix=str(iteration), problem_dir=problem_dir)
            VAL_grounding_successful = VAL_successful and VAL_ground_successful
                
            self.current_phase = "INITIAL_PLANNING"
            save_statistics(
                dir=results_dir,
                workflow_iteration=iteration,
                plan_successful=(plan is not None),
                pddlenv_error_log=pddlenv_error_log,
                planner_error_log=planner_error_log,
                VAL_grounding_log=VAL_grounding_log,
                VAL_validation_log=VAL_validation_log,
                planner_statistics=planner_statistics,
                phase=self.current_phase
            )
                
            # Inner loop
            PDDL_loop_iteration = 0
            if not planning_successful or not VAL_grounding_successful:
                self.current_phase = "PDDL_REFINEMENT"
                    
                while PDDL_loop_iteration < self.pddl_gen_iterations and \
                    (not planning_successful or not VAL_grounding_successful):
                    logging.info(f"------------ Refinement iteration {PDDL_loop_iteration+1}/{self.pddl_gen_iterations} ------------")
                    logging.debug(f"Problem directory: {problem_dir}")
                        
                    logging.info("Refining problem")
                    new_problem = refine_problem(
                        planner_output_file_path=planner_output_file_path,
                        domain_file_path=domain_file_path,
                        problem_file_path=problem_file_path,
                        scene_graph=extracted_sg_str,
                        task=current_goal,
                        logs_dir=logs_dir,
                        workflow_iteration=iteration,
                        refinement_iteration=PDDL_loop_iteration,
                        pddlenv_error_log=pddlenv_error_log,
                        planner_error_log=planner_error_log,
                        VAL_validation_log=VAL_validation_log,
                        VAL_grounding_log=VAL_grounding_log,
                        scene_graph_grounding_log=scene_graph_grounding_log
                    )
                        
                    scene_graph_grounding_log = None
                        
                    new_problem_dir = problem_dir.split("/")[:-1] + [f"refinement_{PDDL_loop_iteration+1}"]
                    new_problem_dir = "/" + os.path.join(*new_problem_dir)
                    os.makedirs(new_problem_dir, exist_ok=True)
                        
                    new_problem_file_path = os.path.join(new_problem_dir, "problem.pddl")
                        
                    logging.debug(f"Saving new problem to {new_problem_file_path}")
                    with open(new_problem_file_path, "w") as file:
                        file.write(new_problem)
                            
                    problem_dir = new_problem_dir
                    problem_file_path = new_problem_file_path
                    PDDL_loop_iteration += 1
                        
                    # Check new planning phase
                    plan_file_path = os.path.join(problem_dir, f"plan_{PDDL_loop_iteration}.out")  # TODO adjust if needed
                    planner_output_file_path = os.path.join(problem_dir, "logs", f"planner_output_{PDDL_loop_iteration}.log")
                        
                    # Attempt planning with refined problem
                    plan, pddlenv_error_log, planner_error_log, planner_statistics = plan_with_output(domain_file_path, problem_dir, plan_file_path)    
                    planning_successful = plan is not None
                        
                    # Check planning with VAL (inner loop version) using helper method.
                    VAL_successful, VAL_validation_log, VAL_ground_successful, VAL_grounding_log = \
                        self._check_val_planning(domain_file_path, problem_file_path, plan_file_path, 
                                                translation_suffix=f"{PDDL_loop_iteration}", problem_dir=problem_dir)
                    VAL_grounding_successful = VAL_successful and VAL_ground_successful
                        
                    save_statistics(
                        dir=results_dir,
                        workflow_iteration=iteration,
                        plan_successful=(plan is not None),
                        pddlenv_error_log=pddlenv_error_log,
                        planner_error_log=planner_error_log,
                        planner_statistics=planner_statistics,
                        phase=self.current_phase,
                        pddl_refinement_iteration=PDDL_loop_iteration,
                        VAL_grounding_log=VAL_grounding_log,
                        VAL_validation_log=VAL_validation_log
                    )
            else:
                planning_successful = True
                pddlenv_error_log = None
                planner_error_log = None
                planner_statistics = None
                
            # Record the number of refinements for each iteration
            refinements_per_iteration.append(PDDL_loop_iteration)
                
            if planning_successful and VAL_grounding_successful:
                logging.info("Planning and grounding successful")
            else:
                logging.info("Out of PDDL refinements")
                    
            # Grounding in the 3D Scene Graph
            if planning_successful and plan and VAL_grounding_successful:
                if self.ground_in_sg:
                    logging.info("Grounding in the 3DSG")
                    grounding_success_percentage, scene_graph_grounding_log = verify_groundability_in_scene_graph(
                        plan=plan,
                        graph=extracted_sg,
                        domain_file_path=domain_file_path,
                        problem_dir=problem_dir,
                        move_action_str="move_to",
                        location_relation_str="at",
                        location_type_str="room",
                        initial_robot_location=initial_robot_location,
                    )
                        
                    logging.info(f"Grounding result: {grounding_success_percentage}")
                    logging.debuf(scene_graph_grounding_log)
                        
                    grounding_successful = True if grounding_success_percentage == 1 else False
                        
                    self.current_phase = "SCENE_GRAPH_GROUNDING"
                    save_statistics(
                        dir=results_dir,
                        workflow_iteration=iteration,
                        planner_statistics=planner_statistics,
                        phase=self.current_phase,
                        scene_graph_grounding_log=scene_graph_grounding_log,
                        grounding_success_percentage=grounding_success_percentage
                    )
                else:
                    logging.info("Skipping grounding in the 3DSG")
                    grounding_successful = True
                    scene_graph_grounding_log = None
                        
            # If we achieved 100% grounding success, we can break the loop as we correctly achieved the original goal        
            if grounding_successful:
                logging.info("Perfect grounding success, stopping workflow")
                scene_graph_grounding_log = None
                break
            else:
                # Goal relaxation
                logging.info("Grounding not successful. Performing goal relaxation")
                alternatives, current_goal = dict_replaceable_objects(extracted_sg_str, current_goal,
                                                                        iteration, logs_dir)  # TODO alternatives?
                # Generate new goal
                current_goal = relax_goal(extracted_sg_str, current_goal)
                # Record relaxed goal
                goals.append(current_goal)
                # Prepare next iteration
                iteration += 1
            
        # Return final problem and plan
        return (problem_file_path, plan_file_path, current_goal,
                planning_successful, grounding_successful, task_possible,
                possibility_explanation, iteration, refinements_per_iteration, goals,
                "", "")

        
    def _check_val(self, domain_file_path, problem_file_path=None):
        """
        Helper method to perform VAL validation.
        If problem_file_path is provided, validates both domain and problem.
        Otherwise, validates only the domain.
        """
        if problem_file_path:
            logging.info("Checking problem and domain with VAL")
            VAL_successful, VAL_validation_log = VAL_validate(domain_file_path, problem_file_path)
        else:
            logging.info("Checking domain with VAL")
            VAL_successful, VAL_validation_log = VAL_validate(domain_file_path)
        if not VAL_successful:
            logging.info("VAL validation check failed")
            logging.debug(VAL_validation_log)
        else:
            logging.info("VAL validation check passed")
        return VAL_successful, VAL_validation_log


    def _check_val_planning(self, domain_file_path, problem_file_path, plan_file_path, translation_suffix, problem_dir):
        """
        Helper method to perform VAL validation and grounding for planning.
        Translates the plan using the given translation_suffix and then checks validation and grounding.
        Returns a tuple of (VAL_successful, VAL_validation_log, VAL_ground_successful, VAL_grounding_log).
        """
        # Translate the plan into a format parsable by VAL
        translated_plan_path = os.path.join(problem_dir, f"translated_plan_{translation_suffix}.txt") # TODO DISCUSS THIS
        translate_plan(plan_file_path, translated_plan_path)
        logging.info("Checking planning with VAL")
        VAL_successful, VAL_validation_log = VAL_validate(domain_file_path, problem_file_path, translated_plan_path)
        if VAL_successful:
            logging.info("VAL validation check passed")
            logging.info("Attempting to ground the plan")
            VAL_ground_successful, VAL_grounding_log = VAL_ground(domain_file_path, problem_file_path)
            if VAL_ground_successful:
                logging.info("Plan is grounded")
            else:
                logging.info("Plan is not grounded")
                logging.debug(VAL_grounding_log)
        else:
            logging.info("VAL validation check failed")
            logging.debug(VAL_validation_log)
            VAL_ground_successful = False
            VAL_grounding_log = None
        return VAL_successful, VAL_validation_log, VAL_ground_successful, VAL_grounding_log