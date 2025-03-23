import os
import csv
import traceback
import json

from .base_pipeline import BasePipeline

from src.context_matters.utils import (
    save_file, save_statistics,
    save_graph
)
from src.context_matters.logger_cfg import logger
from src.context_matters.planner import plan_with_output
from src.context_matters.pddl_verification import verify_groundability_in_scene_graph, convert_JSON_to_locations_dictionary, VAL_validate, VAL_parse, VAL_ground, translate_plan
from src.context_matters.delta_prompts import generate_pddl_domain, prune_scene_graph, generate_pddl_problem, decompose_pddl_goal

from pddlgym.core import PDDLEnv

class DeltaPipeline(BasePipeline):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name: str = "DELTA"
        self.experiment_name = super()._construct_experiment_name()

    def _initialize_csv(self, csv_filepath):
        # Initialize CSV with headers
        header = ["Task", "Scene", "Problem", "Planning Successful", "Grounding Successful", 
                "Plan Length", "Number of subgoals", "Failure stage", "Failure Reason"]
        
        with open(csv_filepath, mode="w", newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(header)
            
    def _run_and_log_pipeline(self, task_name, scene_name, problem_id, results_problem_dir,
                            domain_file_path, domain_description, scene_graph_file_path,
                            csv_filepath):
        try:
            results = self._delta_planning(
                goal_file_path=os.path.join(results_problem_dir, "task.txt"),
                initial_location_file_path=os.path.join(results_problem_dir, "init_loc.txt"),
                scene_graph_file_path=scene_graph_file_path,
                domain_file_path=domain_file_path,
                results_dir=results_problem_dir,
                domain_description=domain_description,
                )
            (final_domain_file_path, final_pruned_scene_graph, final_problem_file_path, 
            final_subgoals_file_paths, planning_successful, grounding_successful, 
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
                logger.info("DELTA pipelines successful")
                logger.debug(planning_successful)
                logger.debug(grounding_successful)
            else:
                logger.info("DELTA pipelines completed with issues")
                logger.debug(planning_successful)
                logger.debug(grounding_successful)
                
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
    
    
    def _delta_planning(self, **kwargs):
        
        goal_file_path = kwargs["goal_file_path"]
        initial_location_file_path = kwargs["initial_location_file_path"]
        scene_graph_file_path = kwargs["scene_graph_file_path"]
        domain_file_path = kwargs["domain_file_path"]
        results_dir = kwargs["results_dir"]
        domain_description = kwargs["domain_description"]
        
        logs_dir = os.path.join(results_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        initial_robot_location = open(initial_location_file_path, "r").read()
        
        if domain_file_path is None:
            self.current_phase = "DOMAIN GENERATION"
            assert (
                domain_description is not None,
                "Provide the domain or the domain description"
            )
            
            logger.info("Generating PDDL domain")
            domain_pddl = generate_pddl_domain(goal_file_path, domain_description,
                                                logs_dir=logs_dir, model=self.model)
            domain_file_path = os.path.join(results_dir, "domain", "domain.pddl")
            
            # Save generated PDDL domain
            os.makedirs(os.path.dirname(domain_file_path), exist_ok=True)
            with open(domain_file_path, "w") as f:
                f.write(domain_pddl)
                
        self.current_phase = "DOMAIN VALIDATION"
        logger.info("Validating the domain")
        val_parse_success, val_parse_log = VAL_validate(domain_file_path)
        save_statistics(
            dir=results_dir,
            workflow_iteration=0,
            phase=self.current_phase,
            VAL_validation_log=val_parse_log
        )
        
        if not val_parse_success:
            return domain_file_path, None, None, None, False, False, None, "DOMAIN_GENERATION", val_parse_log
        logger.info("Domain validation successful")
        
        logger.info("Pruning scene graph")
        self.current_phase = "PRUNING_SCENE_GRAPH"
        pruned_sg = prune_scene_graph(scene_graph_file_path, goal_file_path, initial_robot_location, 
                                        logs_dir=logs_dir, model=self.model)
        
        # Save pruned scene graph
        pruned_sg_path_readable = os.path.join(results_dir, "pruned_sg.txt")
        with open(pruned_sg_path_readable, "w") as f:
            f.write(json.dumps(pruned_sg, indent=4))

        pruned_sg_path_npz = os.path.join(results_dir, "pruned_sg.npz")
        save_graph(pruned_sg, pruned_sg_path_npz)
        
        logger.info("Generating PDDL problem")
        self.current_phase = "PROBLEM GENERATION"
        problem_pddl = generate_pddl_problem(pruned_sg_path_npz, goal_file_path, domain_file_path,
                                            initial_robot_location, logs_dir=logs_dir, model=self.model)
        
        # Save the generated PDDL problem
        problem_pddl_path = os.path.join(results_dir, "problem.pddl")
        with open(problem_pddl_path, "w") as f:
            f.write(problem_pddl)
            
        logger.info("Validating the problem")
        val_parse_success, val_parse_log = VAL_validate(domain_file_path, problem_pddl_path)
        
        save_statistics(
            dir=results_dir,
            workflow_iteration=0,
            phase=self.current_phase,
            VAL_validation_log=val_parse_log
        )
        
        if not val_parse_success:
            return domain_file_path, pruned_sg, problem_pddl_path, None, False, False, None, "PROBLEM_GENERATION", val_parse_log
        logger.info("Problem validation successful")
        
        logger.info("Generating PDDL sub-goals")
        self.current_phase = "SUBGOAL_GENERATION"
        sub_goals_pddl = decompose_pddl_goal(problem_pddl_path, domain_file_path, initial_robot_location, 
                                            logs_dir=logs_dir, model=self.model)
        
        logger.info("Autoregressive planning and grounding")
        self.current_phase="PLANNING"
        
        old_pddl_env = None
        plans = []
        sub_goals_file_paths = []
        
        for i, sub_goal in enumerate(sub_goals_pddl):
            
            sub_goal_dir = os.path.join(results_dir, f"sub_goal_{i+1}")
            os.makedirs(sub_goal_dir, exist_ok=True)
            sub_goal_file = os.path.join(sub_goal_dir, f"sub_goal_{i+1}.pddl")
            sub_goals_file_paths.append(sub_goal_file)
            with open(sub_goal_file, "w") as f:
                f.write(sub_goal)
                
        # Evaluate the sequence of sub-goals
        # 1) Replace the :object and :init sections of each subgoal with the goal state of the prev subgoal (or the initial state of the non-decomposed problem)
        # 2) Generate a plan
        # 3) [OPTIONAL] Ground the plan in the scene graph
        for i, sub_goal in enumerate(sub_goals_pddl):
            self.current_phase = f"SUBGOAL_{i+1}:GENERATION"

            grounding_succesful = False
            planning_succesful = False

            sub_goal_dir = os.path.join(results_dir, f"sub_goal_{i+1}")
            sub_goal_file = os.path.join(sub_goal_dir, f"sub_goal_{i+1}.pddl")

            # If this is the first iteration (old_pddl_env is None), use the PDDLEnv of the main (non-decomposed problem) to get the initial state of the first subgoal
            if old_pddl_env is None:
                logger.debug(domain_file_path)
                logger.debug(results_dir)
                old_pddl_env = PDDLEnv(domain_file_path, results_dir, operators_as_actions=True)
                old_pddl_env.reset()

            # Get the final state of the previous environment
            obs = old_pddl_env.get_state()

            # Extract PDDL objects
            pddl_objects = []
            pddl_objects_str = "(:objects\n"
            for obj in obs.objects:
                obj = str(obj).split(":")
                pddl_objects.append((obj[0], obj[1]))
                pddl_objects_str += f"    {obj[0]} - {obj[1]}\n"
            pddl_objects_str += ")\n\n"

            # Extract PDDL predicates        
            state = obs.literals

            pddl_predicates = []
            pddl_predicates_str = "(:init\n"
            for literal in state:
                predicate_name = literal.predicate.name
                predicate_variables = []
                for variable in literal.variables:
                    predicate_variables.append(variable.split(":")[0])
                pddl_predicates_str += f"    ({predicate_name} {' '.join(predicate_variables)})\n"
                pddl_predicates.append((predicate_name, *predicate_variables))
            pddl_predicates_str += ")\n\n"

            # In the sub-goal problem file, replace the :objects and :init sections with the goal state of the previous sub-goal
            sub_goal_pddl = open(sub_goal_file, "r").read()
            
            # Replace the whole of the :objects and :init sections of the sub-goal with the goal state of the previous sub-goal
            sub_goal_pddl = sub_goal_pddl.replace(sub_goal_pddl[sub_goal_pddl.index("(:objects"):sub_goal_pddl.index("(:init")], pddl_objects_str)
            sub_goal_pddl = sub_goal_pddl.replace(sub_goal_pddl[sub_goal_pddl.index("(:init"):sub_goal_pddl.index("(:goal")], pddl_predicates_str)

            # Save the modified sub-goal
            with open(sub_goal_file, "w") as f:
                f.write(sub_goal_pddl)

            self.current_phase = f"SUBGOAL_{i+1}:PLANNING"

            # Initialize env
            new_pddl_env = PDDLEnv(domain_file_path, sub_goal_dir, operators_as_actions=True)

            output_plan_file_path = os.path.join(sub_goal_dir, f"plan_{i+1}.txt")

            # Compute plan on env
            plan, pddlenv_error_log, planner_error_log, planner_statistics = plan_with_output(domain_file_path, sub_goal_dir, output_plan_file_path, env=new_pddl_env)
            
            if plan is not None and plan:       
                
                planning_succesful = True
                
                # Verify with VAL
                translated_plan_path = os.path.join(sub_goal_dir, f"translated_plan_{i+1}.txt")

                # Translate the plan into a format parsable by VAL
                translate_plan(output_plan_file_path, translated_plan_path)

                logger.info("Validating the plan")
                self.current_phase = f"SUBGOAL_{i+1}:VAL:VALIDATION"
                
                # Use VAL to validate the plan
                val_succesful, val_log = VAL_validate(domain_file_path, sub_goal_file, translated_plan_path)
                logger.info(f"Result: {val_succesful}")
                logger.debug(val_log)

                logger.info("Validating grounding")
                # Use VAL to ground the plan
                val_ground_succesful, val_ground_log = VAL_ground(domain_file_path, sub_goal_file)
                logger.info(f"Result: {val_ground_succesful}")
                logger.debug(val_ground_log)

                grounding_succesful = val_succesful and val_ground_succesful

                self.current_phase = f"SUBGOAL_{i+1}:VAL:GROUNDING"
                save_statistics(
                    dir=results_dir,
                    workflow_iteration=0,
                    phase=self.current_phase,
                    plan_successful=planning_succesful,
                    pddlenv_error_log=pddlenv_error_log,
                    planner_error_log=planner_error_log,
                    VAL_validation_log=val_log,
                    VAL_grounding_log=val_ground_log,
                    planner_statistics=planner_statistics
                )


                # If this experiment requires it, try grounding the plan in the real scene graph
                if grounding_succesful and self.ground_in_sg:
                    self.current_phase = f"SUBGOAL_{i+1}:SCENE_GRAPH:GROUNDING"
                    logger.info("Grounding in scene graph")

                    # Convert the scene graph into a format readable by the grounder
                    extracted_locations_dictionary = convert_JSON_to_locations_dictionary(pruned_sg)

                    # Save the extracted locations dictionary to file
                    extracted_locations_dictionary_file_path = os.path.join(sub_goal_dir, "extracted_locations_dictionary.json")
                    with open(extracted_locations_dictionary_file_path, "w") as f:
                        json.dump(extracted_locations_dictionary, f, indent=4)

                    grounding_success_percentage, grounding_error_log = verify_groundability_in_scene_graph(
                        plan, 
                        None, 
                        domain_file_path=domain_file_path, 
                        problem_dir=sub_goal_dir, 
                        move_action_str="move_to",
                        location_relation_str="at",
                        location_type_str="room",
                        initial_robot_location=initial_robot_location,
                        pddlgym_environment = new_pddl_env,
                        locations_dictionary = extracted_locations_dictionary
                    )
                    
                    logger.info(f"Result: {grounding_success_percentage}")
                    logger.debug(grounding_error_log)

                    grounding_succesful = grounding_success_percentage == 1

                    save_statistics(
                        dir=results_dir,
                        workflow_iteration=0,
                        phase=self.current_phase,
                        plan_successful=planning_succesful,
                        pddlenv_error_log=pddlenv_error_log,
                        planner_error_log=planner_error_log,
                        VAL_validation_log=val_log,
                        VAL_grounding_log=val_ground_log,
                        grounding_success_percentage=grounding_success_percentage,
                        scene_graph_grounding_log=grounding_error_log
                    )

                if not grounding_succesful:
                    plans.append(plan)
                    return domain_file_path, pruned_sg, problem_pddl_path, sub_goals_file_paths, True, False, plans, self.current_phase, grounding_error_log
                
            else:
                error_log = pddlenv_error_log if pddlenv_error_log is not None else planner_error_log
                
                save_statistics(
                    dir=results_dir,
                    workflow_iteration=0,
                    phase=self.current_phase,
                    plan_successful=planning_succesful,
                    pddlenv_error_log=pddlenv_error_log,
                    planner_error_log=planner_error_log
                )
                
                return domain_file_path, pruned_sg, problem_pddl_path, sub_goals_file_paths, False, False, plans, f"PLANNING:SUBGOAL_{i}", error_log
                    
            plans.append(plan)
            new_pddl_env = old_pddl_env
            logger.info(f"Sub-goal {i+1} completed successfully")
            
        return domain_file_path, pruned_sg, problem_pddl_path, sub_goals_file_paths, planning_succesful, grounding_succesful, plans, "", ""
            
