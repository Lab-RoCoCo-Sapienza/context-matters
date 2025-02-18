from pprint import pprint

import pddlgym
from pddlgym.core import PDDLEnv
from pddlgym_planners.fd import FD

FAST_DOWNWARD_PATH = "downward/fast-downward.py"



def run_planner_FD(domain_file_path, problem_dir, env=None):

    if env is None:
        try:
            # Create PDDLEnv
            env = PDDLEnv(domain_file_path, problem_dir, operators_as_actions = True)
        except Exception as e:
            print("Exception in PDDLEnv: "+str(e))
            return None, str(e), None

    # Use only first problem in directory
    env.fix_problem_index(0)
    
    try:
        # Reset environment
        obs,  debug_info = env.reset()
    except Exception as e:
        print("Exception in PDDLEnv reset: "+str(e))
        return None, str(e), None
        
    # Create planner
    planner = FD()

    # Plan
    try:
        plan = planner(env.domain, obs)
    except Exception as e:
        print("Exception in FD planner: "+str(e))
        return None, None, str(e)

    return plan, None, None


def plan_with_output(domain_file_path, problem_dir, plan_file_path, env=None):

    # PLANNING #
    
    print("\n\n\tPerforming planning...")
    plan, pddlenv_error_log, planner_error_log = run_planner_FD(domain_file_path, problem_dir, env)

    # Save planner output
    with open(plan_file_path, "w") as file:
        if plan is not None:    
            file.write(str(plan))
        elif pddlenv_error_log is not None:
            file.write(pddlenv_error_log)
        elif planner_error_log is not None:
            file.write(planner_error_log)

    return plan, pddlenv_error_log, planner_error_log


def initialize_pddl_environment(domain_file_path, problem_dir):
    # Create PDDLEnv
    env = PDDLEnv(domain_file_path, problem_dir, operators_as_actions = True)

    problem_index = 0

    # Use only first problem in directory
    env.fix_problem_index(problem_index)

    # Produce initial observation
    initial_observation, debug_info = env.reset()

    return env, initial_observation



def execute_plan(env, plan, domain_file_path, problem_dir):

    # Create planner
    planner = FD()

    # Plan
    plan = planner(env.domain, obs)

    # Rollout
    for act in plan:
        obs, reward, done, truncated, debug_info = env.step(act)
        print("Action: "+str(act)) 
        print("Observation: "+str(obs))
        print("Reward: "+str(reward))
        print("Done: "+str(done))
        print("Truncated: "+str(truncated))
    
    return obs, env



def execute_action(env, action):

    # Execute action
    obs, reward, done, truncated, debug_info = env.step(action)
    
    return env, obs