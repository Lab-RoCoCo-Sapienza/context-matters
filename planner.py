from pprint import pprint

import pddlgym
from pddlgym.core import PDDLEnv
from pddlgym_planners.fd import FD

FAST_DOWNWARD_PATH = "downward/fast-downward.py"



def run_planner_FD(domain_file_path, problem_dir):

    # Create PDDLEnv
    env = PDDLEnv(domain_file_path, problem_dir, operators_as_actions = True)

    # Use only first problem in directory
    env.fix_problem_index(0)
    
    # Reset environment
    obs,  debug_info = env.reset()

    # Create planner
    planner = FD()

    # Plan
    try:
        plan = planner(env.domain, obs)
    except Exception as e:
        print(e)
        plan = None

    return plan



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
    #print("Action: "+str(action)) 
    #print("Observation:")
    #pprint(str(obs))
    #print("Reward: "+str(reward))
    #print("Done: "+str(done))
    #print("Truncated: "+str(truncated))
    
    return env, obs
