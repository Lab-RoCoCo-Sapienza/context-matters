from agent import Agent
from utils import *

agent = Agent(api_key="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")

def generate_pddl_domain(task_file):
    """
    Genera il file PDDL per il dominio basandosi su una descrizione testuale del task.
    """
    prompt = """
    Role: You are an excellent domain generator. Given a description of domain knowledge, you can generate a PDDL domain file.

    Example: 
    A robot in a household environment can perform the following actions on various objects.
    For instance, consider the action "mop floor":
    - Natural Language Description:
        For mopping the floor, the agent must be in the room and have the mop in hand.
        The mop must be clean and the floor must not be clean.
        After performing the action, the floor becomes clean, but the mop becomes dirty and the agent’s battery is no longer full.

    - Corresponding PDDL Definition (example_domain.pddl):
    -------------------------------------------------------
    (define (domain household)
      (:requirements :strips)
      (:predicates
        (agent-at ?r)
        (has ?agent ?object)
        (clean ?object)
        (floor-clean ?r)
        (battery-full)
      )
    
    (:action mop_floor
    :parameters (?a - agent ?i - item ?r - room)
    :precondition (and
    (agent_at ?a ?r)
    (item_is_mop ?i)
    (item_pickable ?i)
    (agent_has_item ?a ?i)
    (mop_clean ?i)
    (not(floor_clean ?r))
    )
    :effect (and
    (floor_clean ?r)
    (not(mop_clean ?i))
    (not(battery_full ?a))
    )
    )
    )
    """

    question = f"""
    Instruction:
    Extract new object types and actions from the following description and generate a corresponding PDDL domain file.

    <task>
    A new domain has the following new object types and actions.
    Please generate a corresponding PDDL domain file that incorporates these elements and respects the provided preconditions and effects.
    Write only the PDDL domain and nothing more.
    """

    task_description = Path(task_file).read_text()
    question = question.replace("<task>", task_description)

    domain_pddl = agent.llm_call(prompt, question)
    return domain_pddl.replace("`", "").replace("pddl", "")

# Generazione del dominio
domain = generate_pddl_domain("dataset/Allensville/problem_1/task.txt")
print(domain)


def prune_scene_graph(scene_graph_path, goal_description_path):
    """
    Riduce il Scene Graph (SG) mantenendo solo gli oggetti rilevanti per il goal dato.
    """
    prompt = """
    Role: You are an excellent assistant in pruning SGs with a list
    of SG items and a goal description.

    Example: A SG can be programmed as a nested Python dictionary such as 
    <filtered_graph>
    For accomplishing the put the glass into the sink, the relevant items are glass and the sink in the kitchen.
    """

    # Carica il Scene Graph originale
    scene_graph = read_graph_from_path(Path(scene_graph_path))
    filtered_sg = filter_graph(scene_graph, ["rooms", "objects"])
    prompt = prompt.replace("<filtered_graph>", str(filtered_sg))

    question = """Instruction: Given a new <new_graph> and a <GOAL>, please prune the SG by keeping the relevant items. 
    Write only the pruned graph respecting the original structure.
    """

    # Carica la descrizione del task
    goal_description = Path(goal_description_path).read_text()

    # Carica il nuovo Scene Graph da filtrare
    new_sg = read_graph_from_path(Path(scene_graph_path))
    filtered_new_sg = filter_graph(new_sg, ["rooms", "objects"])
    
    question = question.replace("<new_graph>", str(filtered_new_sg)).replace("<GOAL>", goal_description)

    # Esegui il pruning
    pruned_sg = agent.llm_call(prompt, question)
    return pruned_sg

# Pruning del Scene Graph
pruned_sg = prune_scene_graph(
    "dataset/Allensville/problem_1/Allensville.npz",
    "dataset/Allensville/problem_1/description.txt"
)

print(pruned_sg)


def generate_pddl_problem(scene_graph_path, goal_description_path, domain_pddl_path):
    """
    Genera un file PDDL del problema basandosi su:
    - Un Scene Graph prunato (SG)
    - Una descrizione degli obiettivi (goal)
    - Il file di dominio PDDL generato in precedenza
    """

    prompt = """
    Role: You are an excellent problem generator. Given a Scene Graph (SG) and a desired goal,
    you can generate a PDDL problem file.

    Example:
    Given an <example_graph>, an <example_goal>, and using the predicates 
    defined in <example_domain>, a corresponding PDDL problem file looks like:

    (define (problem house_cleaning)
        (:domain household)
        (:objects
            robot - agent
            kitchen living_room - room
            mop - item
            sink_1 - item
            cola_can banana_peel - item
            rubbish_bin - item
        )
        (:init
            (agent_at robot kitchen)
            (item_is_mop mop)
            (item_pickable mop)
            (item_pickable cola_can)
            (item_pickable banana_peel)
            (floor_clean living_room)
            (not (floor_clean kitchen))
            (mop_clean mop)
        )
        (:goal
            (and
                (item_disposed cola_can)
                (item_disposed banana_peel)
                (floor_clean kitchen)
                (floor_clean living_room)
                (mop_clean mop)
            )
        )
    )
    
    """

    question = """

    Instruction:
    Given a new Scene Graph <scene_graph>, a new goal description <goal_description>, 
    and the predicates defined in <domain_pddl>, generate a new PDDL problem file.
    Write only the PDDL problem and nothing more.
    """

    # Carica il Scene Graph prunato
    scene_graph = read_graph_from_path(Path(scene_graph_path))
    filtered_sg = filter_graph(scene_graph, ["rooms", "objects"])

    prompt = prompt.replace("<example_graph>", """{
                            {
    "floor": {},
    "mop": {},
    "water": {},
    "detergent": {}
}

""")
    prompt = prompt.replace("<example_goal>", "clean the floor")

    prompt = prompt.replace("<example_domain>", """
(define (domain cleaning)
  (:requirements :strips)
  (:predicates
    (agent_at ?r)
    (has ?agent ?object)
    (clean ?object)
    (floor_dirty ?r)
    (floor_clean ?r)
    (mop_wet ?m)
    (mop_dry ?m)
  )

  (:action wet_mop
    :parameters (?a - agent ?m - mop ?b - bucket)
    :precondition (and (has ?a ?m) (has ?b water) (mop_dry ?m))
    :effect (and (mop_wet ?m) (not (mop_dry ?m)))
  )

  (:action mop_floor
    :parameters (?a - agent ?m - mop ?r - room)
    :precondition (and (agent_at ?r) (has ?a ?m) (mop_wet ?m) (floor_dirty ?r))
    :effect (and (floor_clean ?r) (not (floor_dirty ?r)) (mop_dry ?m))
  )
)
""")


    # Carica la descrizione del goal
    goal_description = Path(goal_description_path).read_text()

    # Carica il dominio PDDL
    domain_pddl = Path(domain_pddl_path).read_text()

    # Personalizza il prompt
    question = question.replace("<scene_graph>", str(filtered_sg))
    question = question.replace("<goal_description>", goal_description)
    question = question.replace("<domain_pddl>", domain_pddl)

    # Esegui la generazione del PDDL del problema
    problem_pddl = agent.llm_call(prompt, question)
    return problem_pddl.replace("`", "").replace("pddl", "")

# Generazione del problema PDDL
problem_pddl = generate_pddl_problem(
    "dataset/Allensville/problem_1/Allensville.npz",
    "dataset/Allensville/problem_1/description.txt",
    "dataset/Allensville/domain.pddl"
)

print(problem_pddl)

with open("problem.pddl","w") as f:
    f.write(problem_pddl)


def decompose_pddl_goal(problem_pddl_path, domain_pddl_path):
    """
    Scompone gli obiettivi di un file PDDL del problema in una sequenza di sotto-obiettivi (sub-goals).
    """

    prompt = """
    Role: You are an excellent assistant in decomposing long-term goals. Given a PDDL problem file, 
    you can decompose the goal states into a sequence of sub-goals.

    Example:
    Given an problem.pddl:
    
    (define (problem house_cleaning)
    (:domain cleaning)
    (:objects
        robot - agent
        kitchen living_room - room
        mop - item
        sink - item
        bucket - item
        water - liquid
    )
    (:init
        (agent_at robot kitchen)
        (has bucket water)
        (mop_dry mop)
        (floor_dirty kitchen)
        (floor_dirty living_room)
    )
    (:goal
        (and
            (floor_clean kitchen)
            (floor_clean living_room)
        )
    )
)


, the goal states can be decomposed into a sequence of example sub-goals 
    Using the predicates defined in domain, the example sub-goals can be formulated as:

    sub-goal_1.pddl:
    (define (problem sub-goal-1)
        (:domain cleaning)
        (:objects
            robot - agent
            room - room
            mop - item
            bucket - container
            water - liquid
        )
        (:init
            (agent_at room)
            (has bucket water)
            (mop_dry mop)
            (floor_dirty room)
        )
        (:goal
            (and
                (mop_wet mop)
            )
        )
    )

    sub-goal_2.pddl:
    (define (problem sub-goal-2)
        (:domain cleaning)
        (:objects
            robot - agent
            room - room
            mop - item
            bucket - container
            water - liquid
        )
        (:init
            (agent_at room)
            (has bucket water)
            (mop_wet mop)
            (floor_dirty room)
        )
        (:goal
            (and
                (floor_clean room)
            )
        )
    )
"""
    # Carica il file PDDL del problema
    problem_pddl = Path(problem_pddl_path).read_text()

    # Carica il file PDDL del dominio
    domain_pddl = Path(domain_pddl_path).read_text()

    # Personalizza il prompt con i file PDDL reali
    question = f"""
    Given the following problem PDDL:
    {problem_pddl}

    And the following domain PDDL:
    {domain_pddl}

    Decompose the goal into sub-goals in pddl
    """

    # Esegui la generazione dei sub-goal PDDL
    sub_goals_pddl = agent.llm_call(prompt, question)
    return sub_goals_pddl.replace("`", "").replace("pddl", "")

# Generazione dei sub-goal PDDL
sub_goals_pddl = decompose_pddl_goal(
    "problem.pddl",
    "dataset/Allensville/domain.pddl"
)

print(sub_goals_pddl)
