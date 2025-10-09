<div align="center">
<h1 style="font-size: 50px">Context Matters!</h1> 
<img src="assets/cm.png" width=40%>
<h2>Relaxing Goals with LLMs for Feasible 3D Scene Planning</h2>

<div>
   
[![flat](https://img.shields.io/badge/Project-Website-blue)](https://lab-rococo-sapienza.github.io/context-matters/)
[![arxiv paper](https://img.shields.io/badge/arXiv-pdf-red)](https://arxiv.org/abs/2506.15828)
[![license](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
![flat](https://img.shields.io/badge/python-3.10+-green)
![flat](https://img.shields.io/badge/Ubuntu-22.04-E95420)
![flat](https://img.shields.io/badge/Ubuntu-24.04-E95420)


</div>
<h5>Image credits to: https://www.adexchanger.com/comic-strip/adexchanger-context-matters/</h5>
</div>

# Install

1. Clone this repo
```
git clone --recurse-submodules https://github.com/Lab-RoCoCo-Sapienza/context-matters.git
```

2. Setup a virtual environment (conda, venv, ...) and install the requirements.txt

```
pip install -r requirements.txt
```
   
3. Install the pddlgym_planners submodule
```
cd third-party/pddlgym_planners/
pip install -e .
```
   
4. Install ollama
```
sudo snap install ollama
```

5. Build VAL
```
bash third-party/VAL/scripts/linux/build_linux64.sh build Release
cd third-party/VAL/build/linux64/Release
make
```
# Generate dataset
1. Download both the "medium" and "tiny" data splits from the original [3DSG repo](https://github.com/StanfordVL/3DSceneGraph)
2. Move into `dataset/3dscenegraph` the following .npz files: `Allensville`, `Kemblesville`, `Klickitat`, `Lakeville`, `Leonardo`, `Lindenwood`, `Markleeville`, `Marstons`, `Parole`, `Shelbiana`
3. Make sure the virtual environment is activated, then run 
```
export OPENAI_API_KEY=<your OpenAI API key>
python3 dataset/dataset_creation.py
```

# Run

1. Export your OpenAI key
```
export OPENAI_API_KEY=<your OpenAI API key>
```

2. Our code supports `hydra` modular configuration. This means that to run any baseline, simple execute the following code:
```
python3 main.py pipeline=<name>
```
with options `cm`, `delta`, `sayplan` and `llm_planner`. This will execute the corresponding baseline with default parameters.
To change parameters, you can either edit the corresponding config file directly, or override them in the terminal, e.g. `python3 main.py pipeline=<name> pipeline.max_debug_attemps=3`

Key parameters are:
- `generate_domain`: Controls whether the pipeline uses an LLM to automatically generates a PDDL domain or uses a pre-existing one
- `ground_in_sg`: Controls whether the pipeline performs a validation step to ensure the generated plan is physically possible in the world described by the scene graph
- `workflow_iterations`: Sets the number of self-correction attempts for the entire problem-solving workflow if the initial plan is invalid
- `pddl_gen_iterations`: Specifies the number of retry attempts for the LLM to generate a syntactically correct PDDL domain file

# Metrics
After running the pipelines, the results for each split will be stored in CSV files.
To aggregate the results, run:
```
python3 aggregate_results.py
```

This will generate a detailed metrics JSON file inside the `metrics_res_path` specified in `config.yaml`.

Some key performance metrics are:
- `overall_success_rate`: The percentage of tasks successfully solved with a valid and executable plan
- `grounding_success_rate`: The percentage of generated plans that were realistic and executable when checked against the environment
- `avg_plan_length_successful`: The average number of steps in a successful plan, measuring its efficiency
- `avg_no_nodes`: The average number of states explored by the planner, indicating its search effort
- `avg_no_relaxations`: The average number of times the problem was simplified to find a high-level plan


To plot and visualize the aggregated results, run:
```
python3 generate_plots.py
```

This will create plots inside the `metrics_res_path` and display both comparison and performance tables in the terminal
