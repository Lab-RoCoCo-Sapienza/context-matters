# context-matters

# Install

1) Clone this repo
```
git clone --recurse-submodules https://github.com/Lab-RoCoCo-Sapienza/context-matters.git
```

2) Setup a virtual environment (conda, venv, ...) and install the requirements.txt

```
pip install -r requirements.txt
```
   
4) Install the pddlgym_planners submodule
```
cd third-party/pddlgym_planners/
pip install -e
```
   
5) Install ollama
```
sudo snap install ollama
```

6) Build VAL
```
bash third-party/VAL/scripts/linux/build_linux64.sh build Release
cd third-party/VAL/build/linux64/Release
make
```


# Run
## Dataset creator
Make sure the virtual environment is activated, then run 
```
cd dataset/
python3 dataset_creation.py
```

## Main workflow
Make sure the virtual environment is activated.
Export the following environment variables
```
export BASE_DIR=/path/to/main/repo
export DATA_DIR=/path/to/dataset/repo
export RESULTS_DIR=/path/to/save/results
export OPENAI_API_KEY=<your OpenAI API key>
```

In `config/config.yaml` you can find all the config parameters of our architecture.
You can modify them by simply changing their values in the yaml file.

To run the architecture with one (CM/DELTA) or both models, simply run
```
./scripts/run.sh
```