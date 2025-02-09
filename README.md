# context-matters

# Install

1) Make sure git-lfs is installed:
```
sudo apt-get install git-lfs
```

2) Clone this repo
```
git clone --recurse-submodules https://github.com/Lab-RoCoCo-Sapienza/context-matters.git
```

3) Setup a virtual environment and install the requirements.txt
```
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```
   
4) Install the pddlgym_planners submodule
```
cd third-party/pddlgym_planners/
pip install -e
```
   
