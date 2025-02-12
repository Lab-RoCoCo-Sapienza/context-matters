from utils import *
import os
from pprint import pprint
import json

dataset_dir = "dataset"

for root_dir, dirs, files in os.walk(dataset_dir):
    for dir in dirs:
        if "problem" in dir:
            print(dir)
            path = os.path.join(root_dir, dir)
            print(path)
            graph = read_graph_from_path(Path(os.path.join(path, path.split("/")[-2] + ".npz")))
            print(graph)