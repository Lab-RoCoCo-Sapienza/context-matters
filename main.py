import argparse
from omegaconf import OmegaConf
from src.pipelines import get_pipeline
import os

if __name__ == "__main__":
    
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Run execution pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        help="Directory of the repository"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory of the dataset"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory of the results"
    )
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    data_dir = args.data_dir
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    config_path = args.config
    
    # -------------------- Initialization --------------------
    
    cfg = OmegaConf.load(config_path)
    
    cfg_pipeline = cfg.pipeline
    cfg_agents = cfg.agents
    cfg_methods = cfg.methods
    cfg_splits = cfg.splits
    cfg_params = cfg.parameters
    
    kwargs = {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "results_dir": results_dir
    }
    
    kwargs.update(OmegaConf.to_container(cfg_pipeline, resolve=True))
    kwargs.update(OmegaConf.to_container(cfg_agents, resolve=True))
    kwargs.update(OmegaConf.to_container(cfg_params, resolve=True))
    
    splits = []
    for split in cfg_splits:
        splits.append(split)
        
    kwargs["splits"] = splits
        
    methods = []
    for method in cfg_methods:
        method = get_pipeline(method, **kwargs)
        methods.append(method)
        method.run()
        
    for method in methods:
        print(f"-------------------Running the {method.name} Pipeline -------------------\n")
        method.run()