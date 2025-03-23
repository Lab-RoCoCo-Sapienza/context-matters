from .base_pipeline import BasePipeline
from .delta_pipeline import DeltaPipeline
from .cm_pipeline import ContextMattersPipeline

def get_pipeline(
    name, **kwargs,
) -> BasePipeline:
    if name == "delta":
        return DeltaPipeline(**kwargs)
    elif name == "cm":
        return ContextMattersPipeline(**kwargs)
    else:
        raise ValueError(f"Unknown pipeline {name}")