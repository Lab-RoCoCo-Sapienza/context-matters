import os
from typing import Dict, List
class BasePipeline:
    def __init__(self, **kwargs):
        self.base_dir: str  = kwargs["base_dir"]
        self.data_dir: str = kwargs["data_dir"]
        self.results_dir: str = kwargs["results_dir"]
        self.splits: List[str]  = kwargs["splits"]
        self.generate_domain: bool = kwargs["generate_domain"]
        self.ground_in_sg: bool = kwargs["ground_in_sg"]
        self.model: str = kwargs["model"]
        
        self.api_key: str = os.getenv("API_KEY")
        
        self.scenes_per_task: Dict = {
            "dining_setup": ["Allensville", "Parole", "Shelbiana"],
            "house_cleaning": ["Allensville", "Parole", "Shelbiana"],
            "laundry": ["Kemblesville"],
            "office_setup": ["Allensville", "Parole", "Shelbiana"],
            "other_1": [
                "Beechwood",
                "Benevolence",
                "Coffeen",
                "Collierville",
                "Corozal",
                "Cosmos"
            ],
            "other_2": [
                "McDade",
                "Merom",
                "Mifflinburg", 
                "Muleshoe",
                "Newfields",
                "Noxapater"
            ],
            "pc_assembly": ["Allensville", "Parole", "Shelbiana"],
        }
        self.problems_per_task: Dict = {
            "dining_setup": 6,
            "house_cleaning": 6,
            "laundry": 6,
            "office_setup": 8,
            "other_1": 10,
            "other_2": 9,
            "pc_assembly": 3,
        }

    def run(self):
        raise NotImplementedError("Method not implemented")
