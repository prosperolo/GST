from .humman import HuMMan

def get_dataset(cfg, name):
    dataset_type = cfg.data.dataset_type
    if dataset_type == "humman":
        return HuMMan(cfg, name)
    else:
        raise ValueError(f"Dataset {dataset_type} does not exist")