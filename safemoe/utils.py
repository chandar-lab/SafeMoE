import os

from omegaconf import OmegaConf


def load_config(path, overrides=()):
    """Load a YAML config (with optional, nestable `defaults: <file>` inheritance) and CLI overrides."""
    cfg = OmegaConf.load(path)
    if "defaults" in cfg:
        parent = load_config(os.path.join(os.path.dirname(path), cfg.pop("defaults")))
        cfg = OmegaConf.merge(parent, cfg)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))


def parse_data_spec(spec):
    """'pku:Cybercrime' -> ('pku', 'Cybercrime');  'file.json' -> ('file.json', None)."""
    if ":" in spec and not os.path.exists(spec):
        name, category = spec.split(":", 1)
        return name, category
    return spec, None
