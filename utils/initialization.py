import yaml
import torch as th

from utils.custom_policies import CustomCNN
from utils.schedules import linear_schedule


def resolve_activation_fn(name):
    """Map activation function names from config to torch classes."""
    
    activation_map = {
        "ReLU": th.nn.ReLU,
        "Tanh": th.nn.Tanh,
        "ELU": th.nn.ELU,
        "LeakyReLU": th.nn.LeakyReLU,
        "GELU": th.nn.GELU,
        "SiLU": th.nn.SiLU,
    }
    
    if name not in activation_map:
        raise ValueError(f"Unsupported activation function in config: {name}")
    
    return activation_map[name]


def resolve_feature_extractor(name):
    """Map feature extractor names from config to classes."""
    
    extractor_map = {
        "CustomCNN": CustomCNN,
    }
    
    if name not in extractor_map:
        raise ValueError(f"Unsupported features_extractor_class in config: {name}")
    
    return extractor_map[name]


def config_loader(environment, algorithm="ppo"):
    
    """Load policy settings, algorithm parameters, and train steps from YAML config."""
    
    config_path = f"configs/config_{environment}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {config_path}")

    n_train_steps = config.get("n_train_steps")
    if n_train_steps is None:
        raise KeyError(f"Missing 'n_train_steps' in {config_path}")

    alg_params = config.get(f"{algorithm}_params")
    if alg_params is None:
        raise KeyError(f"Missing '{algorithm}_params' in {config_path}")
    alg_params = alg_params.copy()

    schedule_params = config.get("schedule_params", [])
    for param in schedule_params:
        if param in alg_params:
            alg_params[param] = linear_schedule(alg_params[param])

    policy_config = config.get("policy", {})
    policy_name = policy_config["name"]

    policy_kwargs = None

    # Preferred schema: full kwargs block under policy.kwargs
    if "kwargs" in policy_config:
        policy_kwargs = policy_config["kwargs"].copy()

        if "activation_fn" in policy_kwargs and isinstance(policy_kwargs["activation_fn"], str):
            policy_kwargs["activation_fn"] = resolve_activation_fn(policy_kwargs["activation_fn"])

        if ("features_extractor_class" in policy_kwargs
            and isinstance(policy_kwargs["features_extractor_class"], str)):
            policy_kwargs["features_extractor_class"] = resolve_feature_extractor(
                policy_kwargs["features_extractor_class"]
            )
        
        if "net_arch" in policy_kwargs:
            # Ensure net_arch is in the correct format (dict with 'pi' and 'vf')
            if isinstance(policy_kwargs["net_arch"], dict):
                if "pi" not in policy_kwargs["net_arch"] or "vf" not in policy_kwargs["net_arch"]:
                    raise ValueError(f"'net_arch' must contain 'pi' and 'vf' keys in {config_path}")
            else:
                raise ValueError(f"'net_arch' must be a dict with 'pi' and 'vf' keys in {config_path}")
            
            policy_kwargs["net_arch"] = {
                "pi": policy_kwargs["net_arch"]["pi"],
                "vf": policy_kwargs["net_arch"]["vf"],
            }

    return policy_name, policy_kwargs, alg_params, int(n_train_steps)
