import os
import yaml
CONFIG_PATH = "config/"

def load_config(args):
    if args.channel == 'RGBNIRFINE':
        with open(os.path.join(CONFIG_PATH, 'config_RGBNIR_FINE.yml')) as file:
            config = yaml.safe_load(file)
    elif args.channel == 'RGBNIRFINEPATCH128':
        with open(os.path.join(CONFIG_PATH, 'config_RGBNIR_FINEPATCH128.yml')) as file:
            config = yaml.safe_load(file)
    elif args.channel == 'RGBNIRSWIRFINE':
        with open(os.path.join(CONFIG_PATH, 'config_RGBNIRSWIR_FINE.yml')) as file:
            config = yaml.safe_load(file)
    elif args.channel == 'RGBNIRDISTILFINE':
        with open(os.path.join(CONFIG_PATH, 'config_RGBNIR_DISTIL_FINE.yml')) as file:
            config = yaml.safe_load(file)
    elif args.channel == 'RGBNIRCOARSE':
        with open(os.path.join(CONFIG_PATH, 'config_RGBNIR_COARSE.yml')) as file:
            config = yaml.safe_load(file)
    elif args.channel == 'RGBNIRSWIRCOARSE':
        with open(os.path.join(CONFIG_PATH, 'config_RGBNIRSWIR_COARSE.yml')) as file:
            config = yaml.safe_load(file)
    elif args.channel == 'RGBNIRSWIRDISTILFINE':
        with open(os.path.join(CONFIG_PATH, 'config_RGBNIRSWIR_DISTIL_FINE.yml')) as file:
            config = yaml.safe_load(file)
    else:
        raise NotImplementedError("Please Check the import config training data")

    return config