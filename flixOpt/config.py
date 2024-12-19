import os
from typing import Optional, Union, TypedDict, Literal
import logging

import yaml

logger = logging.getLogger('flixOpt')


def merge_configs(defaults: dict, overrides: dict) -> dict:
    """
    Merge the default configuration with user-provided overrides.

    :param defaults: Default configuration dictionary.
    :param overrides: User configuration dictionary.
    :return: Merged configuration dictionary.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in defaults and isinstance(defaults[key], dict):
            # Recursively merge nested dictionaries
            defaults[key] = merge_configs(defaults[key], value)
        else:
            # Override the default value
            defaults[key] = value
    return defaults


class LoggingConfig(TypedDict):
    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    file: str
    rich: bool


class ModelingConfig(TypedDict):
    BIG: int
    EPSILON: float
    BIG_BINARY_BOUND: int


class ConfigSchema(TypedDict):
    logging: LoggingConfig
    modeling: ModelingConfig


def load_config(config_file: Optional[str] = None) -> ConfigSchema:
    """
    Load the configuration, merging user-provided config with defaults.

    :param config_file: Path to the user's config file (optional).
    :return: Configuration dictionary.
    """
    # Path to the default package config
    default_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    # Load the default config
    with open(default_config_path, "r") as file:
        config = yaml.safe_load(file)

    if config_file is None:
        return ConfigSchema(**config)

    # If the user provides a custom config, merge it with the defaults
    elif not os.path.exists(config_file):
        logger.error(f"No user config file found at {config_file}. Default config will be used.")
    else:
        with open(config_file, "r") as user_file:
            user_config = yaml.safe_load(user_file)
            config = merge_configs(config, user_config)
            logger.info(f"Loaded user config from {config_file}")
        try:
            return ConfigSchema(**config)
        except TypeError as e:
            logger.critical(f'Invalid config file: {e}. \nPlease check your config file "{config_file}" and try again.')
            raise e


# Load the configuration and make it globally accessible
CONFIG: ConfigSchema = load_config()
