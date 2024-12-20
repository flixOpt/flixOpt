import os
from typing import Optional, TypedDict, Literal
import logging

import yaml
from rich.logging import RichHandler
from rich.console import Console

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


class MultilineFormater(logging.Formatter):

    def format(self, record):
        message_lines = record.getMessage().split('\n')

        # Prepare the log prefix (timestamp + log level)
        timestamp = self.formatTime(record, self.datefmt)
        log_level = record.levelname.ljust(8)  # Align log levels for consistency
        log_prefix = f"{timestamp} | {log_level} |"

        # Format all lines
        first_line = [f'{log_prefix} {message_lines[0]}']
        if len(message_lines) > 1:
            lines = first_line + [f"{log_prefix} {line}" for line in message_lines[1:]]
        else:
            lines = first_line

        return '\n'.join(lines)


class ColoredMultilineFormater(MultilineFormater):
    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': '\033[32m',  # Green
        'INFO': '\033[34m',  # Blue
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[1m\033[31m',  # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        lines = super().format(record).splitlines()
        log_color = self.COLORS.get(record.levelname, self.RESET)

        # Create a formatted message for each line separately
        formatted_lines = []
        for line in lines:
            formatted_lines.append(f"{log_color}{line}{self.RESET}")

        return '\n'.join(formatted_lines)


def _get_logging_handler(log_file: Optional[str] = None,
                         use_rich_handler: bool = False) -> logging.Handler:
    """Returns a logging handler for the given log file."""
    if use_rich_handler and log_file is None:
        # RichHandler for console output
        console = Console(width=120)
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            omit_repeated_times=True,
            show_path=False,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s"))  # Simplified formatting

        return rich_handler
    elif log_file is None:
        # Regular Logger with custom formating enabled
        file_handler = logging.StreamHandler()
        file_handler.setFormatter(ColoredMultilineFormater(
            fmt="%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        return file_handler
    else:
        # FileHandler for file output
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(MultilineFormater(
            fmt="%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        return file_handler


def setup_logging(default_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
                  log_file: Optional[str] = 'flixOpt.log',
                  use_rich_handler: bool = False):
    """Setup logging configuration"""
    logger = logging.getLogger('flixOpt')  # Use a specific logger name for your package
    logger.setLevel(get_logging_level_by_name(default_level))
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(_get_logging_handler(use_rich_handler=use_rich_handler))
    if log_file is not None:
        logger.addHandler(_get_logging_handler(log_file, use_rich_handler=False))

    return logger


def get_logging_level_by_name(level_name: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']) -> int:
    possible_logging_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level_name.upper() not in possible_logging_levels:
        raise ValueError(f'Invalid logging level {level_name}')
    else:
        logging_level = getattr(logging, level_name.upper(), logging.WARNING)
        return logging_level


def change_logging_level(level_name: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']):
    logger = logging.getLogger('flixOpt')
    logging_level = get_logging_level_by_name(level_name)
    logger.setLevel(logging_level)
    for handler in logger.handlers:
        handler.setLevel(logging_level)
