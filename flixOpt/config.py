import os
from typing import Optional, Literal, Annotated
import logging
from dataclasses import dataclass, is_dataclass, fields

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


def dataclass_from_dict(cls, data: dict):
    """
    Recursively initialize a dataclass from a dictionary.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} must be a dataclass")

    # Build kwargs for the dataclass constructor
    kwargs = {}
    for field in fields(cls):
        field_name = field.name
        field_type = field.type
        field_value = data.get(field_name)

        # If the field type is a dataclass and the value is a dict, recursively initialize
        if is_dataclass(field_type) and isinstance(field_value, dict):
            kwargs[field_name] = dataclass_from_dict(field_type, field_value)
        else:
            kwargs[field_name] = field_value  # Pass as-is if no special handling is needed

    return cls(**kwargs)


@dataclass()
class ValidatedConfig:
    def __setattr__(self, name, value):
        if field := self.__dataclass_fields__.get(name):
            if metadata := getattr(field.type, '__metadata__', None):
                assert metadata[0](value), f'Invalid value passed to {name!r}: {value=}'
        super().__setattr__(name, value)


@dataclass
class LoggingConfig(ValidatedConfig):
    level: Annotated[Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], lambda level: level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']]
    file: Annotated[str, lambda file: isinstance(file, str)]
    rich: Annotated[bool, lambda rich: isinstance(rich, bool)]


@dataclass
class ModelingConfig(ValidatedConfig):
    BIG: Annotated[int, lambda x: isinstance(x, int)]
    EPSILON: Annotated[float, lambda x: isinstance(x, float)]
    BIG_BINARY_BOUND: Annotated[int, lambda x: isinstance(x, int)]


@dataclass
class ConfigSchema(ValidatedConfig):
    config_name: Annotated[str, lambda x: isinstance(x, str)]
    logging: LoggingConfig
    modeling: ModelingConfig


class CONFIG(ConfigSchema):
    _default_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    _instance = None

    @classmethod
    def load_config(cls, config_file: Optional[str] = None):
        """
        Load the configuration, merging user-provided config with defaults.

        :param config_file: Path to the user's config file (optional).
        :return: Configuration dictionary.
        """
        # Load the default config
        with open(cls._default_config_path, "r") as file:
            config = yaml.safe_load(file)

        if config_file is None:
            cls._instance = dataclass_from_dict(ConfigSchema, config)
        elif not os.path.exists(config_file):          # If the user provides a custom config, merge it with the defaults
            logger.error(f"No user config file found at {config_file}. Default config will be used.")
        else:
            with open(config_file, "r") as user_file:
                user_config = yaml.safe_load(user_file)
                config = merge_configs(config, user_config)
                logger.info(f"Loaded user config from {config_file}")
            try:
                cls._instance =  dataclass_from_dict(ConfigSchema, config)
            except AssertionError as e:
                logger.critical(
                    f'Invalid config file: {e}. \nPlease check your config file "{config_file}" and try again, or use the default config.')
                raise e



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
