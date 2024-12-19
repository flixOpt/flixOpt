"""
This module bundles all common functionality of flixOpt and sets up the logging
"""

from .commons import *
setup_logging(default_level=CONFIG['logging']['level'],
              log_file=CONFIG['logging']['file'],
              use_rich_handler=CONFIG['logging']['rich'])
