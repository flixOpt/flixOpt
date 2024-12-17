"""
This module bundles all common functionality of flixOpt and sets up the logging
"""

from .commons import *
setup_logging('INFO', use_rich_handler=False)
