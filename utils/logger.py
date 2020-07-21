""" Set up the logging settings.
Usage:
    from mlib.utils.logger import *
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    logger.debug/info/warning/error/critical("xxx")

Links:
    Official homepage: https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
    Official config setting: https://docs.python.org/3/library/logging.config.html#logging-config-api
    Comprehensive Logging using YAML Configuration: https://gist.github.com/kingspp/9451566a5555fb022215ca2b7b802f19
    Example: https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
"""
import os
import yaml
import logging
import logging.config

from pathlib2 import Path
from .basic import time_string
from .sysinfo import SysInfo

_package_directory = os.path.dirname(os.path.abspath(__file__))
_default_config_path = os.path.join(_package_directory, "logger.yaml")
_default_level = "INFO"
_str_logging_level = {"NOTSET": logging.NOTSET,
                      "DEBUG": logging.DEBUG,
                      "INFO": logging.INFO,
                      "ERROR": logging.ERROR,
                      "WARNING": logging.WARNING,
                      "CRITICAL": logging.CRITICAL}

__all__ = ["logging", "setup_logging"]


def setup_logging(level=None,
                  config_path=_default_config_path,
                  to_file=True,
                  log_file_path=None,
                  env_key='LOG_CFG'):
    """ Setup logging configuration
    Args:
        config_path: The yaml config file path.
        default_level: The default level of logging.
        env_key: An environment variable. You can set it in cmd before
            execute your python file to specify a config file in cmd.
    Usage:
        `LOG_CFG=my_logging.yaml python my_server.py`
    """
    system_info = SysInfo()
    if system_info.is_linux or system_info.is_macos:
        default_log_file_path = f"/tmp/miracle_debug_{time_string()}.log"
    else:
        default_log_file_path = str(
            Path.home() / "Downloads" / f"miracle_debug_{time_string()}.log")
    if log_file_path is not None:
        _parts = os.path.splitext(log_file_path)
        log_file_path = _parts[0]+"_"+time_string()+_parts[1]

    if level is not None:
        level = level.upper()

    path = config_path
    value = os.getenv(env_key, None)

    if value:
        path = value
    if os.path.exists(path):
        # Load the yaml file config
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        # Modify the loaded logging config
        if level is not None:
            config["root"]["level"] = level
        if to_file:
            config["root"]["handlers"].append("file_handler")
        log_file_path = default_log_file_path if log_file_path is None else log_file_path
        config["handlers"]["file_handler"]["filename"] = str(log_file_path)

        # Parse the config dict into real logging config
        logging.config.dictConfig(config)
    else:
        # Init the basic stream handler
        level = _default_level if level is not None else level
        logging.basicConfig(level=_str_logging_level[level])

        # Add the file log support
        if to_file:
            log_file_path = default_log_file_path if log_file_path is None else log_file_path
            fileHandler = logging.FileHandler(log_file_path)
            logging.getLogger().addHandler(fileHandler)


""" How to use a customized filter
class NoParsingFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('parsing')

logger.addFilter(NoParsingFilter())
"""
