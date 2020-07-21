import importlib
import logging
import subprocess as sp

logger = logging.getLogger("Advanced Import")


def install_if_not_exist(package_name, scope, imported_name=None, import_name=None):
    if import_name is None:
        import_name = package_name
    if imported_name is None:
        if len(import_name.split(".")) > 1:
            imported_name = "".join(*import_name.split(".")[0])
        else:
            imported_name = package_name

    try:
        module = importlib.import_module(import_name)
        scope[imported_name] = module
    except ImportError as e:
        logger.exception(e)
        logger.info(f"Module {package_name} not installed. Installing...")
        return_code = sp.call(f'pip install {package_name}')
        if return_code == 0:
            logger.info(f"Package {package_name} installed successfully!")
            module = importlib.import_module(package_name)
            globals()[imported_name] = module
        else:
            logger.error(f"Package {package_name} installation failed!")
            raise ImportError(
                f"Module {package_name} cannot be installed automatically. Please fix it manually.")
