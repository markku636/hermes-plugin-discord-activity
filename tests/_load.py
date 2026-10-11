"""Import ``discord-activity/`` the way Hermes does (a package loaded from a directory path)."""

import importlib.util
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "discord-activity"
NAME = "discord_activity_under_test"


def load():
    if NAME in sys.modules:
        return sys.modules[NAME]
    spec = importlib.util.spec_from_file_location(NAME, PLUGIN_DIR / "__init__.py",
                                                  submodule_search_locations=[str(PLUGIN_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
    return module


pkg = load()
render = sys.modules[NAME + ".render"]
tracker = sys.modules[NAME + ".tracker"]
discord_api = sys.modules[NAME + ".discord_api"]
