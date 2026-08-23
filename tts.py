from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _run_package_main() -> None:
    package_file = Path(__file__).resolve().parent / "src" / "tts" / "__init__.py"
    spec = spec_from_file_location("tts_package", package_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load package entry point from {package_file}")

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    _run_package_main()
