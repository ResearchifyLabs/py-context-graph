"""Default prompt loader for the decision graph library."""

import os

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_CACHE: dict = {}


def load_prompt(prompt_name: str, **kwargs) -> str:
    file_path = os.path.join(_PROMPTS_DIR, f"{prompt_name}.txt")

    if file_path in _CACHE:
        template = _CACHE[file_path]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read()
        _CACHE[file_path] = template

    return template.format(**kwargs) if kwargs else template
