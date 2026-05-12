from typing import Dict, Callable, Any

class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._descriptions = {}   # 简要描述
        self._params = {}         # 参数说明

    def register(self, name: str, func, description: str, params: str = ""):
        self._tools[name] = func
        self._descriptions[name] = description
        self._params[name] = params

    def get_descriptions(self) -> str:
        lines = []
        for name, desc in self._descriptions.items():
            param_str = self._params.get(name, "")
            if param_str:
                lines.append(f"- {name}: {desc}。参数：{param_str}")
            else:
                lines.append(f"- {name}: {desc}")
        return "\n".join(lines)