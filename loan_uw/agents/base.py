"""BaseAgent: one Semantic Kernel `Kernel` per agent, holding that agent's own plugin of tools.

Why one kernel per agent? Tool isolation - the EMI Agent literally cannot call a credit
tool, because that plugin is not registered in its kernel. Every tool call goes through
`kernel.invoke(...)` and is logged with its arguments and result.
"""
from __future__ import annotations

from semantic_kernel import Kernel

from loan_uw.memory import SharedMemory
from loan_uw.models import AgentReport


class BaseAgent:
    name: str = "BaseAgent"
    plugin_name: str = "base"
    role: str = ""

    def __init__(self) -> None:
        self.kernel = Kernel()
        self.kernel.add_plugin(self.build_plugin(), plugin_name=self.plugin_name)

    def build_plugin(self):  # pragma: no cover - overridden
        raise NotImplementedError

    async def call_tool(self, memory: SharedMemory, tool: str, **kwargs):
        try:
            result = await self.kernel.invoke(
                plugin_name=self.plugin_name, function_name=tool, **kwargs)
        except Exception as exc:
            memory.log(self.name, "TOOL_ERROR", tool=f"{self.plugin_name}.{tool}", error=repr(exc))
            raise
        value = result.value
        memory.log(self.name, "TOOL_CALL", tool=f"{self.plugin_name}.{tool}",
                   args=kwargs, result=value)
        return value

    async def run(self, memory: SharedMemory) -> AgentReport:  # pragma: no cover
        raise NotImplementedError
