"""Read-only system information tools.

Built on `psutil` + the stdlib `platform` module, both of which are
cross-platform, so these tools work in development on Linux/macOS as well
as on the target Windows deployment — only the OS-*specific* tools
(open/close application, volume, screenshot) need a platform branch.
"""

from __future__ import annotations

import platform
import time
from typing import Any, ClassVar

import psutil

from app.core.enums import RiskLevel
from app.tools.base import Tool, ToolMetadata, ToolResult


class GetSystemInfoTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="get_system_info",
        description="Get the OS name, version, hostname, and uptime.",
        risk_level=RiskLevel.LOW,
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        boot_time = psutil.boot_time()
        uptime_seconds = max(0.0, time.time() - boot_time)
        data = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "hostname": platform.node(),
            "architecture": platform.machine(),
            "uptime_seconds": round(uptime_seconds, 1),
        }
        message = f"Running {data['system']} {data['release']}."
        return ToolResult(success=True, data=data, message=message)


class GetProcessesTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="get_processes",
        description="List running processes with their PID, name, and CPU/memory usage.",
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max processes to return (default 20)"}
            },
        },
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        limit = int(payload.get("limit", 20))
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        processes.sort(key=lambda p: p.get("cpu_percent") or 0.0, reverse=True)
        top = processes[:limit]
        return ToolResult(
            success=True,
            data={"processes": top, "total_count": len(processes)},
            message=f"Found {len(processes)} running processes.",
        )


class GetCpuUsageTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="get_cpu_usage",
        description="Get current overall CPU usage as a percentage.",
        risk_level=RiskLevel.LOW,
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        percent = psutil.cpu_percent(interval=0.1)
        return ToolResult(
            success=True, data={"cpu_percent": percent}, message=f"CPU usage is {percent}%."
        )


class GetMemoryUsageTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="get_memory_usage",
        description="Get current RAM usage: total, used, available, and percentage.",
        risk_level=RiskLevel.LOW,
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        mem = psutil.virtual_memory()
        data = {
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent": mem.percent,
        }
        return ToolResult(
            success=True, data=data, message=f"Memory usage is {data['percent']}%."
        )
