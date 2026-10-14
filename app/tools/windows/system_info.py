"""Read-only system information tools.

Built on `psutil` + the stdlib `platform` module, both of which are
cross-platform, so these tools work in development on Linux/macOS as well
as on the target Windows deployment — only the OS-*specific* tools
(open/close application, volume, screenshot) need a platform branch.
"""

from __future__ import annotations

import asyncio
import platform
import time
from typing import Any, ClassVar

import psutil

from app.core.enums import RiskLevel
from app.core.errors import ValidationError
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

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = payload.get("limit", 20)
        try:
            limit = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("get_processes 'limit' must be an integer") from exc
        return {"limit": max(1, min(limit, 100))}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        limit = payload.get("limit", 20)
        # psutil reports 0.0 for a process the first time cpu_percent() is called,
        # so sampling once made the "top by CPU" list arbitrary. Prime every
        # process, wait a moment, then read the real usage over that window.
        procs: list[psutil.Process] = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                proc.cpu_percent(None)
                procs.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        await asyncio.sleep(0.3)

        processes: list[dict[str, Any]] = []
        for proc in procs:
            try:
                processes.append(
                    {
                        "pid": proc.pid,
                        "name": proc.info.get("name"),
                        "cpu_percent": proc.cpu_percent(None),
                        "memory_percent": round(proc.memory_percent(), 2),
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        processes.sort(key=lambda p: p["cpu_percent"], reverse=True)
        top = processes[:limit]
        summary = ", ".join(f"{p['name']} ({p['cpu_percent']:.0f}%)" for p in top[:5])
        return ToolResult(
            success=True,
            data={"processes": top, "total_count": len(processes)},
            message=f"Found {len(processes)} running processes. Top by CPU: {summary}.",
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
        return ToolResult(success=True, data=data, message=f"Memory usage is {data['percent']}%.")
