import asyncio
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config import AppConfig, get_config
from sentinel.defenses.security_manager import SecurityManager


@dataclass
class MCPServerConfig:
    name: str
    command: Sequence[str]
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class MCPToolResult:
    server: str
    tool: str
    content: str
    success: bool = True
    error: Optional[str] = None


class StdioMCPClient:
    """Minimal MCP stdio JSON-RPC client for list_tools and call_tool."""

    def __init__(self, server: MCPServerConfig, timeout: float = 20.0):
        self.server = server
        self.timeout = timeout
        self.process = None
        self._request_id = 0

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def start(self) -> None:
        if not self.server.command:
            raise ValueError(f"MCP server {self.server.name} 缺少启动命令。")
        executable = self.server.command[0]
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"找不到命令：{executable}")

        env = os.environ.copy()
        env.update(self.server.env)
        self.process = await asyncio.create_subprocess_exec(
            *self.server.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        await self.initialize()

    async def initialize(self) -> Dict[str, Any]:
        response = await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "chapter8-private-qa", "version": "0.1.0"},
        })
        await self._notify("notifications/initialized", {})
        return response

    async def list_tools(self) -> List[Dict[str, Any]]:
        response = await self._request("tools/list", {})
        return response.get("tools", [])

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> MCPToolResult:
        try:
            response = await self._request("tools/call", {
                "name": name,
                "arguments": arguments or {},
            })
            content = self._format_tool_content(response)
            is_error = response.get("isError", False)
            return MCPToolResult(
                server=self.server.name,
                tool=name,
                content=content,
                success=not is_error,
                error=content if is_error else None,
            )
        except Exception as exc:
            return MCPToolResult(
                server=self.server.name,
                tool=name,
                content="",
                success=False,
                error=str(exc),
            )

    async def close(self) -> None:
        if self.process is None:
            return
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=3)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None

    async def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        await self._write({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        })

        while True:
            message = await self._read()
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(message["error"])
            return message.get("result", {})

    async def _notify(self, method: str, params: Dict[str, Any]) -> None:
        await self._write({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })

    async def _write(self, message: Dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("MCP process is not running.")
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n"
        self.process.stdin.write(payload)
        await self.process.stdin.drain()

    async def _read(self) -> Dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("MCP process is not running.")
        buffer = ""
        while True:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=self.timeout)
            if not line:
                stderr = ""
                if self.process.stderr is not None:
                    try:
                        stderr = (await asyncio.wait_for(self.process.stderr.read(), timeout=0.2)).decode("utf-8", errors="ignore")
                    except Exception:
                        stderr = ""
                raise RuntimeError(f"MCP server closed stdout. {stderr}".strip())

            text = line.decode("utf-8")
            stripped = text.strip()
            if not stripped:
                continue
            if not buffer and not stripped.startswith(("{", "[")):
                continue

            buffer += text
            try:
                return json.loads(buffer)
            except json.JSONDecodeError:
                if len(buffer) > 10_000_000:
                    raise

    def _format_tool_content(self, response: Dict[str, Any]) -> str:
        parts = []
        for item in response.get("content", []):
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)


class MCPManager:
    """Select and call a small set of MCP servers, then expose results as QA context."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        servers: Optional[Dict[str, MCPServerConfig]] = None,
        client_cls: Any = StdioMCPClient,
        security_manager: Any = None,
    ):
        self.config = config or get_config()
        self.servers = servers or build_default_mcp_servers(self.config)
        self.client_cls = client_cls
        self.security_manager = security_manager or SecurityManager.from_app_config(self.config)
        self._openai_tool_map: Dict[str, Dict[str, str]] = {}

    def enabled_servers(self) -> Dict[str, MCPServerConfig]:
        return {name: server for name, server in self.servers.items() if server.enabled}

    def online_search_available(self) -> bool:
        return "brave-search" in self.enabled_servers()

    def retrieve_context(self, question: str) -> List[Dict[str, Any]]:
        if not self.config.mcp_enabled:
            return []
        return self._run_async(self.retrieve_context_async(question))

    def retrieve_online_context(self, question: str) -> List[Dict[str, Any]]:
        if not self.config.mcp_enabled:
            return []
        return self._run_async(self.retrieve_online_context_async(question))

    async def retrieve_context_async(self, question: str) -> List[Dict[str, Any]]:
        tasks = self.plan_tool_calls(question)
        return await self._execute_planned_tool_calls(tasks)

    async def retrieve_online_context_async(self, question: str) -> List[Dict[str, Any]]:
        tasks = self.plan_online_search_tool_calls(question)
        return await self._execute_planned_tool_calls(tasks)

    async def _execute_planned_tool_calls(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not tasks:
            return []

        memories = []
        for task in tasks:
            result = await self.call_tool(task["server"], task["tool"], task["arguments"])
            content = result.content if result.success else f"MCP 调用失败：{result.error}"
            observation = {
                "role": f"mcp:{result.server}.{result.tool}",
                "type": "mcp",
                "content": self._truncate(content),
                "metadata": {
                    "server": result.server,
                    "tool": result.tool,
                    "success": result.success,
                    "error": result.error,
                    "arguments": task["arguments"],
                },
            }
            if self.security_manager is not None:
                try:
                    observation, decision = self.security_manager.sanitize_tool_observation(observation)
                    if decision.findings:
                        observation.setdefault("metadata", {})["tool_output_sanitize_decision"] = decision.to_dict()
                except Exception:
                    pass
            memories.append(observation)
        return memories

    def plan_online_search_tool_calls(self, question: str) -> List[Dict[str, Any]]:
        if "brave-search" not in self.enabled_servers():
            return []
        return [{
            "server": "brave-search",
            "tool": "brave_web_search",
            "arguments": {"query": question, "count": 5},
        }]

    def plan_tool_calls(self, question: str) -> List[Dict[str, Any]]:
        tasks = []
        url_match = re.search(r"https?://[^\s，。)）]+", question)
        if url_match and "fetch" in self.enabled_servers():
            tasks.append({"server": "fetch", "tool": "fetch", "arguments": {"url": url_match.group(0)}})

        if "filesystem" in self.enabled_servers() and self._looks_like_file_listing(question):
            tasks.append({
                "server": "filesystem",
                "tool": "list_directory",
                "arguments": {"path": str(Path(self.config.mcp_filesystem_root).resolve())},
            })

        if "sqlite" in self.enabled_servers() and self._looks_like_sqlite_schema(question):
            tasks.append({"server": "sqlite", "tool": "list_tables", "arguments": {}})

        return tasks[:3]

    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        server = self.servers[server_name]
        async with self.client_cls(server, timeout=self.config.mcp_timeout_seconds) as client:
            tools = await client.list_tools()
        return self._filter_tool_descriptors(server_name, tools)

    async def call_tool(self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> MCPToolResult:
        server = self.servers[server_name]
        async with self.client_cls(server, timeout=self.config.mcp_timeout_seconds) as client:
            tools = await client.list_tools()
            available = {tool.get("name") for tool in tools}
            if available and tool_name not in available:
                fallback = self._find_tool_name(tools, tool_name)
                if fallback is None:
                    return MCPToolResult(
                        server=server_name,
                        tool=tool_name,
                        content="",
                        success=False,
                        error=f"工具不存在，可用工具：{', '.join(sorted(available))}",
                    )
                tool_name = fallback
            return await client.call_tool(tool_name, arguments)

    async def probe_server(self, server_name: str) -> Dict[str, Any]:
        tools = await self.list_tools(server_name)
        return {
            "server": server_name,
            "tool_count": len(tools),
            "tools": [tool.get("name", "") for tool in tools],
        }

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        if not self.config.mcp_enabled:
            return []
        return self._run_async(self.get_openai_tools_async())

    async def get_openai_tools_async(self) -> List[Dict[str, Any]]:
        openai_tools = []
        self._openai_tool_map = {}
        for server_name in self.enabled_servers():
            try:
                tools = await self.list_tools(server_name)
            except Exception:
                continue
            for tool in tools:
                tool_name = tool.get("name")
                if not tool_name:
                    continue
                function_name = self._openai_function_name(server_name, tool_name)
                self._openai_tool_map[function_name] = {
                    "server": server_name,
                    "tool": tool_name,
                }
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": self._tool_description(server_name, tool),
                        "parameters": self._tool_parameters(tool),
                    },
                })
        return openai_tools

    def _filter_tool_descriptors(self, server_name: str, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.security_manager is None:
            return tools
        safe_tools = []
        for tool in tools:
            try:
                decision = self.security_manager.check_tool_descriptor({
                    **tool,
                    "server": server_name,
                })
            except Exception:
                safe_tools.append(tool)
                continue
            if decision.action in {"block", "quarantine"}:
                continue
            safe_tool = dict(tool)
            if decision.findings:
                safe_tool.setdefault("_sentinel", {})["descriptor_decision"] = decision.to_dict()
            safe_tools.append(safe_tool)
        return safe_tools

    def call_openai_tool(self, function_name: str, arguments: Any) -> MCPToolResult:
        return self._run_async(self.call_openai_tool_async(function_name, arguments))

    async def call_openai_tool_async(self, function_name: str, arguments: Any) -> MCPToolResult:
        if function_name not in self._openai_tool_map:
            await self.get_openai_tools_async()
        mapping = self._openai_tool_map.get(function_name)
        if not mapping:
            return MCPToolResult(
                server="unknown",
                tool=function_name,
                content="",
                success=False,
                error=f"未知 MCP tool function：{function_name}",
            )

        parsed_arguments = self._parse_tool_arguments(arguments)
        return await self.call_tool(mapping["server"], mapping["tool"], parsed_arguments)

    def _find_tool_name(self, tools: List[Dict[str, Any]], preferred: str) -> Optional[str]:
        names = [tool.get("name", "") for tool in tools]
        if preferred == "list_directory":
            for candidate in ["list_directory", "directory_tree", "list_allowed_directories"]:
                if candidate in names:
                    return candidate
        if preferred == "list_tables":
            for candidate in ["list_tables", "describe_table"]:
                if candidate in names:
                    return candidate
        if preferred == "fetch" and "fetch" in names:
            return "fetch"
        if preferred in {"brave_web_search", "web_search", "search"}:
            for candidate in ["brave_web_search", "web_search", "search"]:
                if candidate in names:
                    return candidate
        return None

    def _looks_like_file_listing(self, question: str) -> bool:
        return any(marker in question for marker in ["列出文件", "文件列表", "目录", "本地文件", "有哪些文件"])

    def _looks_like_sqlite_schema(self, question: str) -> bool:
        lowered = question.lower()
        return "sqlite" in lowered or "数据库表" in question or "有哪些表" in question

    def _truncate(self, content: str) -> str:
        max_chars = self.config.mcp_max_result_chars
        if len(content) <= max_chars:
            return content
        return content[: max_chars - 3] + "..."

    def _run_async(self, coro):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        if running_loop.is_running():
            import threading

            result_box: Dict[str, Any] = {}

            def runner() -> None:
                try:
                    result_box["result"] = asyncio.run(coro)
                except Exception as exc:
                    result_box["error"] = exc

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()
            thread.join()
            if "error" in result_box:
                raise result_box["error"]
            return result_box.get("result")

        return running_loop.run_until_complete(coro)

    def _openai_function_name(self, server_name: str, tool_name: str) -> str:
        raw_name = f"mcp__{server_name}__{tool_name}"
        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", raw_name)
        return sanitized[:64]

    def _tool_description(self, server_name: str, tool: Dict[str, Any]) -> str:
        description = tool.get("description") or f"MCP tool {tool.get('name', '')}"
        return f"[MCP:{server_name}] {description}"

    def _tool_parameters(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        if not isinstance(schema, dict):
            schema = {}
        if schema.get("type") != "object":
            schema = {
                "type": "object",
                "properties": schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {},
            }
        schema.setdefault("properties", {})
        schema.setdefault("additionalProperties", True)
        return schema

    def _parse_tool_arguments(self, arguments: Any) -> Dict[str, Any]:
        if arguments is None or arguments == "":
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {"input": arguments}
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        return {"input": arguments}


def build_default_mcp_servers(config: AppConfig) -> Dict[str, MCPServerConfig]:
    root = str(Path(config.mcp_filesystem_root).resolve())
    servers = {
        "filesystem": MCPServerConfig(
            name="filesystem",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem", root],
            enabled=config.mcp_filesystem_enabled,
        ),
        "fetch": MCPServerConfig(
            name="fetch",
            command=_fetch_command(config),
            enabled=config.mcp_fetch_enabled,
        ),
        "sqlite": MCPServerConfig(
            name="sqlite",
            command=[_console_script_path("mcp-server-sqlite"), "--db-path", str(Path(config.mcp_sqlite_db_path).resolve())],
            enabled=config.mcp_sqlite_enabled,
        ),
        "brave-search": MCPServerConfig(
            name="brave-search",
            command=["npx", "-y", "@modelcontextprotocol/server-brave-search"],
            env={"BRAVE_API_KEY": config.mcp_brave_api_key or ""},
            enabled=config.mcp_brave_enabled and bool(config.mcp_brave_api_key),
        ),
    }
    return servers


def _console_script_path(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).with_name(name)
    if candidate.exists():
        return str(candidate)
    return name


def _fetch_command(config: AppConfig) -> List[str]:
    command = [_console_script_path("mcp-server-fetch")]
    if config.mcp_fetch_ignore_robots_txt:
        command.append("--ignore-robots-txt")
    return command
