"""Pydantic configuration model for the sandbox package."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


class SandboxDebugConfig(BaseModel):
    """Debug socket server configuration (nested under sandbox.debug)."""

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel, populate_by_name=True)

    enabled: bool = False
    socket: str = "sandbox/debug.sock"
    max_message_bytes: int = 1048576  # 1 MiB 上限

    @field_validator("max_message_bytes")
    @classmethod
    def _validate_max_message_bytes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("maxMessageBytes must be > 0")
        return v


class SandboxConfig(BaseModel):
    """Sandbox execution configuration (boxlite microVM)."""

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel, populate_by_name=True)

    # "none"    → DirectExecutor：直接在宿主机运行，不提供隔离（默认）
    # "auto"    → 自动检测：当前唯一支持的后端是 boxlite；
    #             检测失败时在启动阶段报错
    # "boxlite" → 强制使用 boxlite；同时执行可用性探测，不可用时报错
    backend: Literal["none", "auto", "boxlite"] = "none"
    image: str = "ubuntu:22.04"
    cpus: int = 2
    memory_mib: int = 2048
    disk_size_gb: int | None = None  # None 表示使用 boxlite 默认的临时磁盘
    # 网络：True 表示完全开放；False 表示禁网；列表表示域名白名单
    allow_net: bool | list[str] = True
    # 额外卷挂载：每项格式为 [host_path, vm_path, "ro"|"rw"]
    extra_volumes: list[list[str]] = Field(default_factory=list)
    # 沙箱内单次 exec 调用的默认超时时间（秒）
    default_timeout: int = 120
    # 启动时 echo-ok 探测的超时时间（秒）
    verify_timeout: int = 30
    # 拉取镜像并创建虚拟机的超时时间（秒）
    create_timeout: int = 300
    # 调试套接字服务器（嵌套对象）
    debug: SandboxDebugConfig = Field(default_factory=SandboxDebugConfig)

    @field_validator("allow_net")
    @classmethod
    def _validate_allow_net(cls, v: bool | list[str]) -> bool | list[str]:
        if isinstance(v, list) and not v:
            raise ValueError(
                "allow_net: [] is ambiguous — an empty allowlist may mean 'allow all' or "
                "'allow none' depending on the boxlite runtime. "
                "Use allow_net: false to disable networking entirely."
            )
        return v

    @field_validator("extra_volumes")
    @classmethod
    def _validate_volumes(cls, v: list[list[str]]) -> list[list[str]]:
        for entry in v:
            if len(entry) != 3 or entry[2] not in ("ro", "rw"):
                raise ValueError(f"Invalid volume entry {entry!r}; each entry must be [host_path, vm_path, 'ro'|'rw']")
            if not Path(entry[0]).is_absolute():
                raise ValueError(f"Volume host path must be absolute: {entry[0]!r}")
            if not Path(entry[1]).is_absolute():
                raise ValueError(f"Volume VM path must be absolute: {entry[1]!r}")
        return v
