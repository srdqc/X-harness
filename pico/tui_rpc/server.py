"""asyncio JSON-RPC 2.0 server loop over a full-duplex socket (or, for tests, a
POSIX pipe pair).

Topology: the parent listens on a TCP-loopback socket; the Node child connects,
sends an auth token line, then exchanges newline-JSON frames over the same
connection. ``RpcServer`` is given the accepted socket *object* and wires it via
``loop.connect_accepted_socket`` (cross-platform: selector + proactor loops).
A legacy pipe path (``request_fd``/``notify_fd`` + ``connect_read/write_pipe``)
is retained for the ``--check`` smoke and unit tests.

`RpcServer` owns the read pump (one line-delimited JSON frame per iteration),
dispatches concurrently via `asyncio.create_task` so a long-running streaming
subscription doesn't block other RPC calls, and serializes writes with an
`asyncio.Lock` so concurrent dispatch tasks can't interleave bytes on the wire.

Frame size limit: 1 MiB (specs §2.5). Larger frames trigger immediate
shutdown of the connection.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pico.tui_rpc.dispatcher import Dispatcher


# 按 specs/tui-ipc.md §2.5 的规定
MAX_FRAME_BYTES = 1 * 1024 * 1024  # 1 MiB 上限


class RpcServer:
    """Read JSON-RPC frames from `request_fd`, write responses to `notify_fd`.

    Args:
        request_fd: POSIX fd opened for reading (the Node→Python pipe).
        notify_fd:  POSIX fd opened for writing (the Python→Node pipe).
        dispatcher: a `Dispatcher` instance with all handlers registered.

    The server takes ownership of the FDs: they are closed on `stop()`.
    """

    def __init__(
        self,
        request_fd: int = -1,
        notify_fd: int = -1,
        dispatcher: "Dispatcher | None" = None,
        *,
        sock: "socket.socket | None" = None,
        auth_token: str | None = None,
    ) -> None:
        self._request_fd = request_fd
        self._notify_fd = notify_fd
        self._dispatcher = dispatcher
        # 跨平台生产传输：以对象形式传入单个已连接的 TCP 回环套接字，
        # 不对套接字文件描述符执行 Windows 不支持的 os.dup。设置后优先于 fd 对。
        self._sock = sock
        # 对端必须首先发送的可选共享密钥行。任何本地进程都能访问 TCP 回环，
        # 不像由 0600 权限保护的 AF_UNIX 文件，因此令牌用于恢复“只有我们启动的子进程
        # 才能通信”这一信任边界。None 禁用检查，供管道和测试路径使用。
        self._auth_token = auth_token

        self._reader: asyncio.StreamReader | None = None
        self._write_transport: asyncio.WriteTransport | None = None
        self._write_protocol: asyncio.BaseProtocol | None = None
        self._write_lock = asyncio.Lock()
        self._pending: set[asyncio.Task] = set()
        self._stopped = asyncio.Event()
        self._started = asyncio.Event()

    @property
    def started(self) -> asyncio.Event:
        """Set once the read pump has attached to the FD; useful for tests."""
        return self._started

    # ----- 写入端 -------------------------------------------------------

    async def send_frame(self, frame: dict) -> None:
        """Serialize and write a single JSON frame + newline to `notify_fd`.

        All writes (responses + notifications) MUST go through this method so
        the lock serializes them.
        """
        if self._write_transport is None:
            raise RuntimeError("RpcServer.send_frame called before serve_forever()")
        data = (json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            self._write_transport.write(data)

    # ----- 主循环 --------------------------------------------------------

    async def serve_forever(self) -> None:
        """Run the read/dispatch/write pump until EOF or `stop()`."""
        loop = asyncio.get_running_loop()

        reader = asyncio.StreamReader(limit=MAX_FRAME_BYTES)
        reader_protocol = asyncio.StreamReaderProtocol(reader)

        # P0 修复（2026-05-15）：``tui_commands.run_subprocess_with_rpc`` 的生产传输
        # 将同一已接受的 Unix 套接字 fd 复制为 ``request_fd`` 和 ``notify_fd``。
        # CPython 的 ``connect_write_pipe`` 会创建 ``_UnixWritePipeTransport``，并在写 fd 上
        # 注册读回调，以检测“对端关闭管道”（读端 EOF）。这对真管道有效，
        # 但对双向 SOCK_STREAM 套接字是致命的：任何入站字节（如 ``system.hello``）都会
        # 触发读回调，继而执行 ``_close()``，后续 ``send_frame`` 全部静默失效。
        # 连续丢弃 5 次后，asyncio 还会记录 ``"pipe closed by peer or os.write(pipe, data)
        # raised exception."``，这正是 ``/cha`` 斜杠自动补全时观察到的症状。
        #
        # 改为在同一 fd 上使用 ``connect_accepted_socket``（全双工选择器传输）。
        # 该传输的 ``.write()`` 不会误判对端关闭。仍保留基于管道的路径，
        # 作为测试和 CI 中直接连接 ``os.pipe()`` 对的回退。
        req_is_sock = notif_is_sock = False
        if self._sock is None:
            try:
                req_is_sock = stat.S_ISSOCK(os.fstat(self._request_fd).st_mode)
                notif_is_sock = stat.S_ISSOCK(os.fstat(self._notify_fd).st_mode)
            except OSError:
                req_is_sock = notif_is_sock = False

        if self._sock is not None:
            # 跨平台生产路径：通过单个已连接的套接字对象（TCP 回环）进行全双工传输。
            # 选择器（POSIX）和 Proactor（Windows）事件循环都实现了 connect_accepted_socket，
            # 传入套接字对象也避免对 Windows 不支持的套接字 fd 执行 os.dup。
            self._sock.setblocking(False)
            transport, _ = await loop.connect_accepted_socket(lambda: reader_protocol, self._sock)
            self._write_transport = transport
            self._write_protocol = reader_protocol
        elif req_is_sock and notif_is_sock:
            # 两个 fd 都是同一已接受套接字的副本。关闭读副本，将写副本回收为
            # ``socket.socket``；全双工传输只需一个句柄。
            try:
                os.close(self._request_fd)
            except OSError:
                pass
            sock = socket.socket(fileno=self._notify_fd)
            sock.setblocking(False)
            transport, _ = await loop.connect_accepted_socket(lambda: reader_protocol, sock)
            self._write_transport = transport
            self._write_protocol = reader_protocol
        else:
            # 旧路径和测试路径：使用 ``os.pipe()`` 裸管道。通过 `os.fdopen` 让传输层
            # 拥有 Python 文件对象；传输关闭时，事件循环会关闭底层 fd。
            await loop.connect_read_pipe(lambda: reader_protocol, os.fdopen(self._request_fd, "rb", buffering=0))
            write_transport, write_protocol = await loop.connect_write_pipe(
                asyncio.BaseProtocol,
                os.fdopen(self._notify_fd, "wb", buffering=0),
            )
            self._write_transport = write_transport
            self._write_protocol = write_protocol

        self._reader = reader

        self._started.set()
        logger.info(
            "tui_rpc: RpcServer started (pid={}, request_fd={}, notify_fd={}, mode={})",
            os.getpid(),
            self._request_fd,
            self._notify_fd,
            "socket-obj" if self._sock is not None else ("socket" if req_is_sock else "pipe"),
        )

        # TCP 回环传输的信任边界门禁：对端必须在任何 JSON-RPC 帧之前，
        # 先发送以换行符结尾的共享密钥行。只有我们启动的 Node 子进程知道它
        # （通过环境变量传递），因此会在分派前拒绝连入端口的恶意本地进程。
        # 管道和测试路径通过 None 禁用此检查。
        if self._auth_token is not None:
            try:
                first = await asyncio.wait_for(reader.readuntil(b"\n"), timeout=10.0)
            except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                logger.error("tui_rpc: auth token not received; closing connection")
                self._stopped.set()
                return
            if first.rstrip(b"\n") != self._auth_token.encode("utf-8"):
                logger.error("tui_rpc: auth token mismatch; closing connection")
                self._stopped.set()
                return

        try:
            while not self._stopped.is_set():
                try:
                    line = await reader.readuntil(b"\n")
                except asyncio.IncompleteReadError as exc:
                    # EOF 表示对端已关闭，需处理现有的不完整字节。
                    if exc.partial:
                        logger.warning(
                            "tui_rpc: incomplete final frame ({} bytes); dropping",
                            len(exc.partial),
                        )
                    break
                except asyncio.LimitOverrunError:
                    logger.error(
                        "tui_rpc: frame exceeds {} bytes; closing connection",
                        MAX_FRAME_BYTES,
                    )
                    break

                if len(line) > MAX_FRAME_BYTES:
                    logger.error("tui_rpc: frame {} bytes > {} cap; closing", len(line), MAX_FRAME_BYTES)
                    break

                # 将分派启动为独立任务，避免流式或慢处理器阻塞后续读取。
                task = asyncio.create_task(self._handle_frame(line))
                self._pending.add(task)
                task.add_done_callback(self._pending.discard)
        finally:
            await self._shutdown()

    async def _handle_frame(self, raw: bytes) -> None:
        try:
            try:
                frame = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # JSON-RPC §-32700 parse_error 响应（ID 未知时为 null）。
                resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "parse_error",
                        "data": {"reason": str(exc)},
                    },
                }
                await self.send_frame(resp)
                return

            response = await self._dispatcher.dispatch(frame)

            # 如果原始帧省略 `id`（JSON-RPC 2.0 通知），则禁止响应。分派器会原样回显
            # 收到的 ID，因此只在入站帧明确不含 ID 时禁止。
            if isinstance(frame, dict) and "id" not in frame:
                return
            await self.send_frame(response)
        except Exception:
            # 最后一道保护，避免单个有缺陷的处理器终止整个消息泵。
            logger.exception("tui_rpc: _handle_frame failed")

    async def _shutdown(self) -> None:
        # 取消所有正在执行的分派任务。
        for task in list(self._pending):
            if not task.done():
                task.cancel()
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
        self._pending.clear()

        if self._write_transport is not None:
            try:
                self._write_transport.close()
            except Exception:
                logger.exception("tui_rpc: error closing write transport")
            self._write_transport = None

        self._stopped.set()
        logger.info("tui_rpc: RpcServer stopped (pid={})", os.getpid())

    async def stop(self) -> None:
        """Signal the read loop to exit and wait for cleanup."""
        self._stopped.set()
        # 无法轻易中断 `readuntil`，但关闭写端并设置 `_stopped` 后，EOF 之后的下一次迭代
        # 会退出。调用方通常只需取消 serve_forever 任务。


__all__ = ["RpcServer", "MAX_FRAME_BYTES"]
