from __future__ import annotations

import asyncio
import json
import threading
from typing import Callable

import websockets


class DisplayServer:

    def __init__(self, host: str = "localhost", port: int = 8765,
                 on_message: Callable[[str], None] | None = None):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_message = on_message

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            async for message in websocket:
                if self._on_message:
                    self._on_message(message)
        finally:
            self._clients.discard(websocket)

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()

    def start(self):
        self._loop = asyncio.new_event_loop()
        thread = threading.Thread(target=self._loop.run_until_complete,
                                  args=(self._serve(),), daemon=True)
        thread.start()
        print(f"Display server running at ws://{self.host}:{self.port}")

    def send(self, chunk: str, index: int, score: float, model: str):
        if not self._loop or not self._clients:
            return
        payload = json.dumps({"chunk": chunk, "index": index, "score": round(score, 1),
                              "model": model})
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: str):
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                self._clients.discard(ws)
