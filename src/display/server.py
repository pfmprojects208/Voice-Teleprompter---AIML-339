import asyncio
import json
import threading
import websockets


class DisplayServer:

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            await websocket.wait_closed()
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

    def send(self, chunk: str, index: int, score: float):
        if not self._loop or not self._clients:
            return
        payload = json.dumps({"chunk": chunk, "index": index, "score": round(score, 1)})
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: str):
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                self._clients.discard(ws)
