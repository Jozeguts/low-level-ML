from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from scheduler import ContinuousBatchScheduler, GenerationRequest


@dataclass(frozen=True)
class GeneratePayload:
    request_id: str
    input_ids: list[int]
    max_new_tokens: int = 32
    priority: int = 0


def make_handler(submit: Callable[[GeneratePayload], None], cancel: Callable[[str], bool]):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path == "/v1/generate":
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    data = json.loads(self.rfile.read(size))
                    payload = GeneratePayload(
                        request_id=str(data["request_id"]),
                        input_ids=[int(x) for x in data["input_ids"]],
                        max_new_tokens=int(data.get("max_new_tokens", 32)),
                        priority=int(data.get("priority", 0)),
                    )
                    submit(payload)
                    self._json(202, {"request_id": payload.request_id, "status": "queued"})
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._json(400, {"error": str(exc)})
                return
            self._json(404, {"error": "not found"})

        def do_DELETE(self) -> None:
            prefix = "/v1/requests/"
            if self.path.startswith(prefix):
                request_id = self.path[len(prefix):]
                self._json(200 if cancel(request_id) else 404, {"request_id": request_id})
                return
            self._json(404, {"error": "not found"})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


class SchedulerService:
    """Adapter keeping HTTP concerns separate from the scheduling core."""

    def __init__(self, scheduler: ContinuousBatchScheduler):
        self.scheduler = scheduler

    def submit(self, payload: GeneratePayload) -> None:
        self.scheduler.submit(
            GenerationRequest(
                request_id=payload.request_id,
                prompt_tokens=payload.input_ids,
                max_new_tokens=payload.max_new_tokens,
                priority=payload.priority,
            )
        )

    def cancel(self, request_id: str) -> bool:
        return self.scheduler.cancel(request_id)

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        handler = make_handler(self.submit, self.cancel)
        server = ThreadingHTTPServer((host, port), handler)
        print(f"listening on http://{host}:{port}")
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == "__main__":
    SchedulerService(ContinuousBatchScheduler()).serve()
