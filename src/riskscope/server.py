"""HTTP entry point for RiskScope."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .service import Service, ServiceError


def env_address() -> tuple[str, int]:
    raw = os.environ.get("RISKSCOPE_ADDR", "127.0.0.1:8080")
    host, _, port = raw.rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"invalid RISKSCOPE_ADDR: {raw!r}")
    return host, int(port)


class Handler(BaseHTTPRequestHandler):
    service = Service()

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self.send_json(200, self.service.health())
            return
        self.send_json(404, {"error": {"code": "not_found", "message": f"no route for {self.path}"}})

    def do_POST(self) -> None:
        if self.path not in (
            "/market-risk/historical-var",
            "/market-risk/stress-test",
            "/market-risk/var-backtest",
            "/market-risk/covariance-estimate",
            "/market-risk/parametric-var",
            "/credit-risk/counterparty-exposure",
            "/liquidity-risk/liquidity-gap",
            "/portfolio-risk/aggregate",
        ):
            self.send_json(404, {"error": {"code": "not_found", "message": f"no route for {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            self.send_json(
                400,
                {"error": {"code": "invalid_request", "message": "invalid Content-Length"}},
            )
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            if self.path == "/market-risk/historical-var":
                result = self.service.historical_var(raw)
            elif self.path == "/market-risk/var-backtest":
                result = self.service.var_backtest(raw)
            elif self.path == "/market-risk/covariance-estimate":
                result = self.service.covariance_estimate(raw)
            elif self.path == "/market-risk/parametric-var":
                result = self.service.parametric_var(raw)
            elif self.path == "/credit-risk/counterparty-exposure":
                result = self.service.counterparty_exposure(raw)
            elif self.path == "/liquidity-risk/liquidity-gap":
                result = self.service.liquidity_gap(raw)
            elif self.path == "/portfolio-risk/aggregate":
                result = self.service.portfolio_aggregate(raw)
            else:
                result = self.service.stress_test(raw)
        except ServiceError as exc:
            self.send_json(
                exc.status,
                {"error": {"code": exc.code, "message": exc.message}},
            )
            return
        self.send_json(200, result)

    def log_message(self, fmt: str, *args: object) -> None:
        """Silence per-request logging so recorded output stays stable."""


def main() -> int:
    parser = argparse.ArgumentParser(prog="riskscope.server", description="市场风险、信用风险与流动性风险管理引擎")
    host, port = env_address()
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RiskScope listening on http://{args.host}:{httpd.server_address[1]}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
