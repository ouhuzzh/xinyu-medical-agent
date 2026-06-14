import sys
import time
import uuid
import logging
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import config
from api.dependencies import get_container
from api.routes import auth_routes, chat, documents, mcp_servers, memory, system
from core.runtime_security import validate_runtime_security_or_raise


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    validate_runtime_security_or_raise()

    app = FastAPI(title="Medical Agentic Assistant API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.API_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started_at = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            user_id = getattr(request.state, "user_id", "")
            thread_id = getattr(request.state, "thread_id", "")
            route_type = getattr(request.state, "route_type", request.url.path)
            status_code = getattr(response, "status_code", 500)
            logger.info(
                "api_request request_id=%s user_id=%s thread_id=%s route_type=%s method=%s path=%s status=%s duration_ms=%.2f",
                request_id,
                user_id or "-",
                thread_id or "-",
                route_type or "-",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
            )
            if response is not None:
                response.headers["X-Request-ID"] = request_id

    app.include_router(system.router)
    app.include_router(auth_routes.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(mcp_servers.router)
    app.include_router(memory.router)

    @app.on_event("shutdown")
    async def _shutdown_db_pool():
        from db.connection import close_connection_pool
        close_connection_pool()

    if config.APP_ENV != "development":
        get_container()
    return app


app = create_app()
