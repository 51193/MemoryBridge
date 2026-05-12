"""Token authentication middleware."""

from typing import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

CallNext = Callable[[Request], Awaitable[Response]]


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Require valid Bearer token on all routes except /health."""

    async def dispatch(
        self, request: Request, call_next: CallNext
    ) -> Response:
        if request.url.path == "/health":
            return await call_next(request)

        if not request.app.state.token_enabled:
            return await call_next(request)

        auth: str = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "TOKEN_MISSING"},
            )

        token: str = auth[7:]
        if not request.app.state.token_store.validate(token):
            return JSONResponse(
                status_code=401,
                content={"detail": "TOKEN_INVALID"},
            )

        return await call_next(request)
