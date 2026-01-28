"""GSWA FastAPI Application.

Main entry point for the Gilles-Style Writing Assistant API.
"""
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from gswa.config import get_settings
from gswa.api.routes import router


# Authentication configuration
# Authentication is ENABLED by default with credentials from config.py
# Override with environment variables:
#   GSWA_AUTH_USER=your-username
#   GSWA_AUTH_PASS=your-password
#   GSWA_AUTH_ENABLED=false  (to disable auth)
def get_auth_settings():
    """Get authentication settings from config or environment."""
    settings = get_settings()
    # Environment variables take precedence
    user = os.environ.get("GSWA_AUTH_USER", settings.auth_user)
    passwd = os.environ.get("GSWA_AUTH_PASS", settings.auth_pass)
    # Allow disabling auth via environment variable
    enabled_env = os.environ.get("GSWA_AUTH_ENABLED", "").lower()
    if enabled_env == "false":
        enabled = False
    else:
        enabled = settings.auth_enabled
    return user, passwd, enabled

AUTH_USER, AUTH_PASS, AUTH_ENABLED = get_auth_settings()

security = HTTPBasic(auto_error=False)


def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify HTTP Basic Auth credentials if authentication is enabled."""
    if not AUTH_ENABLED:
        return True  # No auth required

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    # Use secrets.compare_digest to prevent timing attacks
    is_user_ok = secrets.compare_digest(credentials.username.encode(), AUTH_USER.encode())
    is_pass_ok = secrets.compare_digest(credentials.password.encode(), AUTH_PASS.encode())

    if not (is_user_ok and is_pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return True


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting GSWA server...")
    settings = get_settings()
    logger.info(f"LLM server: {settings.vllm_base_url}")
    logger.info(f"External API allowed: {settings.allow_external_api}")

    yield

    # Shutdown
    logger.info("Shutting down GSWA server...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    get_settings()  # This validates security settings

    app = FastAPI(
        title="GSWA - Gilles-Style Writing Assistant",
        description="Local offline scientific paper rewriter",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add authentication middleware if enabled
    if AUTH_ENABLED:
        from starlette.middleware.authentication import AuthenticationMiddleware
        from starlette.authentication import (
            AuthCredentials, AuthenticationBackend, SimpleUser, AuthenticationError
        )
        from starlette.responses import Response
        import base64

        class BasicAuthBackend(AuthenticationBackend):
            async def authenticate(self, conn):
                if "Authorization" not in conn.headers:
                    return None

                auth = conn.headers["Authorization"]
                try:
                    scheme, credentials = auth.split()
                    if scheme.lower() != "basic":
                        return None
                    decoded = base64.b64decode(credentials).decode("utf-8")
                    username, _, password = decoded.partition(":")

                    is_user_ok = secrets.compare_digest(username, AUTH_USER)
                    is_pass_ok = secrets.compare_digest(password, AUTH_PASS)

                    if is_user_ok and is_pass_ok:
                        return AuthCredentials(["authenticated"]), SimpleUser(username)
                except Exception:
                    pass
                return None

        @app.middleware("http")
        async def auth_middleware(request, call_next):
            # Skip auth for health check
            if request.url.path == "/v1/health":
                return await call_next(request)

            auth = request.headers.get("Authorization")
            if not auth:
                return Response(
                    content="Authentication required",
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="GSWA"'},
                )

            try:
                scheme, credentials = auth.split()
                if scheme.lower() != "basic":
                    raise ValueError("Invalid scheme")
                decoded = base64.b64decode(credentials).decode("utf-8")
                username, _, password = decoded.partition(":")

                is_user_ok = secrets.compare_digest(username, AUTH_USER)
                is_pass_ok = secrets.compare_digest(password, AUTH_PASS)

                if not (is_user_ok and is_pass_ok):
                    raise ValueError("Invalid credentials")
            except Exception:
                return Response(
                    content="Invalid credentials",
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="GSWA"'},
                )

            return await call_next(request)

        logger.info(f"Authentication ENABLED - user: {AUTH_USER}")
    else:
        logger.info("Authentication DISABLED - set GSWA_AUTH_ENABLED=true or use defaults")

    # Include API routes
    app.include_router(router)

    # Serve static files (web UI)
    web_dir = Path(__file__).parent.parent.parent / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
        logger.info(f"Serving web UI from {web_dir}")
    else:
        logger.warning(f"Web UI directory not found at {web_dir}, skipping static file serving")

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
