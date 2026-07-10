from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.core.logging import setup_logging
from app.core.exceptions import register_exception_handlers
from app.api.v1.router import router as api_v1_router
import structlog

# Initialize logging configuration
setup_logging()
logger = structlog.get_logger()

app = FastAPI(
    title="Document Scanner API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Scan-Metadata"]
)

# Register custom handlers for application and general exceptions
register_exception_handlers(app)

# Include API v1 router with proper prefix
app.include_router(api_v1_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("startup_complete", cors_origins=settings.cors_origin_list, max_upload_size_mb=settings.max_upload_size_mb)
