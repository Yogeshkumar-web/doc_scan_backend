from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse

class AppError(Exception):
    """Base exception for app errors that can be formatted as JSON responses."""
    def __init__(self, error_code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code

class InvalidFileType(AppError):
    def __init__(self, message: str = "Only JPEG and PNG images are supported."):
        super().__init__("INVALID_FILE_TYPE", message, status_code=400)

class FileTooLarge(AppError):
    def __init__(self, message: str = "File size exceeds limit."):
        super().__init__("FILE_TOO_LARGE", message, status_code=413)

class ProcessingFailed(AppError):
    def __init__(self, message: str = "Image processing failed."):
        super().__init__("PROCESSING_FAILED", message, status_code=500)

def register_exception_handlers(app: FastAPI):
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error_code": exc.error_code,
                "message": exc.message
            }
        )
    
    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        # Prevent leaking raw traceback except logging it
        import structlog
        logger = structlog.get_logger()
        logger.error("unhandled_exception", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred."
            }
        )
