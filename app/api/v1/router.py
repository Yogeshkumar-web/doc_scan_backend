from fastapi import APIRouter
from app.api.v1.scan import router as scan_router
from app.api.v1.pdf import router as pdf_router
from app.api.v1.health import router as health_router

router = APIRouter()

# Include health router (endpoints are prefix-less or prefixed)
# Wait, the spec specifies: GET /api/v1/health, POST /api/v1/scan, POST /api/v1/pdf/generate
# So let's include them in the router with appropriate tags
router.include_router(health_router)
router.include_router(scan_router)
router.include_router(pdf_router)
