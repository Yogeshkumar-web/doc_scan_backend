from fastapi import APIRouter, Response
import structlog
from app.schemas.pdf import PdfGenerateRequest
from app.services.pdf_service import generate_pdf
from app.core.exceptions import AppError

router = APIRouter()
logger = structlog.get_logger()

@router.post("/pdf/generate")
async def generate_pdf_endpoint(request: PdfGenerateRequest):
    if not request.pages or len(request.pages) == 0:
        logger.warning("empty_page_list")
        raise AppError("EMPTY_PAGE_LIST", "Page list cannot be empty.", status_code=400)
    
    try:
        pdf_bytes = generate_pdf(request.pages)
    except Exception as e:
        logger.error("pdf_generation_failed", error=str(e), exc_info=True)
        raise AppError("PDF_GENERATION_FAILED", f"Failed to generate PDF: {str(e)}", status_code=400)
        
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="scanned_document.pdf"'
        }
    )
