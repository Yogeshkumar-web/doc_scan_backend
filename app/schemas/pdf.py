from pydantic import BaseModel, Field

class PdfGenerateRequest(BaseModel):
    pages: list[str] = Field(..., min_length=1, description="List of base64 image pages")
