"""
OCR Service - Google Cloud Vision integration
Simple, reliable OCR without preprocessing
"""
import os
import io
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

from google.cloud import vision
from PIL import Image
import pymupdf

from backend.config import settings

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedOCRResult:
    """Simplified OCR result for compatibility"""
    full_text: str
    confidence: float
    is_handwritten: bool
    has_mixed_content: bool
    raw_result: Any = None


@dataclass
class BoundingBox:
    """Bounding box for text region"""
    x: int
    y: int
    width: int
    height: int
    vertices: List[Dict[str, int]]


@dataclass
class TextBlock:
    """Individual text block with metadata"""
    text: str
    confidence: float
    bounding_box: BoundingBox
    block_type: str
    is_handwritten: bool
    language: Optional[str] = None
    page_number: int = 1


@dataclass
class OCRResult:
    """Complete OCR result for a document"""
    raw_text: str
    cleaned_text: str
    blocks: List[TextBlock]
    overall_confidence: float
    content_type: str
    page_count: int
    processing_time_ms: int
    word_count: int
    low_confidence_regions: List[TextBlock]
    metadata: Dict[str, Any]


class OCRService:
    """Simple OCR Service using Google Cloud Vision"""
    
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
        self.low_confidence_threshold = settings.OCR_CONFIDENCE_THRESHOLD
    
    def process_document(self, file_path: str) -> OCRResult:
        """Process a document (image or PDF)"""
        start_time = datetime.now()
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            result = self._process_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
            result = self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        result.processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        return result
    
    def process_image_bytes(self, image_bytes: bytes, filename: str = "upload") -> OCRResult:
        """Process image from bytes"""
        start_time = datetime.now()
        result = self._extract_text(image_bytes)
        result.processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        return result
    
    def _process_image(self, file_path: Path) -> OCRResult:
        """Process a single image file"""
        with open(file_path, 'rb') as f:
            content = f.read()
        return self._extract_text(content)
    
    def _process_pdf(self, file_path: Path) -> OCRResult:
        """Process a multi-page PDF"""
        doc = pymupdf.open(file_path)
        all_blocks = []
        all_text_parts = []
        total_confidence = 0
        confidence_count = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = pymupdf.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            
            page_result = self._extract_text(img_bytes, page_num + 1)
            all_blocks.extend(page_result.blocks)
            all_text_parts.append(page_result.raw_text)
            
            if page_result.overall_confidence > 0:
                total_confidence += page_result.overall_confidence
                confidence_count += 1
        
        doc.close()
        
        combined_text = "\n\n".join(all_text_parts)
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
        
        return OCRResult(
            raw_text=combined_text,
            cleaned_text=combined_text,
            blocks=all_blocks,
            overall_confidence=avg_confidence,
            content_type=self._detect_content_type(all_blocks),
            page_count=len(all_text_parts),
            processing_time_ms=0,
            word_count=len(combined_text.split()),
            low_confidence_regions=[b for b in all_blocks if b.confidence < self.low_confidence_threshold],
            metadata={"source_file": str(file_path), "file_type": "pdf"}
        )
    
    def _extract_text(self, image_bytes: bytes, page_number: int = 1) -> OCRResult:
        """Extract text using Google Cloud Vision"""
        image = vision.Image(content=image_bytes)
        
        # Use document_text_detection for best results
        response = self.client.document_text_detection(
            image=image,
            image_context={"language_hints": ["en"]}
        )
        
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        
        blocks = []
        raw_text = ""
        total_confidence = 0
        confidence_count = 0
        
        if response.full_text_annotation:
            raw_text = response.full_text_annotation.text
            
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join([s.text for s in word.symbols])
                            word_conf = word.confidence if hasattr(word, 'confidence') else 0.9
                            
                            vertices = word.bounding_box.vertices
                            bbox = BoundingBox(
                                x=vertices[0].x if vertices else 0,
                                y=vertices[0].y if vertices else 0,
                                width=(vertices[2].x - vertices[0].x) if len(vertices) >= 3 else 0,
                                height=(vertices[2].y - vertices[0].y) if len(vertices) >= 3 else 0,
                                vertices=[{"x": v.x, "y": v.y} for v in vertices]
                            )
                            
                            text_block = TextBlock(
                                text=word_text,
                                confidence=word_conf,
                                bounding_box=bbox,
                                block_type='word',
                                is_handwritten=word_conf < 0.75,
                                page_number=page_number
                            )
                            blocks.append(text_block)
                            total_confidence += word_conf
                            confidence_count += 1
        
        overall_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
        
        return OCRResult(
            raw_text=raw_text,
            cleaned_text=raw_text,
            blocks=blocks,
            overall_confidence=overall_confidence,
            content_type=self._detect_content_type(blocks),
            page_count=1,
            processing_time_ms=0,
            word_count=len(raw_text.split()),
            low_confidence_regions=[b for b in blocks if b.confidence < self.low_confidence_threshold],
            metadata={"page_number": page_number, "total_blocks": len(blocks)}
        )
    
    def _detect_content_type(self, blocks: List[TextBlock]) -> str:
        """Detect if document is printed, handwritten, or mixed"""
        if not blocks:
            return "unknown"
        handwritten_count = sum(1 for b in blocks if b.is_handwritten)
        ratio = handwritten_count / len(blocks)
        if ratio > 0.7:
            return "handwritten"
        elif ratio > 0.2:
            return "mixed"
        return "printed"
    
    def extract_text(self, file_path: str) -> SimplifiedOCRResult:
        """Simplified interface for document processor"""
        try:
            result = self.process_document(file_path)
            return SimplifiedOCRResult(
                full_text=result.raw_text,
                confidence=result.overall_confidence,
                is_handwritten=result.content_type == "handwritten",
                has_mixed_content=result.content_type == "mixed",
                raw_result=result
            )
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return SimplifiedOCRResult(
                full_text="",
                confidence=0.0,
                is_handwritten=False,
                has_mixed_content=False,
                raw_result=None
            )
    
    def to_dict(self, result: OCRResult) -> Dict:
        """Convert OCR result to dictionary"""
        return {
            "raw_text": result.raw_text,
            "cleaned_text": result.cleaned_text,
            "blocks": [
                {
                    "text": b.text,
                    "confidence": b.confidence,
                    "bounding_box": asdict(b.bounding_box),
                    "block_type": b.block_type,
                    "is_handwritten": b.is_handwritten,
                    "page_number": b.page_number
                }
                for b in result.blocks
            ],
            "overall_confidence": result.overall_confidence,
            "content_type": result.content_type,
            "page_count": result.page_count,
            "processing_time_ms": result.processing_time_ms,
            "word_count": result.word_count,
            "low_confidence_count": len(result.low_confidence_regions),
            "metadata": result.metadata
        }
