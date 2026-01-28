"""
OCR Service - Google Cloud Vision integration with enhanced features
Handles: Printed text, handwritten text, mixed content detection,
bounding boxes, confidence tracking
"""
import os
import io
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
from pathlib import Path

from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image
import pymupdf  # PyMuPDF for PDF handling

from backend.config import settings

# Set up Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedOCRResult:
    """Simplified OCR result for compatibility with document processor"""
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
    block_type: str  # 'word', 'line', 'paragraph', 'block'
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
    content_type: str  # 'printed', 'handwritten', 'mixed'
    page_count: int
    processing_time_ms: int
    word_count: int
    low_confidence_regions: List[TextBlock]
    metadata: Dict[str, Any]


class OCRService:
    """
    Enhanced OCR Service using Google Cloud Vision
    
    Features:
    - Printed and handwritten text extraction
    - Mixed content detection
    - Bounding box preservation
    - Confidence tracking
    - Multi-page PDF support
    """
    
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
        self.handwriting_threshold = 0.85  # Confidence threshold for handwriting detection
        self.low_confidence_threshold = settings.OCR_CONFIDENCE_THRESHOLD
    
    def process_document(self, file_path: str) -> OCRResult:
        """
        Process a document (image or PDF) and extract text
        
        Args:
            file_path: Path to the document file
            
        Returns:
            OCRResult with all extracted data
        """
        start_time = datetime.now()
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            result = self._process_pdf(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
            result = self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        result.processing_time_ms = processing_time
        
        return result
    
    def process_image_bytes(self, image_bytes: bytes, filename: str = "upload") -> OCRResult:
        """Process image from bytes"""
        start_time = datetime.now()
        result = self._extract_text_from_bytes(image_bytes)
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        result.processing_time_ms = processing_time
        return result
    
    def _process_image(self, file_path: Path) -> OCRResult:
        """Process a single image file"""
        with open(file_path, 'rb') as f:
            content = f.read()
        return self._extract_text_from_bytes(content)
    
    def _process_pdf(self, file_path: Path) -> OCRResult:
        """Process a multi-page PDF"""
        doc = pymupdf.open(file_path)
        all_blocks = []
        all_text_parts = []
        total_confidence = 0
        confidence_count = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Convert page to image
            mat = pymupdf.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            
            # OCR the page
            page_result = self._extract_text_from_bytes(img_bytes, page_number=page_num + 1)
            all_blocks.extend(page_result.blocks)
            all_text_parts.append(page_result.raw_text)
            
            if page_result.overall_confidence > 0:
                total_confidence += page_result.overall_confidence
                confidence_count += 1
        
        doc.close()
        
        # Combine results
        combined_text = "\n\n--- Page Break ---\n\n".join(all_text_parts)
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
        
        # Determine content type
        content_type = self._detect_content_type(all_blocks)
        
        # Find low confidence regions
        low_confidence_regions = [
            block for block in all_blocks 
            if block.confidence < self.low_confidence_threshold
        ]
        
        return OCRResult(
            raw_text=combined_text,
            cleaned_text=combined_text,  # Will be cleaned by TextCleaningService
            blocks=all_blocks,
            overall_confidence=avg_confidence,
            content_type=content_type,
            page_count=len(doc) if hasattr(doc, '__len__') else 1,
            processing_time_ms=0,
            word_count=len(combined_text.split()),
            low_confidence_regions=low_confidence_regions,
            metadata={
                "source_file": str(file_path),
                "file_type": "pdf"
            }
        )
    
    def _extract_text_from_bytes(self, image_bytes: bytes, page_number: int = 1) -> OCRResult:
        """
        Extract text from image bytes using Google Cloud Vision
        Uses both text_detection and document_text_detection for best results
        """
        image = vision.Image(content=image_bytes)
        
        # Use document_text_detection for better structured output
        response = self.client.document_text_detection(
            image=image,
            image_context={"language_hints": ["en"]}
        )
        
        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")
        
        blocks = []
        raw_text = ""
        total_confidence = 0
        confidence_count = 0
        
        if response.full_text_annotation:
            raw_text = response.full_text_annotation.text
            
            # Process each page (usually 1 for images)
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_text = ""
                    block_confidence = 0
                    block_conf_count = 0
                    
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join([
                                symbol.text for symbol in word.symbols
                            ])
                            word_confidence = word.confidence if hasattr(word, 'confidence') else 0.9
                            
                            # Create bounding box
                            vertices = word.bounding_box.vertices
                            bbox = BoundingBox(
                                x=vertices[0].x if vertices else 0,
                                y=vertices[0].y if vertices else 0,
                                width=(vertices[2].x - vertices[0].x) if len(vertices) >= 3 else 0,
                                height=(vertices[2].y - vertices[0].y) if len(vertices) >= 3 else 0,
                                vertices=[{"x": v.x, "y": v.y} for v in vertices]
                            )
                            
                            # Detect if handwritten based on confidence patterns
                            is_handwritten = self._is_likely_handwritten(word_confidence, word_text)
                            
                            word_block = TextBlock(
                                text=word_text,
                                confidence=word_confidence,
                                bounding_box=bbox,
                                block_type='word',
                                is_handwritten=is_handwritten,
                                page_number=page_number
                            )
                            blocks.append(word_block)
                            
                            block_text += word_text + " "
                            block_confidence += word_confidence
                            block_conf_count += 1
                            total_confidence += word_confidence
                            confidence_count += 1
        
        # Calculate overall confidence
        overall_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
        
        # Determine content type
        content_type = self._detect_content_type(blocks)
        
        # Find low confidence regions
        low_confidence_regions = [
            block for block in blocks 
            if block.confidence < self.low_confidence_threshold
        ]
        
        return OCRResult(
            raw_text=raw_text,
            cleaned_text=raw_text,
            blocks=blocks,
            overall_confidence=overall_confidence,
            content_type=content_type,
            page_count=1,
            processing_time_ms=0,
            word_count=len(raw_text.split()),
            low_confidence_regions=low_confidence_regions,
            metadata={
                "page_number": page_number,
                "total_blocks": len(blocks)
            }
        )
    
    def _is_likely_handwritten(self, confidence: float, text: str) -> bool:
        """
        Heuristic to detect if text is likely handwritten
        Handwritten text typically has:
        - Lower confidence scores
        - More irregular patterns
        """
        # Lower confidence often indicates handwriting
        if confidence < 0.75:
            return True
        
        # Check for common handwriting patterns (irregular capitalization, etc.)
        if len(text) > 1:
            # Unusual character patterns
            has_mixed_case = any(c.isupper() for c in text[1:]) and any(c.islower() for c in text)
            if has_mixed_case and confidence < 0.85:
                return True
        
        return False
    
    def _detect_content_type(self, blocks: List[TextBlock]) -> str:
        """Detect if document is printed, handwritten, or mixed"""
        if not blocks:
            return "unknown"
        
        handwritten_count = sum(1 for b in blocks if b.is_handwritten)
        total_count = len(blocks)
        
        handwritten_ratio = handwritten_count / total_count if total_count > 0 else 0
        
        if handwritten_ratio > 0.7:
            return "handwritten"
        elif handwritten_ratio > 0.2:
            return "mixed"
        else:
            return "printed"
    
    def get_text_at_region(self, blocks: List[TextBlock], 
                          x: int, y: int, width: int, height: int) -> str:
        """Get text within a specific bounding region"""
        region_text = []
        for block in blocks:
            bbox = block.bounding_box
            # Check if block overlaps with region
            if (bbox.x < x + width and bbox.x + bbox.width > x and
                bbox.y < y + height and bbox.y + bbox.height > y):
                region_text.append(block.text)
        return " ".join(region_text)
    
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
    
    def extract_text(self, file_path: str) -> SimplifiedOCRResult:
        """
        Extract text from a document - simplified interface for document processor
        
        Args:
            file_path: Path to the document file
            
        Returns:
            SimplifiedOCRResult with extracted text and metadata
        """
        try:
            result = self.process_document(file_path)
            
            is_handwritten = result.content_type == "handwritten"
            has_mixed_content = result.content_type == "mixed"
            
            return SimplifiedOCRResult(
                full_text=result.raw_text,
                confidence=result.overall_confidence,
                is_handwritten=is_handwritten,
                has_mixed_content=has_mixed_content,
                raw_result=result
            )
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            # Return empty result on error
            return SimplifiedOCRResult(
                full_text="",
                confidence=0.0,
                is_handwritten=False,
                has_mixed_content=False,
                raw_result=None
            )
