import PyPDF2
from typing import List
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file and clean formatting"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    
                    # 1. Remove header/footer artifacts (simple heuristic: remove single lines with numbers)
                    lines = page_text.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        # Skip lines that are just numbers (page numbers) or very short headers
                        if len(line.strip()) < 4 and line.strip().isdigit():
                            continue
                        cleaned_lines.append(line)
                    
                    text += "\n".join(cleaned_lines) + "\n\n" # Double newline between pages
            
            logger.info(f"Extracted raw text from PDF")
            
            # 2. POST-PROCESSING: Merge broken lines
            # Replace hyphenated words at line end (e.g., "Market-\ning" -> "Marketing")
            text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
            
            # Replace single newlines with space (to fix sentence flow)
            # BUT keep double newlines (paragraphs)
            # This regex looks for a newline that is NOT preceded by a period/colon 
            # and NOT followed by another newline.
            text = re.sub(r'(?<![\.\:\n])\n(?!\n)', ' ', text)
            
            # Collapse multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            logger.info(f"Cleaned text length: {len(text)}")
            return text

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks prioritizing sentence boundaries"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # If we are at the end of text, just take what's left
            if end >= text_length:
                chunks.append(text[start:].strip())
                break
                
            chunk_attempt = text[start:end]
            
            # PRIORITIZE SPLIT POINTS
            # We want to split at:
            # 1. A period followed by a space ". " (Best)
            # 2. A newline (Acceptable for paragraphs)
            # 3. A generic space " " (Worst case, mid-sentence but clean word break)
            
            last_period = chunk_attempt.rfind('. ')
            last_newline = chunk_attempt.rfind('\n')
            last_space = chunk_attempt.rfind(' ')
            
            break_point = -1
            
            # Logic: If we find a period in the second half of the chunk, use it.
            # This ensures we don't create tiny chunks just because a period appeared early.
            if last_period > self.chunk_size * 0.5:
                break_point = last_period + 1 # Include the period
            elif last_newline > self.chunk_size * 0.5:
                break_point = last_newline
            elif last_space > self.chunk_size * 0.5:
                break_point = last_space
            else:
                # Fallback: Hard cut if no delimiters found (rare)
                break_point = self.chunk_size 
                
            final_chunk = text[start : start + break_point].strip()
            
            if final_chunk:
                chunks.append(final_chunk)
            
            # Overlap Logic
            # Move start pointer back by overlap amount, but ensure we don't get stuck
            start = (start + break_point) - self.chunk_overlap
            
            # Safety check to prevent infinite loops if overlap is too big relative to split
            if start >= (start + break_point): 
                start = start + break_point 

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks