from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import shutil
from dotenv import load_dotenv
import logging

from app.models import (
    TrainResponse,
    QuestionRequest, QuestionResponse,
    ScoreResponse,
    GenerateQuestionsRequest, GenerateQuestionsResponse,
    Question, DeleteCollectionResponse
)

from app.utils.vector_store import VectorStore
from app.services.pdf_processor import PDFProcessor
from app.services.rag_service import RAGService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Microservice",
    description="A RAG-based microservice for PDF training, Q&A, and assignment scoring",
    version="1.0.0"
)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vectorstore")
PDF_UPLOAD_PATH = os.getenv("PDF_UPLOAD_PATH", "./data/pdfs")

# Create directories if they don't exist
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(PDF_UPLOAD_PATH, exist_ok=True)

# Initialize services
vector_store = VectorStore(persist_directory=VECTOR_STORE_PATH, model_name=MODEL_NAME)
pdf_processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
rag_service = RAGService(vector_store=vector_store, pdf_processor=pdf_processor)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Microservice",
        "version": "1.0.0"
    }


@app.post("/api/train", response_model=TrainResponse)
async def train_endpoint(
    collection_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Train the RAG model on a PDF file
    
    - **collection_name**: Unique name for this training session
    - **file**: PDF file to train on
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        file_path = os.path.join(PDF_UPLOAD_PATH, f"{collection_name}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved PDF to: {file_path}")
        
        # Train the RAG model
        success, message, chunks_count = rag_service.train_on_pdf(
            pdf_path=file_path,
            collection_name=collection_name
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=message)
        
        return TrainResponse(
            success=True,
            message=message,
            collection_name=collection_name,
            chunks_count=chunks_count
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in train endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/question", response_model=QuestionResponse)
async def question_endpoint(request: QuestionRequest):
    """
    Ask a question based on trained RAG model
    
    - **collection_name**: Name of the trained collection
    - **question**: Question to ask
    - **top_k**: Number of relevant chunks to retrieve (default: 5)
    """
    try:
        success, answer, sources = rag_service.answer_question(
            collection_name=request.collection_name,
            question=request.question,
            top_k=request.top_k
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=answer)
        
        return QuestionResponse(
            success=True,
            question=request.question,
            answer=answer,
            sources=sources
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in question endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/score", response_model=ScoreResponse)
async def score_endpoint(
    collection_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Score an assignment PDF against a trained reference model
    
    - **collection_name**: Name of the trained reference collection
    - **file**: Assignment PDF to score
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded assignment file
        assignment_path = os.path.join(
            PDF_UPLOAD_PATH, 
            f"assignment_{file.filename}"
        )
        
        with open(assignment_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved assignment to: {assignment_path}")
        
        # Score the assignment
        success, score, total_questions = rag_service.score_assignment(
            reference_collection=collection_name,
            assignment_pdf_path=assignment_path
        )
        
        if not success:
            return ScoreResponse(
                    success=False,
                    score=0.0,
                    total_questions=0
                )
        
        return ScoreResponse(
            success=True,
            score=score,
            total_questions=total_questions
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collections")
async def list_collections():
    """List all available trained collections"""
    try:
        collections = vector_store.client.list_collections()
        collection_names = [col.name for col in collections]
        
        return {
            "success": True,
            "collections": collection_names,
            "count": len(collection_names)
        }
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_questions_endpoint(request: GenerateQuestionsRequest):
    """
    Generate questions from trained material
    
    - **collection_name**: Name of the trained collection
    - **num_questions**: Number of questions to generate (default: 5)
    - **difficulty**: Difficulty level - easy, moderate, or hard (default: moderate)
    """
    try:
        success, message, questions = rag_service.generate_questions(
            collection_name=request.collection_name,
            num_questions=request.num_questions,
            difficulty=request.difficulty.value
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=message)
        
        # Convert to Question objects
        question_objects = [Question(**q) for q in questions]
        
        return GenerateQuestionsResponse(
            success=True,
            collection_name=request.collection_name,
            total_questions=len(question_objects),
            difficulty=request.difficulty.value,
            questions=question_objects
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in generate_questions endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/api/collection/{collection_name}", response_model=DeleteCollectionResponse)
async def delete_collection_endpoint(collection_name: str):
    """
    Delete a trained collection from the database
    
    - **collection_name**: Name of the collection to delete
    """
    try:
        success, message = rag_service.delete_collection(collection_name=collection_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=message)
        
        return DeleteCollectionResponse(
            success=True,
            message=message,
            collection_name=collection_name
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in delete_collection endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)