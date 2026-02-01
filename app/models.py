from pydantic import BaseModel
from typing import Optional, List
from typing import List, Optional
from enum import Enum

class TrainRequest(BaseModel):
    collection_name: str
    

class TrainResponse(BaseModel):
    success: bool
    message: str
    collection_name: str
    chunks_count: int


class QuestionRequest(BaseModel):
    collection_name: str
    question: str
    top_k: Optional[int] = 5


class QuestionResponse(BaseModel):
    success: bool
    question: str
    answer: str
    sources: List[str]


class ScoreRequest(BaseModel):
    collection_name: str  # The trained reference model
    
class QuestionGrade(BaseModel):
    question_number: int
    score: int
    feedback: Optional[str] = None
    
class ScoreResponse(BaseModel):
    success: bool
    score: float  # Just the number (0-100)
    total_questions: int
    breakdown: List[QuestionGrade]

class DifficultyLevel(str, Enum):
    easy = "easy"
    moderate = "moderate"
    hard = "hard"


class GenerateQuestionsRequest(BaseModel):
    collection_name: str
    num_questions: int = 5
    difficulty: DifficultyLevel = DifficultyLevel.moderate

class Question(BaseModel):
    question_number: int
    question: str
    difficulty: str
    topic: str
    expected_answer_hint: str


class GenerateQuestionsResponse(BaseModel):
    success: bool
    collection_name: str
    total_questions: int
    difficulty: str
    questions: List[Question]

class DeleteCollectionRequest(BaseModel):
    collection_name: str

class DeleteCollectionResponse(BaseModel):
    success: bool
    message: str
    collection_name: str