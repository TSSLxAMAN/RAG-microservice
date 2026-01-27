import re
import json
import logging
import random
import requests
from typing import List, Dict, Tuple
from app.utils.vector_store import VectorStore
from app.services.pdf_processor import PDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, vector_store: VectorStore, pdf_processor: PDFProcessor):
        self.vector_store = vector_store
        self.pdf_processor = pdf_processor
    
    def train_on_pdf(self, pdf_path: str, collection_name: str) -> Tuple[bool, str, int]:
        """Train the RAG model on a PDF"""
        try:
            # Extract text from PDF
            logger.info(f"Processing PDF: {pdf_path}")
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                return False, "Failed to extract text from PDF", 0
            
            # Chunk the text
            chunks = self.pdf_processor.chunk_text(text)
            
            if not chunks:
                return False, "Failed to create text chunks", 0
            
            # Create collection and add documents
            self.vector_store.create_collection(collection_name)
            
            # Create metadata for each chunk
            metadatas = [{"chunk_id": i, "source": pdf_path} for i in range(len(chunks))]
            
            self.vector_store.add_documents(
                collection_name=collection_name,
                texts=chunks,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully trained on {len(chunks)} chunks")
            return True, f"Successfully trained on PDF with {len(chunks)} chunks", len(chunks)
            
        except Exception as e:
            logger.error(f"Error in train_on_pdf: {str(e)}")
            return False, f"Error: {str(e)}", 0
    
    def answer_question(self, collection_name: str, question: str, top_k: int = 5) -> Tuple[bool, str, List[str]]:
        """Answer a question using the trained RAG model"""
        try:
            # Check if collection exists
            if not self.vector_store.collection_exists(collection_name):
                return False, f"Collection '{collection_name}' not found. Please train first.", []
            
            # Search for relevant documents
            logger.info(f"Searching for: {question}")
            results = self.vector_store.search(
                collection_name=collection_name,
                query=question,
                n_results=top_k
            )
            
            if not results['documents'] or not results['documents'][0]:
                return False, "No relevant information found in the trained documents.", []
            
            # Get the documents and distances
            documents = results['documents'][0]
            distances = results['distances'][0] if 'distances' in results else []
            
            # Generate answer based on retrieved documents
            answer = self._generate_answer(question, documents, distances)
            
            return True, answer, documents
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            return False, f"Error: {str(e)}", []
    
    def _generate_answer(self, question: str, documents: List[str], distances: List[float]) -> str:
        """Generate an answer from retrieved documents (simple extraction-based)"""
        # This is a simple implementation. For production, you'd use a local LLM here
        
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        # Combine the most relevant documents
        context = "\n\n".join(documents[:3])  # Use top 3 documents
        
        # Simple answer generation (in production, use a local LLM like LLaMA, GPT4All, etc.)
        answer = f"Based on the trained documents:\n\n{context[:800]}..."
        
        if distances and distances[0] < 0.5:  # High confidence
            answer = f"[High Confidence] {answer}"
        elif distances and distances[0] < 0.7:  # Medium confidence
            answer = f"[Medium Confidence] {answer}"
        else:
            answer = f"[Low Confidence] {answer}"
        
        return answer
    
    def score_assignment(self, reference_collection: str, assignment_pdf_path: str) -> Tuple[bool, float, int]:
        """
        Scores assignment using Llama 3.2.
        Returns: (Success, Average Score, Question Count)
        """
        try:
            # 1. Extract Text & Parse Q/A
            text = self.pdf_processor.extract_text_from_pdf(assignment_pdf_path)
            print(text)
            qa_pairs = self._parse_assignment_qa(text)
            print(qa_pairs)
            if not qa_pairs:
                logger.warning("No Q&A format detected, falling back to simple text scoring")
                # Fallback logic if needed, or return 0
                return False, 0.0, 0

            total_score = 0
            
            # 2. Score each Question individually
            for item in qa_pairs:
                question = item['question']
                student_answer = item['answer']
                
                # A. Retrieve the "Correct Answer" context from DB
                # We use the QUESTION to find the relevant part of the textbook
                results = self.vector_store.search(
                    collection_name=reference_collection,
                    query=question, 
                    n_results=1 # Get the top matching paragraph
                )
                
                if not results['documents']:
                    continue
                    
                reference_context = results['documents'][0][0]
                
                # B. Ask Llama 3.2 to Grade it
                score = self._ask_llm_to_grade(question, student_answer, reference_context)
                total_score += score

            # 3. Calculate Average
            final_average = total_score / len(qa_pairs)
            return True, round(final_average, 2), len(qa_pairs)

        except Exception as e:
            logger.error(f"Scoring Error: {str(e)}")
            return False, 0.0, 0

    def _ask_llm_to_grade(self, question: str, answer: str, context: str) -> int:
        """Send prompt to Local LLM to get a 0-100 score"""
        prompt = f"""
        You are a strict teacher grading an exam.
        
        Reference Material: "{context}"
        
        Exam Question: "{question}"
        Student Answer: "{answer}"
        
        Task: Grade the student's answer based ONLY on the Reference Material.
        If the answer is correct according to the reference, give a high score.
        If it contradicts the reference or is irrelevant, give a low score.
        
        OUTPUT FORMAT: Return ONLY a single integer from 0 to 100. Do not write any text.
        """
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result_text = response.json()['response'].strip()
                # Extract number from response (in case LLM is chatty)
                import re
                match = re.search(r'\d+', result_text)
                if match:
                    return int(match.group())
            
            return 0 # Default to 0 on failure
            
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return 0


    def _parse_assignment_qa(self, text: str) -> List[Dict]:
        """
        Robust PDF Q&A parser with Fallback.
        Strategy 1: Look for numbered questions (e.g., "1. What is...")
        Strategy 2 (Fallback): If no numbers found, treat paragraphs as Q&A pairs.
        """
        
        # --- CLEANUP PHASE ---
        # Normalize PDF garbage
        text = text.replace('\f', '\n')
        text = re.sub(r'\n{2,}', '\n\n', text) # Normalize paragraph breaks
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        qa_pairs = []
        
        # --- STRATEGY 1: STRICT NUMBERED PARSING ---
        question_lines = []
        answer_lines = []
        in_question = False
        
        # Regex for "1.", "Q1", "1)", "(1)"
        question_start = re.compile(r'^(?:Q(?:uestion)?\s*\d+|\(?\d+[\.\)])\s*')

        for line in lines:
            # New question detected
            if question_start.match(line):
                # Flush previous
                if question_lines:
                    qa_pairs.append({
                        "question": " ".join(question_lines).strip(),
                        "answer": " ".join(answer_lines).strip()
                    })
                
                question_lines = []
                answer_lines = []
                in_question = True
                
                # Remove the number marker (e.g. "1. ") so we just get text
                clean = question_start.sub('', line).strip()
                question_lines.append(clean)
                continue

            if in_question:
                # Simple heuristic: Answers usually start after the question line
                # If the line looks like a sentence start, assume it's part of answer
                if len(answer_lines) == 0 and not line.endswith('?'):
                    in_question = False # Switch to answer mode
                    
                if in_question:
                    question_lines.append(line)
                else:
                    answer_lines.append(line)
            else:
                # If we haven't found a question number yet, ignore or add to previous
                if answer_lines:
                    answer_lines.append(line)

        # Flush the last block
        if question_lines:
            qa_pairs.append({
                "question": " ".join(question_lines).strip(),
                "answer": " ".join(answer_lines).strip()
            })

        # --- STRATEGY 2: FALLBACK (PARAGRAPH MODE) ---
        # If Strategy 1 failed to find ANY questions, we use this.
        if not qa_pairs:
            logger.warning("Regex parsing failed. Switching to Paragraph Fallback mode.")
            
            # Split text by double newlines (paragraphs)
            paragraphs = text.split('\n\n')
            
            for p in paragraphs:
                p = p.strip()
                if len(p) < 30: continue # Skip tiny garbage lines
                
                # Heuristic: Use the first sentence as the "Question" (Topic)
                # and the whole paragraph as the "Answer".
                # This allows the Vector DB to find the right context.
                sentences = re.split(r'(?<=[.!?])\s+', p)
                if sentences:
                    topic_query = sentences[0]
                    
                    qa_pairs.append({
                        "question": topic_query,  # We use this to search the DB
                        "answer": p               # We grade the whole paragraph
                    })

        return qa_pairs


    def _score_answer(self, reference_collection: str, question: str, answer: str) -> Dict:
        """Score a single answer intelligently"""
        
        # 1. Content Relevance Score (40%)
        content_score = self._score_content_relevance(reference_collection, question, answer)
        
        # 2. Answer Completeness Score (30%)
        completeness_score = self._score_answer_completeness(answer, question)
        
        # 3. Question Alignment Score (30%)
        alignment_score = self._score_question_alignment(reference_collection, question, answer)
        
        # Calculate weighted total
        total_score = (
            content_score * 0.40 +
            completeness_score * 0.30 +
            alignment_score * 0.30
        ) * 100

        
        # Generate specific feedback for this answer
        feedback = self._generate_answer_feedback(
            total_score, content_score, completeness_score, alignment_score
        )
        
        return {
            'question': question[:100] + "..." if len(question) > 100 else question,
            'answer_length': len(answer.split()),
            'score': round(total_score, 2),
            'content_relevance': round(content_score * 100, 2),
            'completeness': round(completeness_score * 100, 2),
            'question_alignment': round(alignment_score * 100, 2),
            'feedback': feedback
        }


    def _score_content_relevance(self, reference_collection: str, question: str, answer: str) -> float:
        """Check how well the answer matches reference material (0-1)"""
        
        try:
            # Split answer into sentences for better matching
            sentences = re.split(r'[.!?]+', answer)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if not sentences:
                return 0.0
            
            # Score each sentence against reference
            sentence_scores = []
            
            for sentence in sentences[:5]:  # Check up to 5 sentences
                query = f"{question}. {sentence}. {answer[:150]}"
                results = self.vector_store.search(
                    collection_name=reference_collection,
                    query=query,
                    n_results=3
                )

                
                if results['distances'] and results['distances'][0]:
                    distances = results['distances'][0]
                    similarities = [
                        max(0, 1 - (d * 0.7))
                        for d in distances
                    ]
                    sentence_scores.append(sum(similarities) / len(similarities))

            
            # Return average similarity
            return sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error in content relevance scoring: {str(e)}")
            return 0.5


    def _score_answer_completeness(self, answer: str, question: str) -> float:
        """Score answer completeness based on length and structure (0-1)"""
        
        word_count = len(answer.split())
        sentence_count = len(re.split(r'[.!?]+', answer))
        
        # Determine expected length based on question type
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['define', 'what is', 'list']):
            # Short answer expected (30-80 words)
            expected_min, expected_max = 30, 80
        elif any(word in question_lower for word in ['explain', 'describe', 'discuss']):
            # Medium answer expected (80-150 words)
            expected_min, expected_max = 80, 150
        elif any(word in question_lower for word in ['analyze', 'evaluate', 'compare', 'critically']):
            # Long answer expected (150-300 words)
            expected_min, expected_max = 150, 300
        else:
            # Default medium length
            expected_min, expected_max = 60, 120
        
        # Score based on word count
        if word_count < expected_min * 0.5:
            length_score = 0.3  # Too short
        elif word_count < expected_min:
            length_score = 0.6  # Somewhat short
        elif expected_min <= word_count <= expected_max:
            length_score = 1.0  # Good length
        elif word_count <= expected_max * 1.5:
            length_score = 0.9  # Slightly long but okay
        else:
            length_score = 0.7  # Too verbose
        
        # Bonus for proper structure (multiple sentences)
        structure_score = min(1.0, sentence_count / 3) if sentence_count > 0 else 0.5
        
        # Check for key elements (examples, explanations)
        has_examples = any(word in answer.lower() for word in ['example', 'such as', 'like', 'for instance'])
        has_explanation = any(word in answer.lower() for word in ['because', 'therefore', 'thus', 'means', 'refers to'])
        
        quality_bonus = 0.1 * (has_examples + has_explanation)
        
        final_score = min(1.0, (length_score * 0.6 + structure_score * 0.4) + quality_bonus)
        
        return final_score


    def _score_question_alignment(self, reference_collection: str, question: str, answer: str) -> float:
        """Check if answer actually addresses the question (0-1)"""
        
        try:
            # Extract key terms from question
            question_lower = question.lower()
            
            # Remove question words to get the topic
            question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'define', 'explain', 'describe', 'list', 'discuss', 'analyze', 'evaluate', 'compare']
            
            topic_words = question_lower
            for qw in question_words:
                topic_words = topic_words.replace(qw, '')
            
            # Get important words from question (nouns, key terms)
            important_words = [word for word in topic_words.split() if len(word) > 4]
            
            # Check how many question keywords appear in answer
            answer_lower = answer.lower()
            matches = sum(1 for word in important_words if word in answer_lower)
            
            keyword_score = min(1.0, matches / max(1, len(important_words)))
            
            # Check if answer is relevant to the question using semantic search
            # Combine question + answer and check against reference
            combined = f"{question} {answer[:200]}"
            
            results = self.vector_store.search(
                collection_name=reference_collection,
                query=combined,
                n_results=2
            )
            
            if results['distances'] and results['distances'][0]:
                distance = results['distances'][0][0]
                semantic_score = max(0, 1 - distance)
            else:
                semantic_score = 0.5
            
            # Combine scores
            alignment_score = (keyword_score * 0.4 + semantic_score * 0.6)
            
            return alignment_score
            
        except Exception as e:
            logger.error(f"Error in question alignment scoring: {str(e)}")
            return 0.5


    def _generate_answer_feedback(self, total_score: float, content: float, completeness: float, alignment: float) -> str:
        """Generate specific feedback for an answer"""
        
        feedback_parts = []
        
        # Overall assessment
        if total_score >= 85:
            feedback_parts.append("Excellent answer!")
        elif total_score >= 70:
            feedback_parts.append("Good answer.")
        elif total_score >= 50:
            feedback_parts.append("Fair answer, but needs improvement.")
        else:
            feedback_parts.append("Needs significant improvement.")
        
        # Specific issues
        if content < 0.6:
            feedback_parts.append("Content doesn't match reference material well.")
        
        if completeness < 0.6:
            feedback_parts.append("Answer is too brief or lacks proper structure.")
        
        if alignment < 0.6:
            feedback_parts.append("Answer doesn't fully address the question asked.")
        
        return " ".join(feedback_parts)


    def _generate_detailed_feedback(self, avg_score: float, detailed_scores: List[Dict]) -> str:
        """Generate comprehensive feedback for the entire assignment"""
        
        feedback_parts = []
        
        # Overall performance
        if avg_score >= 85:
            feedback_parts.append("🌟 Excellent work! Your assignment demonstrates strong understanding of the material.")
        elif avg_score >= 70:
            feedback_parts.append("✅ Good job! You have a solid grasp of most concepts.")
        elif avg_score >= 50:
            feedback_parts.append("⚠️ Fair attempt. You understand some concepts but need to study more thoroughly.")
        else:
            feedback_parts.append("❌ Needs significant improvement. Please review the material and seek help.")
        
        # Specific strengths and weaknesses
        strong_answers = [s for s in detailed_scores if s['score'] >= 80]
        weak_answers = [s for s in detailed_scores if s['score'] < 60]
        
        if strong_answers:
            feedback_parts.append(f"\n\n✓ Strong areas: {len(strong_answers)} questions answered excellently.")
        
        if weak_answers:
            feedback_parts.append(f"\n✗ Needs work: {len(weak_answers)} questions need more attention.")
        
        # Average metrics
        avg_content = sum(s['content_relevance'] for s in detailed_scores) / len(detailed_scores)
        avg_completeness = sum(s['completeness'] for s in detailed_scores) / len(detailed_scores)
        avg_alignment = sum(s['question_alignment'] for s in detailed_scores) / len(detailed_scores)
        
        feedback_parts.append(f"\n\n📊 Performance Breakdown:")
        feedback_parts.append(f"• Content Accuracy: {avg_content:.1f}%")
        feedback_parts.append(f"• Answer Completeness: {avg_completeness:.1f}%")
        feedback_parts.append(f"• Question Alignment: {avg_alignment:.1f}%")
        
        # Recommendations
        feedback_parts.append("\n\n💡 Recommendations:")
        if avg_content < 70:
            feedback_parts.append("• Review the reference material more carefully")
        if avg_completeness < 70:
            feedback_parts.append("• Provide more detailed and structured answers")
        if avg_alignment < 70:
            feedback_parts.append("• Make sure to directly answer what is being asked")
        
        return "".join(feedback_parts)


    def _get_scoring_breakdown(self, detailed_scores: List[Dict]) -> Dict:
        """Get statistical breakdown of scores"""
        
        if not detailed_scores:
            return {}
        
        scores = [s['score'] for s in detailed_scores]
        
        return {
            "highest_score": round(max(scores), 2),
            "lowest_score": round(min(scores), 2),
            "median_score": round(sorted(scores)[len(scores)//2], 2),
            "questions_above_80": sum(1 for s in scores if s >= 80),
            "questions_below_60": sum(1 for s in scores if s < 60)
        }


    def _score_by_chunks(self, reference_collection: str, assignment_text: str) -> Tuple[bool, float, str, Dict]:
        """Fallback: Score by chunks if no Q&A structure found"""
        
        # Chunk the assignment text
        assignment_chunks = self.pdf_processor.chunk_text(assignment_text)
        
        if not assignment_chunks:
            return False, 0.0, "Failed to process assignment", {}
        
        # Score each chunk
        chunk_scores = []
        
        for chunk in assignment_chunks:
            results = self.vector_store.search(
                collection_name=reference_collection,
                query=chunk,
                n_results=1
            )
            
            if results['distances'] and results['distances'][0]:
                distance = results['distances'][0][0]
                similarity = max(0, 1 - distance) * 100
                chunk_scores.append(similarity)
        
        avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
        
        feedback = f"Assignment scored based on content chunks. {self._generate_feedback(avg_score)}"
        
        details = {
            "total_chunks": len(assignment_chunks),
            "average_score": round(avg_score, 2),
            "scoring_method": "chunk-based",
            "chunk_scores": [round(s, 2) for s in chunk_scores[:10]]
        }
        
        return True, round(avg_score, 2), feedback, details
    
    def _generate_feedback(self, score: float) -> str:
        """Generate feedback based on score"""
        if score >= 90:
            return "Excellent! The assignment closely matches the reference material."
        elif score >= 75:
            return "Good work! The assignment covers most of the reference material well."
        elif score >= 60:
            return "Fair. The assignment covers some topics but could be more comprehensive."
        elif score >= 40:
            return "Needs improvement. The assignment has limited alignment with reference material."
        else:
            return "Poor. The assignment shows minimal alignment with the reference material."
        
    def generate_questions(
    self, 
    collection_name: str, 
    num_questions: int = 5, 
    difficulty: str = "moderate"
) -> Tuple[bool, str, List[dict]]:
        """Generate questions using a Local LLM (Ollama)"""
        try:
            # 1. Retrieve Documents (Randomly select a few chunks from the DB)
            collection = self.vector_store.client.get_collection(name=collection_name)
            all_docs = collection.get()
            documents = all_docs['documents']

            if not documents:
                return False, "No documents found", []

            # Pick random distinct contexts to ensure variety
            selected_docs = random.sample(documents, min(num_questions, len(documents)))
            
            generated_questions = []

            # 2. Iterate and Generate using LLM
            for i, doc_context in enumerate(selected_docs):
                # Construct the Prompt
                prompt = f"""
                Context: "{doc_context}"
                
                Task: Create 1 distinct {difficulty} level question and its answer based STRICTLY on the text above.
                
                Format your response as a JSON object with keys: "question", "answer", "topic".
                """
                
                # Call Local LLM (Ollama Example)
                # Ensure Ollama is running on port 11434
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2", # Or "mistral", "llama3"
                        "prompt": prompt,
                        "stream": False,
                        "format": "json" 
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    try:
                        # Parse the JSON string from the LLM
                        llm_data = json.loads(result['response'])
                        
                        generated_questions.append({
                            "question_number": i + 1,
                            "question": llm_data.get("question"),
                            "difficulty": difficulty,
                            "topic": llm_data.get("topic", "General"),
                            "expected_answer_hint": llm_data.get("answer")
                        })
                    except json.JSONDecodeError:
                        continue # Skip if LLM hallucinated bad JSON

            return True, f"Generated {len(generated_questions)} questions", generated_questions

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return False, str(e), []

    def delete_collection(self, collection_name: str) -> Tuple[bool, str]:
        """Delete a collection from the vector database"""
        try:
            # Check if collection exists
            if not self.vector_store.collection_exists(collection_name):
                return False, f"Collection '{collection_name}' not found."
            
            # Delete the collection
            self.vector_store.client.delete_collection(name=collection_name)
            
            logger.info(f"Successfully deleted collection: {collection_name}")
            return True, f"Collection '{collection_name}' deleted successfully."
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False, f"Error deleting collection: {str(e)}"