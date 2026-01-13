import re
import os
import logging
import random
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
    
    def score_assignment(self, reference_collection: str, assignment_pdf_path: str) -> Tuple[bool, float, str, Dict]:
        """Smart scoring system for assignments with detailed analysis"""
        try:
            # Check if reference collection exists
            if not self.vector_store.collection_exists(reference_collection):
                return False, 0.0, f"Reference collection '{reference_collection}' not found.", {}
            
            # Extract text from assignment PDF
            logger.info(f"Scoring assignment: {assignment_pdf_path}")
            assignment_text = self.pdf_processor.extract_text_from_pdf(assignment_pdf_path)
            
            if not assignment_text.strip():
                return False, 0.0, "Failed to extract text from assignment PDF", {}
            
            # Parse the assignment into Q&A pairs
            qa_pairs = self._parse_assignment_qa(assignment_text)
            
            if not qa_pairs:
                # Fallback to chunk-based scoring if no Q&A structure found
                return self._score_by_chunks(reference_collection, assignment_text)
            
            # Score each answer intelligently
            detailed_scores = []
            total_score = 0
            
            for qa in qa_pairs:
                answer_score = self._score_answer(
                    reference_collection=reference_collection,
                    question=qa['question'],
                    answer=qa['answer']
                )
                detailed_scores.append(answer_score)
                total_score += answer_score['score']
            
            # Calculate average score
            avg_score = (total_score / len(detailed_scores)) if detailed_scores else 0
            
            # Generate comprehensive feedback
            feedback = self._generate_detailed_feedback(avg_score, detailed_scores)
            
            # Prepare details
            details = {
                "total_questions": len(qa_pairs),
                "average_score": round(avg_score, 2),
                "detailed_scores": detailed_scores,
                "scoring_breakdown": self._get_scoring_breakdown(detailed_scores)
            }
            
            return True, round(avg_score, 2), feedback, details
            
        except Exception as e:
            logger.error(f"Error in score_assignment: {str(e)}")
            return False, 0.0, f"Error: {str(e)}", {}


    def _parse_assignment_qa(self, text: str) -> List[Dict]:
        """Robust PDF Q&A parser"""

        # Normalize PDF garbage
        text = text.replace('\f', '\n')
        text = re.sub(r'\n{2,}', '\n', text)

        lines = [l.strip() for l in text.split('\n') if l.strip()]

        qa_pairs = []
        question_lines = []
        answer_lines = []
        in_question = False

        question_start = re.compile(r'^(?:Q(?:uestion)?\s*\d+|\d+[\.\)])\s*')

        for line in lines:
            # New question detected
            if question_start.match(line):
                # Flush previous Q&A (even if answer is short)
                if question_lines:
                    qa_pairs.append({
                        "question": " ".join(question_lines).strip(),
                        "answer": " ".join(answer_lines).strip()
                    })

                question_lines = []
                answer_lines = []
                in_question = True

                clean = question_start.sub('', line).strip()
                question_lines.append(clean)
                continue

            # Still collecting question (wrapped lines)
            if in_question:
                # Heuristic: answers usually start with declarative sentences
                if re.match(r'^[A-Z].{20,}$', line):
                    in_question = False
                    answer_lines.append(line)
                else:
                    question_lines.append(line)
                continue

            # Answer content
            answer_lines.append(line)

        # 🔥 CRITICAL FIX: always flush last block
        if question_lines:
            qa_pairs.append({
                "question": " ".join(question_lines).strip(),
                "answer": " ".join(answer_lines).strip()
            })

        # Gentle cleanup, not execution
        qa_pairs = [
            qa for qa in qa_pairs
            if qa["question"] and len(qa["answer"].split()) >= 5
        ]

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
        """Generate meaningful, topic-focused questions"""
        try:
            if not self.vector_store.collection_exists(collection_name):
                return False, f"Collection '{collection_name}' not found.", []
            
            collection = self.vector_store.client.get_collection(name=collection_name)
            all_docs = collection.get()
            
            if not all_docs['documents']:
                return False, "No documents found in collection", []
            
            documents = all_docs['documents']
            
            # Extract key topics from all documents
            key_topics = self._extract_key_topics(documents, num_questions)
            
            # Generate questions for each topic
            questions = []
            for idx, (topic, content) in enumerate(key_topics):
                question_data = self._generate_topic_question(
                    topic=topic,
                    content=content,
                    question_num=idx + 1,
                    difficulty=difficulty
                )
                questions.append(question_data)
            
            return True, f"Successfully generated {len(questions)} questions", questions
            
        except Exception as e:
            logger.error(f"Error in generate_questions: {str(e)}")
            return False, f"Error: {str(e)}", []


    def _extract_key_topics(self, documents: List[str], num_topics: int) -> List[tuple]:
        """Extract key topics with their content"""
        
        topics = []
        
        for doc in documents:
            # Clean document
            doc = doc.strip()
            doc = re.sub(r'\s+', ' ', doc)
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', doc)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            
            for sentence in sentences:
                # Look for topic indicators
                topic_patterns = [
                    r'^([A-Z][a-zA-Z\s]+(?:Learning|Algorithm|Model|Network|Tree|Regression|Classification))',
                    r'([A-Z][a-zA-Z\s]+)\s+(?:is|are|refers to|means|involves)',
                    r'^(\d+\.\s*)?([A-Z][a-zA-Z\s]+)',
                ]
                
                for pattern in topic_patterns:
                    match = re.search(pattern, sentence)
                    if match:
                        topic = match.group(1) if len(match.groups()) == 1 else match.group(2)
                        topic = topic.strip().strip('0123456789. ')
                        
                        if len(topic.split()) >= 2:  # At least 2 words
                            # Get surrounding context
                            start_idx = max(0, sentences.index(sentence) - 1)
                            end_idx = min(len(sentences), sentences.index(sentence) + 2)
                            context = ' '.join(sentences[start_idx:end_idx])
                            
                            topics.append((topic, context))
                            break
        
        # Remove duplicates and select diverse topics
        unique_topics = []
        seen = set()
        
        for topic, content in topics:
            topic_lower = topic.lower()
            if topic_lower not in seen and len(topic.split()) <= 6:
                unique_topics.append((topic, content))
                seen.add(topic_lower)
                
                if len(unique_topics) >= num_topics:
                    break
        
        # If we don't have enough, use random substantial chunks
        while len(unique_topics) < num_topics and documents:
            doc = random.choice(documents)
            if len(doc) > 100:
                words = doc.split()[:8]
                topic = ' '.join(words)
                unique_topics.append((topic, doc[:400]))
        
        return unique_topics[:num_topics]


    def _generate_topic_question(self, topic: str, content: str, question_num: int, difficulty: str) -> dict:
        """Generate a question for a specific topic"""
        
        if difficulty == "easy":
            templates = [
                f"What is {topic}?",
                f"Define {topic}.",
                f"Explain {topic} in simple terms.",
                f"List the main features of {topic}.",
                f"Describe what you understand by {topic}.",
            ]
            
        elif difficulty == "moderate":
            templates = [
                f"Explain how {topic} works with examples.",
                f"What is the significance of {topic} in machine learning?",
                f"Discuss the key characteristics of {topic}.",
                f"How is {topic} applied in real-world scenarios?",
                f"Compare {topic} with related concepts.",
                f"What are the advantages of using {topic}?",
            ]
            
        else:  # hard
            templates = [
                f"Critically analyze {topic}. What are its strengths and weaknesses?",
                f"How would you implement {topic} to solve a specific problem? Provide a detailed approach.",
                f"Evaluate the effectiveness of {topic} compared to alternative methods.",
                f"Discuss the challenges in implementing {topic} and propose solutions.",
                f"Design a system using {topic}. Justify your design choices.",
            ]
        
        question = random.choice(templates)
        
        # Clean answer hint
        answer_hint = content[:250] + "..." if len(content) > 250 else content
        answer_hint = re.sub(r'\s+', ' ', answer_hint).strip()
        
        return {
            "question_number": question_num,
            "question": question,
            "difficulty": difficulty,
            "topic": topic,
            "expected_answer_hint": answer_hint
        }
    
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