#!/usr/bin/env python3
"""
AI-Powered Resume Screener
An intelligent candidate evaluation system using Deep Learning and Vector Similarity Search

Author: AI Resume Screener Team
Date: September 14, 2025
Version: 1.0.0
"""

import gradio as gr
import os
import logging
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
import traceback

# External libraries
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

# Configuration
from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resume_screener.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handle PDF text extraction and processing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

class ResumeIndex:
    """FAISS-based vector index for resume storage and retrieval"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []
        
    def add_resume(self, text: str, candidate_name: str, file_name: str):
        """Add a resume to the index"""
        try:
            # Create embedding
            embedding = self.embedding_model.encode([text])
            
            # Add to FAISS index
            self.index.add(embedding.astype('float32'))
            
            # Store text and metadata
            self.texts.append(text)
            self.metadata.append({
                'candidate_name': candidate_name,
                'file_name': file_name,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Added resume for {candidate_name} to index")
            
        except Exception as e:
            logger.error(f"Error adding resume to index: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar resumes"""
        try:
            if len(self.texts) == 0:
                return []
            
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search
            k = min(k, len(self.texts))  # Don't search for more than available
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # Valid index
                    results.append({
                        'text': self.texts[idx],
                        'metadata': self.metadata[idx],
                        'similarity_score': float(distance),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            raise

    def search_for_position(self, job_description: str, required_skills: str = "", 
                          experience_level: str = "", k: int = 10) -> List[Dict]:
        """Search and rank resumes for a specific position"""
        try:
            if len(self.texts) == 0:
                return []
            
            # Combine job requirements into search query
            search_components = [job_description]
            if required_skills.strip():
                search_components.append(f"Skills: {required_skills}")
            if experience_level.strip():
                search_components.append(f"Experience: {experience_level}")
            
            combined_query = " ".join(search_components)
            
            # Get all resumes for detailed scoring
            k = min(k, len(self.texts))
            query_embedding = self.embedding_model.encode([combined_query])
            distances, indices = self.index.search(query_embedding.astype('float32'), len(self.texts))
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # Valid index
                    resume_text = self.texts[idx]
                    
                    # Calculate detailed scores
                    similarity_score = float(distance)
                    skill_score = self._calculate_skill_match(resume_text, required_skills)
                    experience_score = self._calculate_experience_match(resume_text, experience_level)
                    
                    # Weighted combined score (lower is better for similarity, higher for others)
                    # Invert similarity score so higher is better
                    normalized_similarity = max(0, 1.0 - (similarity_score / 10.0))
                    combined_score = (normalized_similarity * 0.4) + (skill_score * 0.4) + (experience_score * 0.2)
                    
                    results.append({
                        'text': resume_text,
                        'metadata': self.metadata[idx],
                        'similarity_score': similarity_score,
                        'skill_match_score': skill_score,
                        'experience_score': experience_score,
                        'combined_score': combined_score,
                        'rank': i + 1
                    })
            
            # Sort by combined score (descending - higher is better)
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Re-assign ranks after sorting
            for i, result in enumerate(results):
                result['rank'] = i + 1
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in position-based search: {e}")
            raise

    def _calculate_skill_match(self, resume_text: str, required_skills: str) -> float:
        """Calculate how well resume matches required skills"""
        if not required_skills.strip():
            return 0.5  # Neutral score if no skills specified
        
        resume_lower = resume_text.lower()
        skills_list = [skill.strip().lower() for skill in required_skills.split(',')]
        
        matched_skills = 0
        total_skills = len(skills_list)
        
        for skill in skills_list:
            if skill in resume_lower:
                matched_skills += 1
        
        return matched_skills / total_skills if total_skills > 0 else 0.0

    def _calculate_experience_match(self, resume_text: str, experience_level: str) -> float:
        """Calculate how well resume matches experience requirements"""
        if not experience_level.strip():
            return 0.5  # Neutral score if no experience specified
        
        resume_lower = resume_text.lower()
        experience_lower = experience_level.lower()
        
        # Extract years of experience from resume
        import re
        year_patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'(\d+)\+?\s*years?\s*experience',
            r'(\d+)\+?\s*yrs?\s*experience',
            r'experience.*?(\d+)\+?\s*years?'
        ]
        
        years_found = []
        for pattern in year_patterns:
            matches = re.findall(pattern, resume_lower)
            years_found.extend([int(match) for match in matches])
        
        if not years_found:
            # Look for experience level keywords
            if 'senior' in experience_lower or 'lead' in experience_lower:
                return 0.8 if any(word in resume_lower for word in ['senior', 'lead', 'principal', 'architect']) else 0.3
            elif 'junior' in experience_lower or 'entry' in experience_lower:
                return 0.8 if any(word in resume_lower for word in ['junior', 'entry', 'intern', 'trainee']) else 0.5
            else:
                return 0.5  # Neutral if can't determine
        
        max_years = max(years_found)
        
        # Score based on experience level
        if 'senior' in experience_lower or 'lead' in experience_lower:
            return min(1.0, max_years / 8.0)  # 8+ years is excellent for senior
        elif 'mid' in experience_lower:
            return 1.0 if 3 <= max_years <= 7 else max(0.0, 1.0 - abs(max_years - 5) / 5.0)
        elif 'junior' in experience_lower or 'entry' in experience_lower:
            return 1.0 if max_years <= 3 else max(0.0, 1.0 - (max_years - 3) / 3.0)
        else:
            return min(1.0, max_years / 10.0)  # General scoring

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for resume Q&A"""
    
    def __init__(self, resume_index, qa_model, summarizer):
        self.resume_index = resume_index
        self.qa_model = qa_model
        self.summarizer = summarizer
    
    def answer_query(self, query: str, max_context_length: int = 1000) -> str:
        """Answer questions using RAG with resume context"""
        try:
            # Retrieve relevant resumes
            search_results = self.resume_index.search(query, k=3)
            
            if not search_results:
                return "No resumes found in the database. Please upload some resumes first."
            
            # Prepare context from top results
            context_parts = []
            for result in search_results:
                candidate = result['metadata']['candidate_name']
                text_snippet = result['text'][:max_context_length]
                context_parts.append(f"Candidate: {candidate}\n{text_snippet}")
            
            context = "\n\n".join(context_parts)
            
            # Truncate context if too long
            if len(context) > max_context_length * 3:
                context = context[:max_context_length * 3]
            
            # Generate answer using Q&A model
            try:
                result = self.qa_model(question=query, context=context)
                answer = result['answer']
                confidence = result.get('score', 0.0)
                
                # Enhance answer with context information
                enhanced_answer = f"Based on the resumes in the database:\n\n{answer}\n\n"
                enhanced_answer += f"(Confidence: {confidence:.2f}, Based on {len(search_results)} resume(s))"
                
                return enhanced_answer
                
            except Exception as e:
                logger.warning(f"Q&A model failed, falling back to context: {e}")
                return f"Based on the available resumes:\n\n{context[:500]}..."
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return f"Error processing query: {str(e)}"

class ResumeScreenerApp:
    """Main application class for the Resume Screener"""
    
    def __init__(self):
        logger.info("ResumeScreenerApp initialized")
        self.config = Config()
        
        # Initialize models as None - will be loaded in initialize_system
        self.embedding_model = None
        self.summarizer = None
        self.qa_model = None
        self.resume_index = None
        self.rag_pipeline = None
        
    def load_models(self):
        """Load all required AI models"""
        try:
            logger.info("Loading machine learning models...")
            
            # Sentence transformer for embeddings
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
            
            # Summarization model
            logger.info("Loading summarization model...")
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            logger.info("Summarization model loaded")
            
            # Question-answering model (using more reliable configuration)
            logger.info("Loading Q&A model...")
            try:
                self.qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
            except Exception as e:
                logger.warning(f"Primary Q&A model failed: {e}")
                logger.info("Trying alternative Q&A model...")
                self.qa_model = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
            logger.info("Q&A model loaded")
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def initialize_system(self):
        """Initialize the complete system"""
        logger.info("Initializing AI Resume Screener system...")
        
        # Load models
        self.load_models()
        
        # Initialize resume index
        self.resume_index = ResumeIndex(self.embedding_model)
        
        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline(self.resume_index, self.qa_model, self.summarizer)
        
        logger.info("System initialized successfully!")
    
    def upload_and_index_resume(self, file, candidate_name):
        """Handle resume upload and indexing"""
        try:
            if file is None:
                return "Please select a file to upload."
            
            if not candidate_name.strip():
                return "Please enter a candidate name."
            
            # Process the uploaded file
            text = PDFProcessor.extract_text_from_pdf(file.name)
            
            if not text.strip():
                return "Could not extract text from the PDF file."
            
            # Add to index
            self.resume_index.add_resume(text, candidate_name.strip(), file.name)
            
            return f"Successfully uploaded and indexed resume for {candidate_name}\n\nExtracted text preview:\n{text[:500]}..."
            
        except Exception as e:
            logger.error(f"Error in upload_and_index_resume: {e}")
            return f"Error: {str(e)}"

    def upload_and_index_multiple_resumes(self, files):
        """Handle multiple resume uploads and indexing"""
        try:
            if not files:
                return "Please select at least one PDF file to upload."
            
            results = []
            successful_uploads = 0
            failed_uploads = 0
            
            for file in files:
                try:
                    # Extract candidate name from filename (remove .pdf extension)
                    candidate_name = Path(file.name).stem
                    
                    # Process the uploaded file
                    text = PDFProcessor.extract_text_from_pdf(file.name)
                    
                    if not text.strip():
                        results.append(f"ERROR - {candidate_name}: Could not extract text from PDF")
                        failed_uploads += 1
                        continue
                    
                    # Add to index
                    self.resume_index.add_resume(text, candidate_name, file.name)
                    results.append(f"SUCCESS - {candidate_name}: Successfully uploaded and indexed")
                    successful_uploads += 1
                    
                except Exception as e:
                    candidate_name = Path(file.name).stem if hasattr(file, 'name') else "Unknown"
                    results.append(f"ERROR - {candidate_name}: Error - {str(e)}")
                    failed_uploads += 1
            
            summary = f"Upload Summary:\n- Successful: {successful_uploads}\n- Failed: {failed_uploads}\n\nDetails:\n"
            return summary + "\n".join(results)
            
        except Exception as e:
            logger.error(f"Error in upload_and_index_multiple_resumes: {e}")
            return f"Error: {str(e)}"
    
    def search_resumes(self, query, num_results):
        """Search resumes based on query"""
        try:
            if not query.strip():
                return "Please enter a search query."
            
            results = self.resume_index.search(query.strip(), k=int(num_results))
            
            if not results:
                return "No matching resumes found."
            
            output = f"Found {len(results)} matching resume(s):\n\n"
            
            for result in results:
                candidate = result['metadata']['candidate_name']
                similarity = result['similarity_score']
                preview = result['text'][:300]
                
                output += f"**{result['rank']}. {candidate}** (Similarity: {similarity:.3f})\n"
                output += f"Preview: {preview}...\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error in search_resumes: {e}")
            return f"Error: {str(e)}"

    def rank_candidates_for_position(self, job_description, required_skills, experience_level, num_results):
        """Rank candidates for a specific position"""
        try:
            if not job_description.strip():
                return "Please enter a job description."
            
            results = self.resume_index.search_for_position(
                job_description.strip(), 
                required_skills.strip(), 
                experience_level.strip(), 
                k=int(num_results)
            )
            
            if not results:
                return "No candidates found in the database."
            
            output = f"Top {len(results)} candidates for the position:\n\n"
            output += "**Ranking based on:**\n"
            output += f"- Job Description: {job_description[:100]}{'...' if len(job_description) > 100 else ''}\n"
            if required_skills.strip():
                output += f"- Required Skills: {required_skills}\n"
            if experience_level.strip():
                output += f"- Experience Level: {experience_level}\n"
            output += "\n" + "="*60 + "\n\n"
            
            for result in results:
                candidate = result['metadata']['candidate_name']
                combined_score = result['combined_score']
                skill_score = result['skill_match_score']
                experience_score = result['experience_score']
                
                # Extract key information from resume
                resume_text = result['text']
                preview = resume_text[:400]
                
                output += f"**{result['rank']}. {candidate}**\n"
                output += f"   • Overall Score: {combined_score:.2f}/1.00\n"
                output += f"   • Skill Match: {skill_score:.2f}/1.00 ({skill_score*100:.0f}%)\n"
                output += f"   • Experience Match: {experience_score:.2f}/1.00 ({experience_score*100:.0f}%)\n"
                output += f"   • Resume Preview: {preview}...\n\n"
                output += "-" * 40 + "\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error in rank_candidates_for_position: {e}")
            return f"Error: {str(e)}"
    
    def query_ai(self, question):
        """Process AI queries using RAG"""
        try:
            if not question.strip():
                return "Please enter a question."
            
            response = self.rag_pipeline.answer_query(question.strip())
            return f"AI Response:\n\n{response}"
            
        except Exception as e:
            logger.error(f"Error in query_ai: {e}")
            return f"Error: {str(e)}"
    
    def summarize_candidate(self, candidate_name):
        """Generate AI summary for a specific candidate"""
        try:
            if not candidate_name.strip():
                return "Please enter a candidate name."
            
            # Search for the candidate
            search_results = self.resume_index.search(candidate_name.strip(), k=1)
            
            if not search_results:
                return f"No resume found for candidate: {candidate_name}"
            
            candidate_text = search_results[0]['text']
            
            # Generate summary
            # Split text into chunks if too long
            max_chunk_length = 1000
            if len(candidate_text) > max_chunk_length:
                chunks = [candidate_text[i:i+max_chunk_length] 
                         for i in range(0, len(candidate_text), max_chunk_length)]
                summary_parts = []
                
                for chunk in chunks[:3]:  # Limit to first 3 chunks
                    try:
                        summary = self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                        summary_parts.append(summary[0]['summary_text'])
                    except:
                        continue
                
                final_summary = " ".join(summary_parts)
            else:
                try:
                    summary = self.summarizer(candidate_text, max_length=200, min_length=50, do_sample=False)
                    final_summary = summary[0]['summary_text']
                except:
                    final_summary = candidate_text[:500] + "..."
            
            return f"**Summary for {search_results[0]['metadata']['candidate_name']}:**\n\n{final_summary}"
            
        except Exception as e:
            logger.error(f"Error in summarize_candidate: {e}")
            return f"Error: {str(e)}"

    def create_interface(self):
        """Create and return the Gradio interface"""
        with gr.Blocks(title="AI Resume Screener", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# AI-Powered Resume Screener")
            gr.Markdown("Upload resumes, search candidates, and get AI-powered insights")
            
            with gr.Tab("Upload Resume"):
                with gr.Row():
                    with gr.Column():
                        upload_file = gr.File(
                            label="Select Resumes (PDF) - Multiple files supported",
                            file_types=[".pdf"],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("Upload & Index Resumes", variant="primary")
                    
                    with gr.Column():
                        upload_output = gr.Textbox(
                            label="Upload Status",
                            lines=10,
                            max_lines=15
                        )
                
                upload_btn.click(
                    fn=self.upload_and_index_multiple_resumes,
                    inputs=[upload_file],
                    outputs=upload_output
                )
            
            with gr.Tab("Search Resumes"):
                with gr.Row():
                    with gr.Column():
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter skills, experience, keywords..."
                        )
                        num_results = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Results"
                        )
                        search_btn = gr.Button("Search", variant="primary")
                    
                    with gr.Column():
                        search_output = gr.Textbox(
                            label="Search Results",
                            lines=15,
                            max_lines=20
                        )
                
                search_btn.click(
                    fn=self.search_resumes,
                    inputs=[search_query, num_results],
                    outputs=search_output
                )
            
            with gr.Tab("AI Query"):
                with gr.Row():
                    with gr.Column():
                        ai_question = gr.Textbox(
                            label="Ask AI",
                            placeholder="Who has Python experience? Which candidates are suitable for senior roles?"
                        )
                        query_btn = gr.Button("Ask AI", variant="primary")
                    
                    with gr.Column():
                        ai_output = gr.Textbox(
                            label="AI Response",
                            lines=15,
                            max_lines=20
                        )
                
                query_btn.click(
                    fn=self.query_ai,
                    inputs=ai_question,
                    outputs=ai_output
                )
            
            with gr.Tab("Candidate Summary"):
                with gr.Row():
                    with gr.Column():
                        summary_candidate = gr.Textbox(
                            label="Candidate Name",
                            placeholder="Enter candidate name for AI summary"
                        )
                        summary_btn = gr.Button("Generate Summary", variant="primary")
                    
                    with gr.Column():
                        summary_output = gr.Textbox(
                            label="AI Summary",
                            lines=15,
                            max_lines=20
                        )
                
                summary_btn.click(
                    fn=self.summarize_candidate,
                    inputs=summary_candidate,
                    outputs=summary_output
                )
            
            with gr.Tab("Position Ranking"):
                with gr.Row():
                    with gr.Column():
                        job_description = gr.Textbox(
                            label="Job Description",
                            placeholder="Enter the job description and responsibilities...",
                            lines=4
                        )
                        required_skills = gr.Textbox(
                            label="Required Skills (comma-separated)",
                            placeholder="Python, SQL, Machine Learning, React, etc."
                        )
                        experience_level = gr.Textbox(
                            label="Experience Level",
                            placeholder="Junior, Mid-level, Senior, Lead, etc."
                        )
                        ranking_num_results = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Number of Candidates to Rank"
                        )
                        ranking_btn = gr.Button("Rank Candidates", variant="primary")
                    
                    with gr.Column():
                        ranking_output = gr.Textbox(
                            label="Candidate Rankings",
                            lines=20,
                            max_lines=25
                        )
                
                ranking_btn.click(
                    fn=self.rank_candidates_for_position,
                    inputs=[job_description, required_skills, experience_level, ranking_num_results],
                    outputs=ranking_output
                )
            
            with gr.Tab("System Info"):
                gr.Markdown("""
                ## About AI Resume Screener
                
                This application uses advanced AI models for intelligent resume processing:
                
                - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
                - **Vector Search**: FAISS for similarity search
                - **Summarization**: facebook/bart-large-cnn
                - **Q&A**: distilbert-base-uncased-distilled-squad
                - **Interface**: Gradio web framework
                
                ### Features:
                - PDF resume upload and text extraction
                - Vector-based similarity search
                - RAG (Retrieval-Augmented Generation) for AI queries
                - Automatic candidate summarization
                - Real-time processing and indexing
                """)
        
        return interface
    
    def launch(self, share=False, debug=False):
        """Launch the application"""
        logger.info("Launching AI Resume Screener...")
        
        # Initialize system
        self.initialize_system()
        
        # Create interface
        interface = self.create_interface()
        
        # Launch
        try:
            logger.info(f"Starting server on {self.config.HOST}:{self.config.PORT}")
            interface.launch(
                server_name=self.config.HOST,
                server_port=self.config.PORT,
                share=share,
                debug=debug,
                show_error=True
            )
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.error(f"Error launching application: {e}")
            raise

def main():
    """Main function"""
    print("============================================================")
    print("AI-POWERED RESUME SCREENER")
    print("An Intelligent Candidate Evaluation System")
    print("============================================================")
    
    try:
        app = ResumeScreenerApp()
        app.launch(share=True, debug=False)  # Always share for public access
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Application failed to start: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()