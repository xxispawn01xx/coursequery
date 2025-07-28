"""
Query engine for processing questions against course content using local models.
Combines LlamaIndex retrieval with local Mistral 7B generation.
"""

import logging
from typing import Dict, Any, List, Optional
import re

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

from config import Config
from local_models import LocalModelManager
from course_indexer import CourseIndexer

logger = logging.getLogger(__name__)

class LocalQueryEngine:
    """Query engine using local models for course question answering."""
    
    def __init__(self, model_manager: LocalModelManager):
        """
        Initialize the query engine.
        
        Args:
            model_manager: Local model manager instance
        """
        self.config = Config()
        self.model_manager = model_manager
        self.course_indexer = CourseIndexer()
        self.course_indexer.set_model_manager(model_manager)
        
        # Using local generation only - no external response synthesizer needed
    
    def query(self, 
              query: str, 
              course_name: str, 
              max_results: int = 5, 
              include_sources: bool = True) -> Dict[str, Any]:
        """
        Process a query against a specific course.
        
        Args:
            query: User question
            course_name: Name of the course to query
            max_results: Maximum number of relevant chunks to retrieve
            include_sources: Whether to include source references
            
        Returns:
            Dictionary containing answer and optional sources
        """
        logger.info(f"Processing query for course '{course_name}': {query[:100]}...")
        
        try:
            # Get course index
            index = self.course_indexer.get_course_index(course_name)
            
            if index is None:
                return {
                    'answer': f"Course '{course_name}' not found. Please make sure the course is indexed.",
                    'sources': [],
                    'error': 'Course not found'
                }
            
            # Configure retriever with syllabus weighting
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=max_results,
            )
            
            # Configure postprocessor for similarity filtering
            node_postprocessors = [
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
            
            # Retrieve relevant documents
            retrieved_nodes = retriever.retrieve(query)
            
            # Apply syllabus weighting and re-rank
            weighted_nodes = self._apply_syllabus_weighting(retrieved_nodes)
            
            # Generate context from top nodes
            context = self._build_context(weighted_nodes[:max_results])
            
            # Generate response using local Mistral model
            response = self._generate_local_response(query, context, course_name)
            
            # Prepare sources if requested
            sources = []
            if include_sources:
                sources = self._extract_sources(weighted_nodes[:max_results])
            
            result = {
                'answer': response,
                'sources': sources,
                'context_length': len(context),
                'nodes_retrieved': len(retrieved_nodes)
            }
            
            logger.info(f"Query processed successfully for course '{course_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'error': str(e)
            }
    
    def _apply_syllabus_weighting(self, nodes: List[Any]) -> List[Any]:
        """
        Apply syllabus weighting to retrieved nodes and re-rank them.
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            Re-ranked list of nodes with syllabus weighting applied
        """
        weighted_nodes = []
        
        for node in nodes:
            # Get syllabus weight from metadata
            syllabus_weight = node.metadata.get('syllabus_weight', 1.0)
            
            # Apply weight to the similarity score
            original_score = getattr(node, 'score', 0.0)
            weighted_score = original_score * syllabus_weight
            
            # Store the weighted score
            node.weighted_score = weighted_score
            weighted_nodes.append(node)
        
        # Sort by weighted score (descending)
        weighted_nodes.sort(key=lambda x: getattr(x, 'weighted_score', 0.0), reverse=True)
        
        return weighted_nodes
    
    def _build_context(self, nodes: List[Any]) -> str:
        """
        Build context string from retrieved nodes.
        
        Args:
            nodes: List of nodes to build context from
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, node in enumerate(nodes, 1):
            file_name = node.metadata.get('file_name', 'Unknown')
            is_syllabus = node.metadata.get('is_syllabus', False)
            
            # Add priority indicator for syllabus content
            priority_indicator = " [SYLLABUS]" if is_syllabus else ""
            
            context_part = f"[Source {i}: {file_name}{priority_indicator}]\n{node.text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_local_response(self, query: str, context: str, course_name: str) -> str:
        """
        Generate response using local Mistral model.
        
        Args:
            query: User question
            context: Retrieved context
            course_name: Name of the course
            
        Returns:
            Generated response
        """
        # Build comprehensive prompt
        prompt = self._build_response_prompt(query, context, course_name)
        
        try:
            # Generate response using local model
            response = self.model_manager.generate_response(
                prompt, 
                max_new_tokens=512
            )
            
            # Clean and format response
            cleaned_response = self._clean_and_format_response(response)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error generating local response: {e}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def _build_response_prompt(self, query: str, context: str, course_name: str) -> str:
        """Build the prompt for response generation."""
        prompt = f"""You are an expert real estate education assistant. Answer the following question based on the provided course context from "{course_name}".

IMPORTANT INSTRUCTIONS:
- Provide accurate, detailed answers based ONLY on the given context
- Give priority to information marked as [SYLLABUS] as it represents key course structure
- Be specific and practical in your responses
- If the context doesn't contain enough information to answer fully, say so
- Include relevant examples when available in the context
- Structure your answer clearly with main points

CONTEXT FROM COURSE MATERIALS:
{context}

STUDENT QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def _clean_and_format_response(self, response: str) -> str:
        """Clean and format the generated response."""
        # Remove any instruction leakage
        response = re.sub(r'\[INST\].*?\[/INST\]', '', response, flags=re.DOTALL)
        response = re.sub(r'<s>|</s>', '', response)
        
        # Clean up extra whitespace and line breaks
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = response.strip()
        
        # Ensure the response doesn't start with unwanted prefixes
        unwanted_prefixes = ['ANSWER:', 'Response:', 'Answer:']
        for prefix in unwanted_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response
    
    def _extract_sources(self, nodes: List[Any]) -> List[str]:
        """
        Extract source information from nodes.
        
        Args:
            nodes: List of nodes to extract sources from
            
        Returns:
            List of source strings
        """
        sources = []
        
        for node in nodes:
            file_name = node.metadata.get('file_name', 'Unknown source')
            file_type = node.metadata.get('file_type', '')
            is_syllabus = node.metadata.get('is_syllabus', False)
            
            # Build source string
            source_str = f"{file_name}"
            
            if file_type:
                source_str += f" ({file_type.upper()})"
            
            if is_syllabus:
                source_str += " [Syllabus]"
            
            # Add similarity score if available
            if hasattr(node, 'weighted_score'):
                score = getattr(node, 'weighted_score', 0.0)
                source_str += f" (Relevance: {score:.2f})"
            
            sources.append(source_str)
        
        return sources
    
    def get_course_summary(self, course_name: str) -> Dict[str, Any]:
        """
        Generate a summary of course content.
        
        Args:
            course_name: Name of the course
            
        Returns:
            Course summary information
        """
        try:
            # Get course analytics
            analytics = self.course_indexer.get_course_analytics(course_name)
            
            if not analytics:
                return {'error': f'Course {course_name} not found'}
            
            # Generate a summary query
            summary_query = "What are the main topics and learning objectives covered in this course?"
            
            # Get response
            result = self.query(
                query=summary_query,
                course_name=course_name,
                max_results=10,
                include_sources=False
            )
            
            return {
                'course_name': course_name,
                'summary': result['answer'],
                'total_documents': analytics.get('total_documents', 0),
                'syllabus_documents': analytics.get('syllabus_documents', 0),
                'last_indexed': analytics.get('last_indexed', 'Unknown'),
            }
            
        except Exception as e:
            logger.error(f"Error generating course summary: {e}")
            return {'error': str(e)}
