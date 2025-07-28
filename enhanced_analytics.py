"""
Enhanced Analytics Module - Real Content Analysis
Creates meaningful visualizations from actual course content.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Analyzes course content to extract meaningful insights."""
    
    def __init__(self):
        """Initialize the content analyzer."""
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'can', 'could', 'should', 'would',
            'will', 'shall', 'may', 'might', 'must', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def extract_key_concepts(self, documents: List[str], min_frequency: int = 3) -> List[Tuple[str, int]]:
        """Extract key concepts from documents based on frequency and relevance."""
        
        # Combine all text
        all_text = ' '.join(documents).lower()
        
        # Extract potential concepts (2-4 word phrases)
        concept_patterns = [
            r'\b[A-Za-z]+(?:\s+[A-Za-z]+){1,3}\b',  # 2-4 word phrases
            r'\b[A-Za-z]+\s+(?:analysis|method|approach|strategy|principle|concept|theory)\b',
            r'\b(?:real\s+estate|cash\s+flow|net\s+present|return\s+on|cap\s+rate)\b'
        ]
        
        concept_counts = Counter()
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                # Clean and filter
                words = match.lower().split()
                if len(words) >= 2 and not any(word in self.stop_words for word in words):
                    concept_counts[match.title()] += 1
        
        # Filter by frequency and return top concepts
        return [(concept, count) for concept, count in concept_counts.most_common(20) 
                if count >= min_frequency]
    
    def calculate_concept_relationships(self, documents: List[str], concepts: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate relationships between concepts based on co-occurrence."""
        
        relationships = defaultdict(lambda: defaultdict(float))
        
        for doc in documents:
            doc_lower = doc.lower()
            doc_concepts = [concept for concept in concepts if concept.lower() in doc_lower]
            
            # Calculate co-occurrence within same document
            for i, concept1 in enumerate(doc_concepts):
                for concept2 in doc_concepts[i+1:]:
                    # Weight by inverse distance if we can find positions
                    relationships[concept1][concept2] += 1.0
                    relationships[concept2][concept1] += 1.0
        
        # Normalize relationships
        for concept1 in relationships:
            max_rel = max(relationships[concept1].values()) if relationships[concept1] else 1
            for concept2 in relationships[concept1]:
                relationships[concept1][concept2] /= max_rel
        
        return dict(relationships)
    
    def create_document_clusters(self, documents: List[str], doc_names: List[str]) -> Dict[str, Any]:
        """Create meaningful document clusters based on content similarity."""
        
        # Simple keyword-based clustering for real estate content
        clusters = {
            'Valuation & Analysis': [],
            'Investment & Finance': [],
            'Market & Strategy': [],
            'Legal & Regulatory': [],
            'Operations & Management': [],
            'Other': []
        }
        
        cluster_keywords = {
            'Valuation & Analysis': ['valuation', 'appraisal', 'dcf', 'analysis', 'worth', 'value'],
            'Investment & Finance': ['investment', 'finance', 'loan', 'mortgage', 'roi', 'return', 'cash flow'],
            'Market & Strategy': ['market', 'strategy', 'competition', 'demand', 'supply', 'trends'],
            'Legal & Regulatory': ['legal', 'law', 'regulation', 'compliance', 'contract', 'zoning'],
            'Operations & Management': ['management', 'operations', 'maintenance', 'tenant', 'lease']
        }
        
        for i, (doc, name) in enumerate(zip(documents, doc_names)):
            doc_lower = doc.lower()
            best_cluster = 'Other'
            best_score = 0
            
            for cluster_name, keywords in cluster_keywords.items():
                score = sum(1 for keyword in keywords if keyword in doc_lower)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_name
            
            clusters[best_cluster].append({
                'name': name,
                'index': i,
                'keyword_matches': best_score
            })
        
        return clusters
    
    def generate_content_summary(self, documents: List[str]) -> Dict[str, Any]:
        """Generate comprehensive content summary statistics."""
        
        if not documents:
            return {}
        
        total_words = sum(len(doc.split()) for doc in documents)
        total_chars = sum(len(doc) for doc in documents)
        
        # Extract topics using simple keyword analysis
        real_estate_topics = {
            'Property Valuation': ['valuation', 'appraisal', 'worth', 'price', 'value'],
            'Investment Analysis': ['investment', 'roi', 'return', 'profit', 'yield'],
            'Cash Flow': ['cash flow', 'income', 'expense', 'net operating'],
            'Financing': ['loan', 'mortgage', 'finance', 'leverage', 'debt'],
            'Market Analysis': ['market', 'comparable', 'comp', 'trends', 'demand']
        }
        
        topic_coverage = {}
        for topic, keywords in real_estate_topics.items():
            mentions = sum(sum(1 for keyword in keywords 
                             if keyword.lower() in doc.lower()) 
                          for doc in documents)
            topic_coverage[topic] = mentions
        
        return {
            'total_documents': len(documents),
            'total_words': total_words,
            'total_characters': total_chars,
            'avg_words_per_doc': total_words / len(documents) if documents else 0,
            'topic_coverage': topic_coverage,
            'document_lengths': [len(doc.split()) for doc in documents]
        }

class EnhancedVisualizationManager:
    """Creates meaningful visualizations from real course content."""
    
    def __init__(self):
        """Initialize visualization manager."""
        self.analyzer = ContentAnalyzer()
    
    def create_concept_network_data(self, documents: List[str], doc_names: List[str]) -> Dict[str, Any]:
        """Create data for interactive concept network visualization."""
        
        # Extract concepts
        concepts_with_freq = self.analyzer.extract_key_concepts(documents)
        concepts = [concept for concept, freq in concepts_with_freq[:15]]  # Top 15
        
        if not concepts:
            return {"nodes": [], "edges": [], "message": "No meaningful concepts found in documents"}
        
        # Calculate relationships
        relationships = self.analyzer.calculate_concept_relationships(documents, concepts)
        
        # Create nodes
        nodes = []
        for i, (concept, freq) in enumerate(concepts_with_freq[:15]):
            nodes.append({
                'id': i,
                'label': concept,
                'size': min(freq * 5, 50),  # Scale size by frequency
                'frequency': freq,
                'category': self._categorize_concept(concept)
            })
        
        # Create edges
        edges = []
        edge_id = 0
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                strength = relationships.get(concept1, {}).get(concept2, 0)
                if strength > 0.3:  # Only show strong relationships
                    edges.append({
                        'id': edge_id,
                        'source': i,
                        'target': j,
                        'weight': strength,
                        'label': f"{strength:.2f}"
                    })
                    edge_id += 1
        
        return {
            'nodes': nodes,
            'edges': edges,
            'total_concepts': len(concepts),
            'total_relationships': len(edges)
        }
    
    def create_document_similarity_data(self, documents: List[str], doc_names: List[str]) -> Dict[str, Any]:
        """Create meaningful document similarity visualization data."""
        
        if len(documents) < 2:
            return {"message": "Need at least 2 documents for similarity analysis"}
        
        # Get document clusters
        clusters = self.analyzer.create_document_clusters(documents, doc_names)
        
        # Create visualization data
        viz_data = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for cluster_idx, (cluster_name, docs) in enumerate(clusters.items()):
            if not docs:
                continue
                
            color = colors[cluster_idx % len(colors)]
            
            for doc in docs:
                # Create meaningful positions based on cluster and content
                base_x = cluster_idx * 2 + np.random.normal(0, 0.3)
                base_y = doc['keyword_matches'] + np.random.normal(0, 0.2)
                
                viz_data.append({
                    'x': base_x,
                    'y': base_y,
                    'document': doc['name'],
                    'cluster': cluster_name,
                    'keyword_matches': doc['keyword_matches'],
                    'color': color,
                    'size': min(doc['keyword_matches'] * 5 + 10, 30)
                })
        
        return {
            'data': viz_data,
            'clusters': list(clusters.keys()),
            'total_clustered': sum(len(docs) for docs in clusters.values())
        }
    
    def _categorize_concept(self, concept: str) -> str:
        """Categorize a concept for visualization."""
        concept_lower = concept.lower()
        
        if any(word in concept_lower for word in ['cash', 'flow', 'income', 'revenue']):
            return 'Financial'
        elif any(word in concept_lower for word in ['market', 'analysis', 'research']):
            return 'Analysis'
        elif any(word in concept_lower for word in ['property', 'real estate', 'building']):
            return 'Property'
        elif any(word in concept_lower for word in ['investment', 'return', 'roi']):
            return 'Investment'
        else:
            return 'General'
    
    def create_learning_pathway_data(self, documents: List[str], doc_names: List[str]) -> Dict[str, Any]:
        """Create suggested learning pathway based on document complexity and relationships."""
        
        if not documents:
            return {"message": "No documents available for pathway analysis"}
        
        # Analyze document complexity
        doc_complexity = []
        for i, (doc, name) in enumerate(zip(documents, doc_names)):
            words = doc.split()
            sentences = doc.split('.')
            
            # Simple complexity metrics
            avg_words_per_sentence = len(words) / max(len(sentences), 1)
            unique_words = len(set(word.lower() for word in words))
            
            # Advanced topic detection
            advanced_keywords = ['valuation', 'dcf', 'analysis', 'investment', 'leverage']
            basic_keywords = ['property', 'real estate', 'buy', 'sell', 'rent']
            
            advanced_score = sum(1 for keyword in advanced_keywords if keyword in doc.lower())
            basic_score = sum(1 for keyword in basic_keywords if keyword in doc.lower())
            
            complexity = (avg_words_per_sentence / 10) + (advanced_score * 2) - basic_score
            
            doc_complexity.append({
                'name': name,
                'complexity': max(complexity, 0),
                'word_count': len(words),
                'advanced_topics': advanced_score,
                'basic_topics': basic_score
            })
        
        # Sort by complexity for suggested pathway
        pathway = sorted(doc_complexity, key=lambda x: x['complexity'])
        
        return {
            'pathway': pathway,
            'total_documents': len(pathway),
            'complexity_range': (min(d['complexity'] for d in pathway), 
                               max(d['complexity'] for d in pathway))
        }