"""
synthesis_engine.py - Cross-Document Synthesis Engine

This module implements intelligent cross-document synthesis:
- Multi-document summarization with conflict detection
- Concept evolution tracking
- Knowledge merging and refinement
- Contradiction resolution

Author: SUM Development Team
License: Apache License 2.0
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter
import hashlib

from memory.semantic_memory import get_semantic_memory_engine
from memory.knowledge_graph import get_knowledge_graph_engine
from application.optimized_summarizer import summarize_text_universal
from Models.summarizer import SimpleSummarizer

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document for synthesis."""
    id: str
    text: str
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict]] = None
    embedding: Optional[List[float]] = None
    timestamp: Optional[float] = None


@dataclass
class SynthesisResult:
    """Result of document synthesis."""
    unified_summary: str
    key_insights: List[str]
    contradictions: List[Dict[str, Any]]
    consensus_points: List[str]
    concept_evolution: Dict[str, List[Dict]]
    confidence_score: float
    source_documents: List[str]
    synthesis_metadata: Dict[str, Any]


class CrossDocumentSynthesisEngine:
    """
    Engine for synthesizing knowledge across multiple documents.
    Handles contradiction detection, concept merging, and knowledge evolution.
    """
    
    def __init__(self):
        """Initialize the synthesis engine."""
        self.memory_engine = get_semantic_memory_engine()
        self.kg_engine = get_knowledge_graph_engine()
        self.summarizer = SimpleSummarizer()
        
        # Synthesis statistics
        self.stats = {
            'documents_processed': 0,
            'contradictions_found': 0,
            'concepts_merged': 0,
            'synthesis_operations': 0
        }
    
    def synthesize_documents(self, 
                           documents: List[Document],
                           synthesis_type: str = "comprehensive",
                           min_consensus: float = 0.6) -> SynthesisResult:
        """
        Synthesize knowledge from multiple documents.
        
        Args:
            documents: List of documents to synthesize
            synthesis_type: Type of synthesis (comprehensive, focused, comparative)
            min_consensus: Minimum agreement threshold for consensus
            
        Returns:
            SynthesisResult containing unified knowledge
        """
        start_time = time.time()
        
        # Prepare documents
        prepared_docs = self._prepare_documents(documents)
        
        # Extract and merge concepts
        concept_network = self._build_concept_network(prepared_docs)
        
        # Detect contradictions
        contradictions = self._detect_contradictions(prepared_docs, concept_network)
        
        # Find consensus points
        consensus_points = self._find_consensus(prepared_docs, min_consensus)
        
        # Track concept evolution
        concept_evolution = self._track_concept_evolution(prepared_docs, concept_network)
        
        # Generate unified summary based on synthesis type
        if synthesis_type == "comprehensive":
            unified_summary = self._comprehensive_synthesis(
                prepared_docs, concept_network, consensus_points
            )
        elif synthesis_type == "focused":
            unified_summary = self._focused_synthesis(
                prepared_docs, concept_network, consensus_points
            )
        else:  # comparative
            unified_summary = self._comparative_synthesis(
                prepared_docs, contradictions, consensus_points
            )
        
        # Extract key insights
        key_insights = self._extract_key_insights(
            concept_network, consensus_points, contradictions
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            len(consensus_points), len(contradictions), len(documents)
        )
        
        # Update statistics
        self.stats['documents_processed'] += len(documents)
        self.stats['contradictions_found'] += len(contradictions)
        self.stats['synthesis_operations'] += 1
        
        # Store synthesis in memory
        self._store_synthesis_memory(unified_summary, documents)
        
        return SynthesisResult(
            unified_summary=unified_summary,
            key_insights=key_insights,
            contradictions=contradictions,
            consensus_points=consensus_points,
            concept_evolution=concept_evolution,
            confidence_score=confidence_score,
            source_documents=[doc.id for doc in documents],
            synthesis_metadata={
                'synthesis_time': time.time() - start_time,
                'synthesis_type': synthesis_type,
                'document_count': len(documents),
                'timestamp': time.time()
            }
        )
    
    def _prepare_documents(self, documents: List[Document]) -> List[Document]:
        """Prepare documents for synthesis by generating summaries and embeddings."""
        prepared = []
        
        for doc in documents:
            # Generate summary if not provided
            if not doc.summary:
                summary_result = summarize_text_universal(doc.text)
                doc.summary = summary_result.get('medium', summary_result.get('summary', ''))
            
            # Generate embedding
            if not doc.embedding:
                doc.embedding = self.memory_engine.generate_embedding(doc.text)
            
            # Extract entities
            if not doc.entities:
                extraction = self.kg_engine.extract_entities_and_relationships(
                    doc.text, source=doc.id
                )
                doc.entities = extraction['entities']
            
            # Set timestamp if not provided
            if not doc.timestamp:
                doc.timestamp = time.time()
            
            prepared.append(doc)
        
        return prepared
    
    def _build_concept_network(self, documents: List[Document]) -> Dict[str, Any]:
        """Build a network of concepts across all documents."""
        concept_network = defaultdict(lambda: {
            'frequency': 0,
            'documents': set(),
            'contexts': [],
            'related_concepts': defaultdict(int)
        })
        
        # Process each document
        for doc in documents:
            # Extract concepts from entities
            for entity in doc.entities:
                concept = entity['name'].lower()
                concept_network[concept]['frequency'] += 1
                concept_network[concept]['documents'].add(doc.id)
                concept_network[concept]['contexts'].append({
                    'doc_id': doc.id,
                    'context': entity.get('properties', {}).get('context', '')
                })
            
            # Find co-occurring concepts
            entity_names = [e['name'].lower() for e in doc.entities]
            for i, concept1 in enumerate(entity_names):
                for concept2 in entity_names[i+1:]:
                    if concept1 != concept2:
                        concept_network[concept1]['related_concepts'][concept2] += 1
                        concept_network[concept2]['related_concepts'][concept1] += 1
        
        # Convert sets to lists for serialization
        for concept in concept_network:
            concept_network[concept]['documents'] = list(
                concept_network[concept]['documents']
            )
        
        return dict(concept_network)
    
    def _detect_contradictions(self, 
                             documents: List[Document],
                             concept_network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect contradictions between documents."""
        contradictions = []
        
        # Compare document pairs
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                # Calculate embedding similarity
                if doc1.embedding and doc2.embedding:
                    similarity = self._cosine_similarity(doc1.embedding, doc2.embedding)
                    
                    # Low similarity with shared concepts indicates potential contradiction
                    shared_concepts = self._find_shared_concepts(
                        doc1.entities, doc2.entities
                    )
                    
                    if shared_concepts and similarity < 0.3:
                        # Analyze specific contradictions
                        specific_contradictions = self._analyze_contradictions(
                            doc1, doc2, shared_concepts
                        )
                        
                        if specific_contradictions:
                            contradictions.append({
                                'doc1_id': doc1.id,
                                'doc2_id': doc2.id,
                                'similarity': float(similarity),
                                'shared_concepts': shared_concepts,
                                'details': specific_contradictions
                            })
        
        return contradictions
    
    def _find_consensus(self, 
                       documents: List[Document],
                       min_consensus: float) -> List[str]:
        """Find points of consensus across documents."""
        consensus_points = []
        
        # Extract key points from each document
        all_points = defaultdict(list)
        
        for doc in documents:
            # Use summary sentences as key points
            if doc.summary:
                sentences = doc.summary.split('. ')
                for sentence in sentences:
                    if len(sentence) > 20:  # Filter out very short sentences
                        # Generate embedding for sentence
                        sent_embedding = self.memory_engine.generate_embedding(sentence)
                        all_points[sentence].append({
                            'doc_id': doc.id,
                            'embedding': sent_embedding
                        })
        
        # Find similar points across documents
        consensus_threshold = int(len(documents) * min_consensus)
        
        for point, occurrences in all_points.items():
            if len(occurrences) >= consensus_threshold:
                consensus_points.append(point)
            else:
                # Check for semantic similarity even if not exact match
                similar_count = self._count_similar_points(
                    point, occurrences[0]['embedding'], all_points
                )
                if similar_count >= consensus_threshold:
                    consensus_points.append(point)
        
        return consensus_points
    
    def _track_concept_evolution(self, 
                               documents: List[Document],
                               concept_network: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Track how concepts evolve across documents over time."""
        concept_evolution = defaultdict(list)
        
        # Sort documents by timestamp
        sorted_docs = sorted(documents, key=lambda d: d.timestamp or 0)
        
        # Track each concept's evolution
        for concept, data in concept_network.items():
            if len(data['documents']) > 1:
                # Build evolution timeline
                for doc in sorted_docs:
                    if doc.id in data['documents']:
                        # Find concept context in this document
                        context = next(
                            (c['context'] for c in data['contexts'] 
                             if c['doc_id'] == doc.id),
                            ''
                        )
                        
                        concept_evolution[concept].append({
                            'doc_id': doc.id,
                            'timestamp': doc.timestamp,
                            'context': context,
                            'related_concepts': self._get_doc_related_concepts(
                                doc, concept, concept_network
                            )
                        })
        
        return dict(concept_evolution)
    
    def _comprehensive_synthesis(self,
                               documents: List[Document],
                               concept_network: Dict[str, Any],
                               consensus_points: List[str]) -> str:
        """Generate a comprehensive synthesis of all documents."""
        # Combine all text for overall summary
        combined_text = "\n\n".join([doc.summary or doc.text[:1000] for doc in documents])
        
        # Generate base summary
        base_summary = self.summarizer.summarize(combined_text, num_sentences=10)
        
        # Enhance with key concepts
        top_concepts = self._get_top_concepts(concept_network, limit=10)
        
        # Build comprehensive summary
        parts = [
            "Comprehensive synthesis of analyzed documents:",
            "",
            base_summary,
            "",
            "Key concepts identified across documents:",
            ". ".join([f"{concept} (appears in {data['frequency']} contexts)" 
                      for concept, data in top_concepts]),
            ""
        ]
        
        if consensus_points:
            parts.extend([
                "Points of consensus:",
                ". ".join(consensus_points[:5]),
                ""
            ])
        
        return "\n".join(parts)
    
    def _focused_synthesis(self,
                          documents: List[Document],
                          concept_network: Dict[str, Any],
                          consensus_points: List[str]) -> str:
        """Generate a focused synthesis highlighting key insights."""
        # Focus on most important concepts
        top_concepts = self._get_top_concepts(concept_network, limit=5)
        
        # Build focused summary around key concepts
        focused_parts = []
        
        for concept, data in top_concepts:
            # Get contexts for this concept
            contexts = [c['context'] for c in data['contexts'][:3]]
            if contexts:
                focused_parts.append(
                    f"{concept.title()}: " + 
                    self.summarizer.summarize(" ".join(contexts), num_sentences=2)
                )
        
        return "Focused synthesis on key concepts:\n\n" + "\n\n".join(focused_parts)
    
    def _comparative_synthesis(self,
                             documents: List[Document],
                             contradictions: List[Dict],
                             consensus_points: List[str]) -> str:
        """Generate a comparative synthesis highlighting differences and similarities."""
        parts = ["Comparative analysis of documents:"]
        
        # Similarities section
        if consensus_points:
            parts.extend([
                "",
                "Shared insights:",
                ". ".join(consensus_points[:5])
            ])
        
        # Differences section
        if contradictions:
            parts.extend([
                "",
                "Conflicting perspectives:"
            ])
            
            for cont in contradictions[:3]:
                parts.append(
                    f"- Documents differ on: {', '.join(cont['shared_concepts'][:3])}"
                )
        
        # Document-specific insights
        parts.extend(["", "Document-specific contributions:"])
        
        for doc in documents[:5]:
            if doc.summary:
                first_sentence = doc.summary.split('.')[0]
                parts.append(f"- Document {doc.id[:8]}: {first_sentence}")
        
        return "\n".join(parts)
    
    def _extract_key_insights(self,
                            concept_network: Dict[str, Any],
                            consensus_points: List[str],
                            contradictions: List[Dict]) -> List[str]:
        """Extract key insights from the synthesis."""
        insights = []
        
        # Insight 1: Most connected concepts
        most_connected = self._find_most_connected_concepts(concept_network, limit=3)
        if most_connected:
            insights.append(
                f"Central themes revolve around: {', '.join(most_connected)}"
            )
        
        # Insight 2: Consensus strength
        if consensus_points:
            insights.append(
                f"Strong agreement found on {len(consensus_points)} key points"
            )
        
        # Insight 3: Contradiction areas
        if contradictions:
            conflict_areas = set()
            for cont in contradictions:
                conflict_areas.update(cont['shared_concepts'][:2])
            insights.append(
                f"Divergent views on: {', '.join(list(conflict_areas)[:5])}"
            )
        
        # Insight 4: Concept evolution
        evolving_concepts = [c for c, evolution in concept_network.items() 
                           if len(evolution.get('documents', [])) > 2]
        if evolving_concepts:
            insights.append(
                f"Concepts showing evolution: {', '.join(evolving_concepts[:3])}"
            )
        
        return insights
    
    def _calculate_confidence(self,
                            consensus_count: int,
                            contradiction_count: int,
                            document_count: int) -> float:
        """Calculate confidence score for the synthesis."""
        if document_count == 0:
            return 0.0
        
        # Base confidence on consensus/contradiction ratio
        if contradiction_count == 0:
            contradiction_factor = 1.0
        else:
            contradiction_factor = consensus_count / (consensus_count + contradiction_count)
        
        # Factor in document count (more documents = higher confidence)
        document_factor = min(1.0, document_count / 10.0)
        
        # Combined confidence
        confidence = (contradiction_factor * 0.7 + document_factor * 0.3)
        
        return round(confidence, 2)
    
    def _store_synthesis_memory(self, synthesis: str, documents: List[Document]):
        """Store the synthesis result in semantic memory."""
        # Create metadata
        metadata = {
            'type': 'synthesis',
            'source_documents': [doc.id for doc in documents],
            'timestamp': time.time(),
            'document_count': len(documents)
        }
        
        # Store in memory
        memory_id = self.memory_engine.store_memory(
            text=synthesis,
            summary=synthesis[:200] + "..." if len(synthesis) > 200 else synthesis,
            metadata=metadata,
            relationships=[doc.id for doc in documents if doc.id]
        )
        
        logger.info(f"Stored synthesis as memory: {memory_id}")
    
    # Helper methods
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _find_shared_concepts(self, 
                            entities1: List[Dict],
                            entities2: List[Dict]) -> List[str]:
        """Find concepts shared between two entity lists."""
        concepts1 = {e['name'].lower() for e in entities1}
        concepts2 = {e['name'].lower() for e in entities2}
        return list(concepts1.intersection(concepts2))
    
    def _analyze_contradictions(self,
                              doc1: Document,
                              doc2: Document,
                              shared_concepts: List[str]) -> List[Dict]:
        """Analyze specific contradictions between documents."""
        contradictions = []
        
        # This is a simplified implementation
        # In a real system, you'd use more sophisticated NLP
        for concept in shared_concepts[:5]:
            # Find sentences mentioning the concept
            doc1_contexts = self._find_concept_contexts(doc1.text, concept)
            doc2_contexts = self._find_concept_contexts(doc2.text, concept)
            
            if doc1_contexts and doc2_contexts:
                # Compare sentiments or assertions
                contradictions.append({
                    'concept': concept,
                    'doc1_context': doc1_contexts[0][:100],
                    'doc2_context': doc2_contexts[0][:100]
                })
        
        return contradictions
    
    def _find_concept_contexts(self, text: str, concept: str) -> List[str]:
        """Find sentences containing a concept."""
        sentences = text.split('. ')
        contexts = []
        
        for sentence in sentences:
            if concept.lower() in sentence.lower():
                contexts.append(sentence)
        
        return contexts
    
    def _count_similar_points(self,
                            point: str,
                            point_embedding: List[float],
                            all_points: Dict) -> int:
        """Count how many similar points exist across documents."""
        similar_count = 0
        threshold = 0.8
        
        for other_point, occurrences in all_points.items():
            if other_point != point:
                for occ in occurrences:
                    similarity = self._cosine_similarity(
                        point_embedding,
                        occ['embedding']
                    )
                    if similarity > threshold:
                        similar_count += 1
                        break
        
        return similar_count
    
    def _get_top_concepts(self, 
                         concept_network: Dict[str, Any],
                         limit: int = 10) -> List[Tuple[str, Dict]]:
        """Get top concepts by frequency and connectivity."""
        # Score concepts by frequency and connections
        concept_scores = {}
        
        for concept, data in concept_network.items():
            frequency_score = data['frequency']
            connection_score = len(data['related_concepts'])
            concept_scores[concept] = frequency_score + (connection_score * 0.5)
        
        # Sort by score
        sorted_concepts = sorted(
            concept_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top concepts with their data
        return [(concept, concept_network[concept]) 
                for concept, _ in sorted_concepts[:limit]]
    
    def _find_most_connected_concepts(self,
                                    concept_network: Dict[str, Any],
                                    limit: int = 5) -> List[str]:
        """Find concepts with most connections."""
        connection_counts = {
            concept: len(data['related_concepts'])
            for concept, data in concept_network.items()
        }
        
        sorted_concepts = sorted(
            connection_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [concept for concept, _ in sorted_concepts[:limit]]
    
    def _get_doc_related_concepts(self,
                                doc: Document,
                                target_concept: str,
                                concept_network: Dict[str, Any]) -> List[str]:
        """Get concepts related to target concept in a specific document."""
        doc_concepts = {e['name'].lower() for e in doc.entities}
        
        if target_concept in concept_network:
            related = concept_network[target_concept]['related_concepts']
            # Filter to only concepts in this document
            return [c for c in related if c in doc_concepts][:5]
        
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis engine statistics."""
        return dict(self.stats)


# Global instance
_synthesis_engine = None


def get_synthesis_engine() -> CrossDocumentSynthesisEngine:
    """Get or create the global synthesis engine."""
    global _synthesis_engine
    if _synthesis_engine is None:
        _synthesis_engine = CrossDocumentSynthesisEngine()
    return _synthesis_engine


# Convenience functions

def synthesize_texts(texts: List[str], 
                    synthesis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Convenience function to synthesize multiple text strings.
    
    Args:
        texts: List of text strings
        synthesis_type: Type of synthesis
        
    Returns:
        Synthesis results as dictionary
    """
    engine = get_synthesis_engine()
    
    # Create Document objects
    documents = []
    for i, text in enumerate(texts):
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        documents.append(Document(
            id=doc_id,
            text=text,
            metadata={'index': i}
        ))
    
    # Synthesize
    result = engine.synthesize_documents(documents, synthesis_type)
    
    # Convert to dictionary
    return {
        'unified_summary': result.unified_summary,
        'key_insights': result.key_insights,
        'contradictions': result.contradictions,
        'consensus_points': result.consensus_points,
        'confidence_score': result.confidence_score,
        'metadata': result.synthesis_metadata
    }


if __name__ == "__main__":
    # Example usage
    texts = [
        "Artificial intelligence is revolutionizing healthcare through early disease detection.",
        "AI in healthcare raises concerns about patient privacy and data security.",
        "Machine learning algorithms can predict diseases before symptoms appear."
    ]
    
    result = synthesize_texts(texts, synthesis_type="comparative")
    
    print("Synthesis Result:")
    print(f"Summary: {result['unified_summary']}")
    print(f"Confidence: {result['confidence_score']}")
    print(f"Key Insights: {result['key_insights']}")