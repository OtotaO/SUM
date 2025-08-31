"""
GraphRAG-based Knowledge Crystallization Engine
Implements Microsoft Research's GraphRAG approach for corpus-level summarization
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
import asyncio
import logging
from llm_backend import llm_backend

logger = logging.getLogger(__name__)

# Safe imports with fallbacks
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    logger.warning("python-louvain not installed. Install with: pip install python-louvain")

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logger.warning("spacy not installed. Install with: pip install spacy")


@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    id: str
    text: str
    type: str
    mentions: List[int]
    attributes: Dict[str, Any]


@dataclass
class Relation:
    """Represents a relation between entities"""
    source: str
    target: str
    type: str
    weight: float
    context: str


@dataclass
class Community:
    """Represents a community of related entities"""
    id: int
    entities: List[Entity]
    relations: List[Relation]
    summary: str
    level: int
    parent: Optional[int] = None


@dataclass
class GraphRAGResult:
    """Result of GraphRAG crystallization"""
    graph: nx.Graph
    communities: List[Community]
    hierarchical_summaries: Dict[int, str]
    global_summary: str
    entity_summaries: Dict[str, str]
    metadata: Dict[str, Any]


class KnowledgeGraphBuilder:
    """Builds knowledge graphs from text documents"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        if HAS_TRANSFORMERS:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedder = None
        
    def extract_entities_and_relations(self, documents: List[str]) -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from documents"""
        entities = {}
        relations = []
        
        if not self.nlp:
            # Fallback to simple entity extraction
            return self._simple_entity_extraction(documents)
        
        for doc_idx, text in enumerate(documents):
            doc = self.nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                entity_id = self._generate_entity_id(ent.text, ent.label_)
                
                if entity_id not in entities:
                    entities[entity_id] = Entity(
                        id=entity_id,
                        text=ent.text,
                        type=ent.label_,
                        mentions=[doc_idx],
                        attributes={'frequency': 1}
                    )
                else:
                    entities[entity_id].mentions.append(doc_idx)
                    entities[entity_id].attributes['frequency'] += 1
            
            # Extract relations through dependency parsing
            for token in doc:
                if token.dep_ in ("nsubj", "dobj", "pobj"):
                    if token.ent_type_ and token.head.ent_type_:
                        source_id = self._generate_entity_id(token.text, token.ent_type_)
                        target_id = self._generate_entity_id(token.head.text, token.head.ent_type_)
                        
                        if source_id in entities and target_id in entities:
                            relations.append(Relation(
                                source=source_id,
                                target=target_id,
                                type=token.dep_,
                                weight=1.0,
                                context=doc.text[max(0, token.idx-50):min(len(doc.text), token.idx+50)]
                            ))
        
        return list(entities.values()), relations
    
    def _simple_entity_extraction(self, documents: List[str]) -> Tuple[List[Entity], List[Relation]]:
        """Simple fallback entity extraction when spacy is not available"""
        entities = {}
        relations = []
        
        for doc_idx, text in enumerate(documents):
            # Extract capitalized words as potential entities
            words = text.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 1:
                    entity_id = self._generate_entity_id(word, "ENTITY")
                    if entity_id not in entities:
                        entities[entity_id] = Entity(
                            id=entity_id,
                            text=word,
                            type="ENTITY",
                            mentions=[doc_idx],
                            attributes={'frequency': 1}
                        )
                    else:
                        entities[entity_id].mentions.append(doc_idx)
        
        return list(entities.values()), relations
    
    def _generate_entity_id(self, text: str, entity_type: str) -> str:
        """Generate unique ID for entity"""
        return hashlib.md5(f"{text.lower()}:{entity_type}".encode()).hexdigest()[:16]
    
    def build_graph(self, entities: List[Entity], relations: List[Relation]) -> nx.Graph:
        """Build NetworkX graph from entities and relations"""
        G = nx.Graph()
        
        # Add nodes
        for entity in entities:
            G.add_node(entity.id, 
                      text=entity.text,
                      type=entity.type,
                      frequency=entity.attributes['frequency'])
        
        # Add edges
        for relation in relations:
            if G.has_edge(relation.source, relation.target):
                G[relation.source][relation.target]['weight'] += relation.weight
            else:
                G.add_edge(relation.source, relation.target, 
                          weight=relation.weight,
                          type=relation.type)
        
        return G


class LeidenCommunityDetector:
    """Detect communities using Leiden algorithm (fallback to Louvain)"""
    
    def detect(self, graph: nx.Graph, resolution: float = 1.0) -> Dict[str, int]:
        """Detect communities in graph"""
        # Using Louvain as fallback (Leiden requires additional installation)
        partition = community_louvain.best_partition(graph, resolution=resolution)
        return partition
    
    def hierarchical_detection(self, graph: nx.Graph) -> List[Dict[str, int]]:
        """Detect communities at multiple resolutions for hierarchy"""
        resolutions = [0.5, 1.0, 1.5, 2.0]
        hierarchies = []
        
        for resolution in resolutions:
            partition = self.detect(graph, resolution)
            hierarchies.append(partition)
        
        return hierarchies


class HierarchicalSummarizer:
    """Generate hierarchical summaries of communities"""
    
    def __init__(self):
        if HAS_TRANSFORMERS:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedder = None
        
    def summarize_communities(self, 
                            graph: nx.Graph, 
                            partition: Dict[str, int],
                            entities: List[Entity]) -> List[Community]:
        """Generate summaries for each community"""
        communities = defaultdict(list)
        
        # Group entities by community
        entity_lookup = {e.id: e for e in entities}
        for node_id, community_id in partition.items():
            if node_id in entity_lookup:
                communities[community_id].append(entity_lookup[node_id])
        
        # Generate summaries
        community_objects = []
        for comm_id, comm_entities in communities.items():
            # Extract subgraph for this community
            node_ids = [e.id for e in comm_entities]
            subgraph = graph.subgraph(node_ids)
            
            # Generate summary based on entities and relations
            summary = self._generate_community_summary(comm_entities, subgraph)
            
            community_objects.append(Community(
                id=comm_id,
                entities=comm_entities,
                relations=self._extract_relations(subgraph),
                summary=summary,
                level=0
            ))
        
        return community_objects
    
    def _generate_community_summary(self, entities: List[Entity], subgraph: nx.Graph) -> str:
        """Generate natural language summary of community"""
        # Get most important entities (by degree centrality)
        if len(subgraph.nodes) > 0:
            centrality = nx.degree_centrality(subgraph)
            top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            top_entity_texts = [entities[i].text for i, _ in enumerate(entities) 
                               if entities[i].id in [e[0] for e in top_entities]]
        else:
            top_entity_texts = [e.text for e in entities[:3]]
        
        # Get entity types
        entity_types = list(set(e.type for e in entities))
        
        # Generate summary
        if len(entities) == 1:
            summary = f"{entities[0].text} ({entities[0].type})"
        elif len(top_entity_texts) > 0:
            summary = f"Community centered around {', '.join(top_entity_texts)}"
            if entity_types:
                summary += f" involving {', '.join(entity_types[:3])}"
        else:
            summary = f"Community of {len(entities)} entities"
        
        return summary
    
    def _extract_relations(self, subgraph: nx.Graph) -> List[Relation]:
        """Extract relations from subgraph"""
        relations = []
        for source, target, data in subgraph.edges(data=True):
            relations.append(Relation(
                source=source,
                target=target,
                type=data.get('type', 'related'),
                weight=data.get('weight', 1.0),
                context=""
            ))
        return relations
    
    def build_hierarchy(self, communities_by_level: List[List[Community]]) -> Dict[int, str]:
        """Build hierarchical summaries from multiple community levels"""
        hierarchical_summaries = {}
        
        for level_idx, communities in enumerate(communities_by_level):
            # Aggregate summaries at this level
            level_summary = self._aggregate_summaries(
                [c.summary for c in communities]
            )
            hierarchical_summaries[level_idx] = level_summary
        
        return hierarchical_summaries
    
    def _aggregate_summaries(self, summaries: List[str]) -> str:
        """Aggregate multiple summaries into coherent text"""
        if len(summaries) == 0:
            return ""
        elif len(summaries) == 1:
            return summaries[0]
        elif len(summaries) <= 3:
            return " ".join(summaries)
        else:
            # Group similar summaries
            embeddings = self.embedder.encode(summaries)
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings)
            
            # Generate summary for each cluster
            cluster_summaries = []
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:  # Not noise
                    cluster_texts = [summaries[i] for i, label in enumerate(clustering.labels_) 
                                   if label == cluster_id]
                    cluster_summaries.append(self._merge_texts(cluster_texts))
            
            # Add unclustered summaries
            noise_summaries = [summaries[i] for i, label in enumerate(clustering.labels_) 
                             if label == -1]
            cluster_summaries.extend(noise_summaries[:2])  # Limit noise
            
            return " ".join(cluster_summaries)
    
    def _merge_texts(self, texts: List[str]) -> str:
        """Merge similar texts into one"""
        if len(texts) == 1:
            return texts[0]
        
        # Find common patterns
        common_words = set(texts[0].split())
        for text in texts[1:]:
            common_words &= set(text.split())
        
        if common_words:
            return f"Multiple aspects involving {', '.join(list(common_words)[:3])}"
        else:
            return texts[0]  # Return first as representative


class GraphRAGCrystallizer:
    """Main GraphRAG crystallization engine"""
    
    def __init__(self):
        self.graph_builder = KnowledgeGraphBuilder()
        self.community_detector = LeidenCommunityDetector()
        self.hierarchical_summarizer = HierarchicalSummarizer()
    
    def crystallize_corpus(self, 
                          documents: List[str],
                          query: Optional[str] = None) -> GraphRAGResult:
        """
        Crystallize knowledge from document corpus using GraphRAG
        
        Args:
            documents: List of documents to process
            query: Optional query to focus summarization
            
        Returns:
            GraphRAGResult with graph, communities, and summaries
        """
        # Extract entities and relations
        entities, relations = self.graph_builder.extract_entities_and_relations(documents)
        
        # Build knowledge graph
        graph = self.graph_builder.build_graph(entities, relations)
        
        # Detect communities at multiple levels
        hierarchies = self.community_detector.hierarchical_detection(graph)
        
        # Generate community summaries
        all_communities = []
        communities_by_level = []
        
        for level_idx, partition in enumerate(hierarchies):
            communities = self.hierarchical_summarizer.summarize_communities(
                graph, partition, entities
            )
            for community in communities:
                community.level = level_idx
            
            all_communities.extend(communities)
            communities_by_level.append(communities)
        
        # Build hierarchical summaries
        hierarchical_summaries = self.hierarchical_summarizer.build_hierarchy(
            communities_by_level
        )
        
        # Generate global summary
        global_summary = self._generate_global_summary(
            hierarchical_summaries, 
            query
        )
        
        # Generate entity-level summaries
        entity_summaries = self._generate_entity_summaries(entities, graph)
        
        # Compile metadata
        metadata = {
            'num_documents': len(documents),
            'num_entities': len(entities),
            'num_relations': len(relations),
            'num_communities': len(all_communities),
            'graph_density': nx.density(graph) if len(graph.nodes) > 0 else 0,
            'levels': len(hierarchies)
        }
        
        return GraphRAGResult(
            graph=graph,
            communities=all_communities,
            hierarchical_summaries=hierarchical_summaries,
            global_summary=global_summary,
            entity_summaries=entity_summaries,
            metadata=metadata
        )
    
    def _generate_global_summary(self, 
                                hierarchical_summaries: Dict[int, str],
                                query: Optional[str] = None) -> str:
        """Generate global summary from hierarchical summaries"""
        if not hierarchical_summaries:
            return "No significant patterns found in corpus."
        
        # Get highest level summary
        highest_level = max(hierarchical_summaries.keys())
        base_summary = hierarchical_summaries[highest_level]
        
        if query:
            # TODO: Focus summary based on query
            return f"Regarding '{query}': {base_summary}"
        else:
            return f"Key insights: {base_summary}"
    
    def _generate_entity_summaries(self, 
                                  entities: List[Entity],
                                  graph: nx.Graph) -> Dict[str, str]:
        """Generate summaries for individual entities"""
        entity_summaries = {}
        
        for entity in entities[:100]:  # Limit for performance
            if entity.id in graph:
                degree = graph.degree(entity.id)
                neighbors = list(graph.neighbors(entity.id))
                
                summary = f"{entity.text} ({entity.type})"
                summary += f" appears {entity.attributes['frequency']} times"
                summary += f" with {degree} connections"
                
                entity_summaries[entity.id] = summary
        
        return entity_summaries
    
    def answer_global_question(self, 
                              result: GraphRAGResult,
                              question: str) -> str:
        """
        Answer global questions about the corpus
        
        Examples:
        - "What are the main themes?"
        - "What are the key relationships?"
        - "What patterns emerge?"
        """
        question_lower = question.lower()
        
        if "theme" in question_lower or "topic" in question_lower:
            # Return community-based themes
            themes = []
            for community in result.communities[:5]:  # Top communities
                if len(community.entities) > 2:
                    themes.append(community.summary)
            
            if themes:
                return f"Main themes: {'; '.join(themes)}"
            else:
                return "No clear themes identified."
        
        elif "relationship" in question_lower or "connection" in question_lower:
            # Analyze graph structure
            if result.graph.number_of_edges() > 0:
                # Get most connected nodes
                centrality = nx.degree_centrality(result.graph)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                
                connections = []
                for node_id, _ in top_nodes:
                    if node_id in result.entity_summaries:
                        connections.append(result.entity_summaries[node_id])
                
                return f"Key relationships: {'; '.join(connections)}"
            else:
                return "No significant relationships found."
        
        elif "pattern" in question_lower:
            return result.global_summary
        
        else:
            # Default to global summary
            return result.global_summary


# Example usage
if __name__ == "__main__":
    # Example documents
    documents = [
        "Apple Inc. is developing new AI features for the iPhone. Tim Cook announced the partnership with OpenAI.",
        "Microsoft and OpenAI are collaborating on Azure cloud services. Satya Nadella sees AI as transformative.",
        "Google's Gemini model competes with OpenAI's GPT series. Sundar Pichai emphasized multimodal capabilities.",
        "Meta's Llama models are open source. Mark Zuckerberg believes in open AI development."
    ]
    
    # Initialize crystallizer
    crystallizer = GraphRAGCrystallizer()
    
    # Process corpus
    result = crystallizer.crystallize_corpus(documents)
    
    # Print results
    print(f"Global Summary: {result.global_summary}")
    print(f"\nMetadata: {json.dumps(result.metadata, indent=2)}")
    
    # Answer global questions
    questions = [
        "What are the main themes?",
        "What are the key relationships?",
        "What patterns emerge?"
    ]
    
    for question in questions:
        answer = crystallizer.answer_global_question(result, question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")