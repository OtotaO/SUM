"""
knowledge_graph.py - Knowledge Graph System for Relationship Mapping

This module implements a knowledge graph layer that enables:
- Entity and relationship extraction
- Graph-based knowledge representation
- Concept evolution tracking
- Multi-hop reasoning

Author: SUM Development Team
License: Apache License 2.0
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx
import spacy
from datetime import datetime

# Try to import Neo4j
try:
    from py2neo import Graph, Node, Relationship
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    created_at: float
    updated_at: float
    frequency: int = 1
    confidence: float = 1.0


@dataclass
class GraphRelationship:
    """Represents a relationship between entities."""
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]
    created_at: float
    confidence: float = 1.0
    strength: float = 1.0


class KnowledgeGraphEngine:
    """
    Knowledge graph engine for entity and relationship management.
    Provides graph-based knowledge representation and reasoning.
    """
    
    def __init__(self,
                 storage_path: str = "./knowledge_graph",
                 neo4j_uri: Optional[str] = None,
                 neo4j_auth: Optional[Tuple[str, str]] = None,
                 spacy_model: str = "en_core_web_sm"):
        """
        Initialize the knowledge graph engine.
        
        Args:
            storage_path: Path for persistent storage
            neo4j_uri: Neo4j connection URI
            neo4j_auth: Neo4j authentication (user, password)
            spacy_model: Spacy model for NER
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize graph backend
        self._init_graph_backend(neo4j_uri, neo4j_auth)
        
        # Initialize NLP components
        self._init_nlp(spacy_model)
        
        # Load or initialize graph data
        self._load_graph_data()
        
        # Statistics
        self.stats = {
            'total_entities': 0,
            'total_relationships': 0,
            'entity_types': defaultdict(int),
            'relationship_types': defaultdict(int),
            'last_update': time.time()
        }
        self._update_stats()
    
    def _init_graph_backend(self, neo4j_uri: Optional[str], neo4j_auth: Optional[Tuple[str, str]]):
        """Initialize the graph database backend."""
        self.use_neo4j = False
        
        if NEO4J_AVAILABLE and neo4j_uri and neo4j_auth:
            try:
                self.neo4j_graph = Graph(neo4j_uri, auth=neo4j_auth)
                self.use_neo4j = True
                logger.info("Connected to Neo4j graph database")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self._init_networkx_backend()
        else:
            self._init_networkx_backend()
    
    def _init_networkx_backend(self):
        """Initialize NetworkX as fallback graph backend."""
        self.nx_graph = nx.DiGraph()
        self.entities = {}
        self.relationships = {}
        logger.info("Using NetworkX graph backend")
    
    def _init_nlp(self, spacy_model: str):
        """Initialize NLP components."""
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            logger.warning("Entity extraction will be limited")
            self.nlp = None
    
    def _load_graph_data(self):
        """Load graph data from disk."""
        if not self.use_neo4j:
            # Load entities
            entities_path = os.path.join(self.storage_path, "entities.json")
            if os.path.exists(entities_path):
                with open(entities_path, 'r') as f:
                    entities_data = json.load(f)
                    self.entities = {
                        eid: Entity(**edata) for eid, edata in entities_data.items()
                    }
            
            # Load relationships
            relationships_path = os.path.join(self.storage_path, "relationships.json")
            if os.path.exists(relationships_path):
                with open(relationships_path, 'r') as f:
                    relationships_data = json.load(f)
                    self.relationships = {
                        rid: GraphRelationship(**rdata) 
                        for rid, rdata in relationships_data.items()
                    }
            
            # Rebuild NetworkX graph
            self._rebuild_nx_graph()
    
    def _rebuild_nx_graph(self):
        """Rebuild NetworkX graph from entities and relationships."""
        self.nx_graph.clear()
        
        # Add nodes
        for entity_id, entity in self.entities.items():
            self.nx_graph.add_node(entity_id, **asdict(entity))
        
        # Add edges
        for rel_id, rel in self.relationships.items():
            self.nx_graph.add_edge(
                rel.source_id,
                rel.target_id,
                relationship_id=rel_id,
                relationship_type=rel.relationship_type,
                **rel.properties
            )
    
    def extract_entities_and_relationships(self, text: str, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Input text
            source: Source identifier for tracking
            
        Returns:
            Dictionary with extracted entities and relationships
        """
        entities = []
        relationships = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract named entities
            entity_map = {}
            for ent in doc.ents:
                entity_id = self._generate_entity_id(ent.text, ent.label_)
                entity = Entity(
                    id=entity_id,
                    name=ent.text,
                    type=ent.label_,
                    properties={
                        'source': source,
                        'context': text[max(0, ent.start_char-50):ent.end_char+50]
                    },
                    created_at=time.time(),
                    updated_at=time.time()
                )
                entities.append(entity)
                entity_map[ent.text] = entity
            
            # Extract relationships using dependency parsing
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'pobj'] and token.head.pos_ == 'VERB':
                    # Find subject and object
                    subject = None
                    obj = None
                    
                    for child in token.head.children:
                        if child.dep_ == 'nsubj':
                            subject = child.text
                        elif child.dep_ in ['dobj', 'pobj']:
                            obj = child.text
                    
                    if subject and obj and subject in entity_map and obj in entity_map:
                        rel_id = self._generate_relationship_id(
                            entity_map[subject].id,
                            entity_map[obj].id,
                            token.head.text
                        )
                        
                        relationship = GraphRelationship(
                            id=rel_id,
                            source_id=entity_map[subject].id,
                            target_id=entity_map[obj].id,
                            relationship_type=token.head.text,
                            properties={
                                'source': source,
                                'context': token.head.sent.text
                            },
                            created_at=time.time()
                        )
                        relationships.append(relationship)
        else:
            # Fallback: simple pattern-based extraction
            import re
            
            # Extract capitalized phrases as entities
            pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            matches = re.findall(pattern, text)
            
            for match in matches:
                entity_id = self._generate_entity_id(match, 'MISC')
                entity = Entity(
                    id=entity_id,
                    name=match,
                    type='MISC',
                    properties={'source': source},
                    created_at=time.time(),
                    updated_at=time.time()
                )
                entities.append(entity)
        
        # Store extracted entities and relationships
        for entity in entities:
            self.add_entity(entity)
        
        for relationship in relationships:
            self.add_relationship(relationship)
        
        return {
            'entities': [asdict(e) for e in entities],
            'relationships': [asdict(r) for r in relationships],
            'extraction_time': time.time()
        }
    
    def add_entity(self, entity: Entity) -> bool:
        """
        Add or update an entity in the graph.
        
        Args:
            entity: Entity to add
            
        Returns:
            Success status
        """
        if self.use_neo4j:
            try:
                node = Node(entity.type, 
                           name=entity.name,
                           entity_id=entity.id,
                           **entity.properties)
                self.neo4j_graph.merge(node, entity.type, "entity_id")
                return True
            except Exception as e:
                logger.error(f"Failed to add entity to Neo4j: {e}")
                return False
        else:
            # Check if entity exists
            if entity.id in self.entities:
                # Update existing entity
                existing = self.entities[entity.id]
                existing.frequency += 1
                existing.updated_at = time.time()
                existing.properties.update(entity.properties)
            else:
                # Add new entity
                self.entities[entity.id] = entity
                self.nx_graph.add_node(entity.id, **asdict(entity))
            
            self._save_graph_data()
            self._update_stats()
            return True
    
    def add_relationship(self, relationship: GraphRelationship) -> bool:
        """
        Add a relationship to the graph.
        
        Args:
            relationship: Relationship to add
            
        Returns:
            Success status
        """
        # Verify entities exist
        if self.use_neo4j:
            try:
                source = self.neo4j_graph.nodes.match(entity_id=relationship.source_id).first()
                target = self.neo4j_graph.nodes.match(entity_id=relationship.target_id).first()
                
                if source and target:
                    rel = Relationship(source, relationship.relationship_type, target,
                                     relationship_id=relationship.id,
                                     **relationship.properties)
                    self.neo4j_graph.merge(rel)
                    return True
                else:
                    logger.warning(f"Entities not found for relationship: {relationship.id}")
                    return False
            except Exception as e:
                logger.error(f"Failed to add relationship to Neo4j: {e}")
                return False
        else:
            if relationship.source_id in self.entities and relationship.target_id in self.entities:
                self.relationships[relationship.id] = relationship
                self.nx_graph.add_edge(
                    relationship.source_id,
                    relationship.target_id,
                    relationship_id=relationship.id,
                    relationship_type=relationship.relationship_type,
                    **relationship.properties
                )
                self._save_graph_data()
                self._update_stats()
                return True
            else:
                logger.warning(f"Entities not found for relationship: {relationship.id}")
                return False
    
    def find_path(self, start_entity_id: str, end_entity_id: str, max_length: int = 5) -> List[List[str]]:
        """
        Find paths between two entities.
        
        Args:
            start_entity_id: Starting entity ID
            end_entity_id: Target entity ID
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        if self.use_neo4j:
            query = """
            MATCH path = (start {entity_id: $start_id})-[*..%d]-(end {entity_id: $end_id})
            RETURN path
            LIMIT 10
            """ % max_length
            
            try:
                results = self.neo4j_graph.run(query, start_id=start_entity_id, end_id=end_entity_id)
                paths = []
                for record in results:
                    path = record['path']
                    entity_ids = [node['entity_id'] for node in path.nodes]
                    paths.append(entity_ids)
                return paths
            except Exception as e:
                logger.error(f"Failed to find paths in Neo4j: {e}")
                return []
        else:
            try:
                # Find all simple paths
                paths = list(nx.all_simple_paths(
                    self.nx_graph,
                    start_entity_id,
                    end_entity_id,
                    cutoff=max_length
                ))
                return paths[:10]  # Limit to 10 paths
            except nx.NetworkXNoPath:
                return []
            except Exception as e:
                logger.error(f"Failed to find paths: {e}")
                return []
    
    def get_entity_context(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get the context around an entity (neighboring entities and relationships).
        
        Args:
            entity_id: Entity ID
            depth: How many hops to explore
            
        Returns:
            Context dictionary
        """
        if entity_id not in self.entities and not self.use_neo4j:
            return {}
        
        context = {
            'center_entity': None,
            'entities': {},
            'relationships': [],
            'depth': depth
        }
        
        if self.use_neo4j:
            query = """
            MATCH (center {entity_id: $entity_id})
            OPTIONAL MATCH path = (center)-[*..%d]-(connected)
            RETURN center, path
            """ % depth
            
            try:
                results = self.neo4j_graph.run(query, entity_id=entity_id)
                # Process results (simplified for brevity)
                return context
            except Exception as e:
                logger.error(f"Failed to get context from Neo4j: {e}")
                return context
        else:
            # Get center entity
            if entity_id in self.entities:
                context['center_entity'] = asdict(self.entities[entity_id])
            
            # Use NetworkX ego graph
            ego = nx.ego_graph(self.nx_graph, entity_id, radius=depth)
            
            # Collect entities
            for node_id in ego.nodes():
                if node_id in self.entities:
                    context['entities'][node_id] = asdict(self.entities[node_id])
            
            # Collect relationships
            for edge in ego.edges(data=True):
                source_id, target_id, data = edge
                if 'relationship_id' in data:
                    rel_id = data['relationship_id']
                    if rel_id in self.relationships:
                        context['relationships'].append(asdict(self.relationships[rel_id]))
            
            return context
    
    def find_communities(self) -> Dict[str, List[str]]:
        """
        Find communities in the knowledge graph.
        
        Returns:
            Dictionary mapping community ID to list of entity IDs
        """
        if self.use_neo4j:
            # Use Neo4j's community detection algorithms
            query = """
            CALL algo.louvain.stream(null, null, {})
            YIELD nodeId, community
            RETURN algo.getNodeById(nodeId).entity_id AS entity_id, community
            """
            
            try:
                results = self.neo4j_graph.run(query)
                communities = defaultdict(list)
                for record in results:
                    communities[str(record['community'])].append(record['entity_id'])
                return dict(communities)
            except Exception as e:
                logger.warning(f"Neo4j community detection failed: {e}")
                # Fall back to NetworkX
        
        # Use NetworkX community detection
        try:
            import community as community_louvain
            
            # Convert to undirected for community detection
            undirected = self.nx_graph.to_undirected()
            
            # Detect communities
            partition = community_louvain.best_partition(undirected)
            
            # Group by community
            communities = defaultdict(list)
            for node_id, comm_id in partition.items():
                communities[str(comm_id)].append(node_id)
            
            return dict(communities)
        except ImportError:
            logger.warning("python-louvain not installed, using connected components")
            
            # Fallback to connected components
            undirected = self.nx_graph.to_undirected()
            communities = {}
            
            for i, component in enumerate(nx.connected_components(undirected)):
                communities[str(i)] = list(component)
            
            return communities
    
    def get_entity_importance(self, entity_id: str) -> float:
        """
        Calculate the importance of an entity based on graph centrality.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Importance score (0-1)
        """
        if entity_id not in self.entities and not self.use_neo4j:
            return 0.0
        
        if self.use_neo4j:
            query = """
            MATCH (e {entity_id: $entity_id})
            OPTIONAL MATCH (e)-[r]-()
            RETURN COUNT(r) as degree
            """
            
            try:
                result = self.neo4j_graph.run(query, entity_id=entity_id).data()
                if result:
                    degree = result[0]['degree']
                    # Normalize by max degree
                    max_degree = self.stats.get('max_degree', 1)
                    return min(degree / max_degree, 1.0)
                return 0.0
            except Exception as e:
                logger.error(f"Failed to calculate importance in Neo4j: {e}")
                return 0.0
        else:
            # Use PageRank for importance
            try:
                pagerank = nx.pagerank(self.nx_graph)
                return pagerank.get(entity_id, 0.0)
            except Exception as e:
                # Fallback to degree centrality
                try:
                    centrality = nx.degree_centrality(self.nx_graph)
                    return centrality.get(entity_id, 0.0)
                except Exception as e2:
                    logger.error(f"Failed to calculate importance: {e2}")
                    return 0.0
    
    def visualize_subgraph(self, 
                          entity_ids: List[str],
                          output_path: Optional[str] = None,
                          show_labels: bool = True) -> Optional[str]:
        """
        Visualize a subgraph containing specified entities.
        
        Args:
            entity_ids: List of entity IDs to include
            output_path: Path to save visualization
            show_labels: Whether to show entity labels
            
        Returns:
            Path to saved visualization or None
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for visualization")
            return None
        
        try:
            # Create subgraph
            if self.use_neo4j:
                # For Neo4j, we'll create a NetworkX graph from query results
                subgraph = nx.DiGraph()
                
                # Add entities and their connections
                for entity_id in entity_ids:
                    if entity_id in self.entities:
                        entity = self.entities[entity_id]
                        subgraph.add_node(entity_id, label=entity.name, type=entity.type)
            else:
                # Extract subgraph from NetworkX
                subgraph = self.nx_graph.subgraph(entity_ids).copy()
            
            if len(subgraph) == 0:
                logger.warning("No entities found for visualization")
                return None
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Layout
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
            
            # Draw nodes by type
            node_colors = []
            for node in subgraph.nodes():
                if node in self.entities:
                    entity_type = self.entities[node].type
                    # Map entity types to colors
                    color_map = {
                        'PERSON': 'lightblue',
                        'ORG': 'lightgreen',
                        'LOC': 'lightcoral',
                        'DATE': 'lightyellow',
                        'MISC': 'lightgray'
                    }
                    node_colors.append(color_map.get(entity_type, 'lightgray'))
                else:
                    node_colors.append('lightgray')
            
            # Draw nodes
            nx.draw_networkx_nodes(subgraph, pos,
                                 node_color=node_colors,
                                 node_size=500,
                                 alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(subgraph, pos,
                                 edge_color='gray',
                                 arrows=True,
                                 alpha=0.5,
                                 arrowsize=20)
            
            # Draw labels
            if show_labels:
                labels = {}
                for node in subgraph.nodes():
                    if node in self.entities:
                        labels[node] = self.entities[node].name[:20]  # Truncate long names
                    else:
                        labels[node] = node[:20]
                
                nx.draw_networkx_labels(subgraph, pos, labels,
                                      font_size=8,
                                      font_weight='bold')
            
            # Add title and legend
            plt.title("Knowledge Graph Visualization", fontsize=16, fontweight='bold')
            
            # Create legend
            legend_elements = [
                mpatches.Patch(color='lightblue', label='Person'),
                mpatches.Patch(color='lightgreen', label='Organization'),
                mpatches.Patch(color='lightcoral', label='Location'),
                mpatches.Patch(color='lightyellow', label='Date'),
                mpatches.Patch(color='lightgray', label='Other')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return output_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Failed to visualize graph: {e}")
            return None
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a unique entity ID."""
        import hashlib
        return hashlib.sha256(f"{name}_{entity_type}".encode()).hexdigest()[:16]
    
    def _generate_relationship_id(self, source_id: str, target_id: str, rel_type: str) -> str:
        """Generate a unique relationship ID."""
        import hashlib
        return hashlib.sha256(f"{source_id}_{target_id}_{rel_type}".encode()).hexdigest()[:16]
    
    def _save_graph_data(self):
        """Save graph data to disk."""
        if not self.use_neo4j:
            # Save entities
            entities_path = os.path.join(self.storage_path, "entities.json")
            entities_data = {eid: asdict(entity) for eid, entity in self.entities.items()}
            with open(entities_path, 'w') as f:
                json.dump(entities_data, f, indent=2)
            
            # Save relationships
            relationships_path = os.path.join(self.storage_path, "relationships.json")
            relationships_data = {rid: asdict(rel) for rid, rel in self.relationships.items()}
            with open(relationships_path, 'w') as f:
                json.dump(relationships_data, f, indent=2)
    
    def _update_stats(self):
        """Update graph statistics."""
        if self.use_neo4j:
            try:
                # Get entity count by type
                query = """
                MATCH (n)
                RETURN labels(n)[0] as type, COUNT(n) as count
                """
                results = self.neo4j_graph.run(query)
                
                self.stats['entity_types'] = defaultdict(int)
                total_entities = 0
                
                for record in results:
                    entity_type = record['type']
                    count = record['count']
                    self.stats['entity_types'][entity_type] = count
                    total_entities += count
                
                self.stats['total_entities'] = total_entities
                
                # Get relationship count
                query = "MATCH ()-[r]->() RETURN COUNT(r) as count"
                result = self.neo4j_graph.run(query).data()
                self.stats['total_relationships'] = result[0]['count'] if result else 0
                
            except Exception as e:
                logger.error(f"Failed to update stats from Neo4j: {e}")
        else:
            self.stats['total_entities'] = len(self.entities)
            self.stats['total_relationships'] = len(self.relationships)
            
            # Count entity types
            self.stats['entity_types'] = defaultdict(int)
            for entity in self.entities.values():
                self.stats['entity_types'][entity.type] += 1
            
            # Count relationship types
            self.stats['relationship_types'] = defaultdict(int)
            for rel in self.relationships.values():
                self.stats['relationship_types'][rel.relationship_type] += 1
            
            # Calculate max degree
            if self.nx_graph.number_of_nodes() > 0:
                degrees = dict(self.nx_graph.degree())
                self.stats['max_degree'] = max(degrees.values()) if degrees else 0
        
        self.stats['last_update'] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return dict(self.stats)


# Global instance
_knowledge_graph_engine = None


def get_knowledge_graph_engine() -> KnowledgeGraphEngine:
    """Get or create the global knowledge graph engine."""
    global _knowledge_graph_engine
    if _knowledge_graph_engine is None:
        _knowledge_graph_engine = KnowledgeGraphEngine()
    return _knowledge_graph_engine


if __name__ == "__main__":
    # Example usage
    engine = get_knowledge_graph_engine()
    
    # Extract entities and relationships from text
    text = """
    Apple Inc. was founded by Steve Jobs in Cupertino, California. 
    The company revolutionized personal computing and later dominated 
    the smartphone market with the iPhone.
    """
    
    results = engine.extract_entities_and_relationships(text, source="example")
    
    print(f"Extracted {len(results['entities'])} entities")
    print(f"Extracted {len(results['relationships'])} relationships")
    
    # Find paths between entities
    if len(results['entities']) >= 2:
        entity_ids = [e['id'] for e in results['entities'][:2]]
        paths = engine.find_path(entity_ids[0], entity_ids[1])
        print(f"Found {len(paths)} paths between entities")
    
    # Get statistics
    stats = engine.get_stats()
    print(f"Graph stats: {stats}")