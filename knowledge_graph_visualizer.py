"""
knowledge_graph.py - Knowledge graph construction and visualization module

This module provides tools for building, analyzing, and visualizing knowledge graphs
based on text data, highlighting relationships between concepts and entities.

Design principles:
- Algorithmic efficiency (Knuth optimization)
- Clean, maintainable code (Torvalds/van Rossum style)
- Security by design (Schneier principles)
- Visualization best practices (Tufte principles)
- Scalable architecture (Fowler patterns)

Author: ototao
License: Apache License 2.0
"""

import logging
import time
import json
import os
import re
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict, Counter
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import numpy as np
from pathlib import Path
import hashlib
from threading import Lock
import uuid

# Logger configuration
logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Knowledge graph constructor and visualizer.
    
    This class builds knowledge graphs from processed text data,
    providing analysis and visualization capabilities to explore
    entity relationships and concept networks.
    """
    
    # Entity type color mapping
    ENTITY_COLORS = {
        'PERSON': '#ff7f0e',      # Orange
        'ORG': '#1f77b4',         # Blue
        'GPE': '#2ca02c',         # Green
        'LOC': '#9467bd',         # Purple
        'DATE': '#8c564b',        # Brown
        'TIME': '#e377c2',        # Pink
        'MONEY': '#bcbd22',       # Olive
        'PERCENT': '#17becf',     # Cyan
        'WORK_OF_ART': '#d62728', # Red
        'EVENT': '#9edae5',       # Light blue
        'PRODUCT': '#c49c94',     # Light brown
        'TOPIC': '#dbdb8d',       # Light yellow
        'CONCEPT': '#c7c7c7',     # Gray
        'DEFAULT': '#7f7f7f'      # Dark gray
    }
    
    def __init__(self, 
                output_dir: str = 'output',
                max_entities: int = 100,
                min_edge_weight: float = 0.1,
                use_cache: bool = True,
                cache_dir: str = None):
        """
        Initialize the knowledge graph builder.
        
        Args:
            output_dir: Directory to save visualizations
            max_entities: Maximum number of entities to include in graph
            min_edge_weight: Minimum weight for edges to be included
            use_cache: Whether to cache graph data
            cache_dir: Directory for caching (None = use output_dir/cache)
        """
        # Configuration
        self.output_dir = output_dir
        self.max_entities = max_entities
        self.min_edge_weight = min_edge_weight
        self.use_cache = use_cache
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up cache directory if enabled
        self.cache_dir = cache_dir or os.path.join(output_dir, 'cache')
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Thread safety for file operations
        self.file_lock = Lock()
        
        # Initialize graph data structures
        self.reset_graph()
        
        logger.info(f"Initialized KnowledgeGraph with max_entities={max_entities}")
    
    def reset_graph(self) -> None:
        """Reset all graph data structures."""
        self.G = nx.Graph()
        self.nodes = {}
        self.edges = {}
        self.entity_counts = Counter()
        self.entity_types = {}
        self.relation_types = {}
        self.source_texts = set()
        self.graph_metadata = {
            "creation_time": time.time(),
            "node_count": 0,
            "edge_count": 0,
            "source_count": 0
        }
    
    def build_from_entities(self, 
                          entities: List[Tuple[str, str, int]], 
                          source_id: str = None) -> None:
        """
        Build graph from extracted entity data.
        
        Args:
            entities: List of (entity_text, entity_type, count) tuples
            source_id: Identifier for the source text
        """
        if not entities:
            logger.warning("No entities provided to build graph")
            return
            
        # Generate source ID if not provided
        if not source_id:
            source_id = f"source_{len(self.source_texts)}"
            
        # Add source to tracking
        self.source_texts.add(source_id)
        
        # Add entities as nodes
        for entity_text, entity_type, count in entities:
            # Clean and normalize entity text
            entity_text = self._normalize_entity(entity_text)
            if not entity_text:
                continue
                
            # Update entity tracking
            self.entity_counts[entity_text] += count
            self.entity_types[entity_text] = entity_type
            
            # Add or update node
            if entity_text in self.nodes:
                node_id = self.nodes[entity_text]
                # Update node weight
                if self.G.has_node(node_id):
                    self.G.nodes[node_id]['weight'] += count
                    self.G.nodes[node_id]['sources'].add(source_id)
            else:
                # Create new node
                node_id = len(self.nodes)
                self.nodes[entity_text] = node_id
                self.G.add_node(
                    node_id,
                    label=entity_text,
                    type=entity_type,
                    weight=count,
                    sources={source_id}
                )
        
        # Add edges based on co-occurrence
        entity_list = [e[0] for e in entities]
        seen_pairs = set()
        
        for i in range(len(entity_list)):
            for j in range(i+1, len(entity_list)):
                # Get normalized entity texts
                source_entity = self._normalize_entity(entity_list[i])
                target_entity = self._normalize_entity(entity_list[j])
                
                if not source_entity or not target_entity or source_entity == target_entity:
                    continue
                    
                # Ensure entities are in the graph
                if source_entity not in self.nodes or target_entity not in self.nodes:
                    continue
                    
                # Get node IDs
                source_id = self.nodes[source_entity]
                target_id = self.nodes[target_entity]
                
                # Avoid duplicate processing
                edge_key = tuple(sorted([source_id, target_id]))
                if edge_key in seen_pairs:
                    continue
                    
                seen_pairs.add(edge_key)
                
                # Add or update edge
                if self.G.has_edge(source_id, target_id):
                    # Increment edge weight
                    self.G[source_id][target_id]['weight'] += 1
                else:
                    # Create new edge
                    self.G.add_edge(
                        source_id,
                        target_id,
                        weight=1,
                        type='co-occurrence'
                    )
                    self.edges[edge_key] = {
                        'source': source_entity,
                        'target': target_entity,
                        'weight': 1,
                        'type': 'co-occurrence'
                    }
        
        # Update metadata
        self.graph_metadata["node_count"] = self.G.number_of_nodes()
        self.graph_metadata["edge_count"] = self.G.number_of_edges()
        self.graph_metadata["source_count"] = len(self.source_texts)
        
        logger.info(f"Built graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
    
    def build_from_relationships(self, 
                              relationships: List[Dict],
                              source_id: str = None) -> None:
        """
        Build graph from explicit relationship data.
        
        Args:
            relationships: List of relationship dictionaries with
                           source, target, type, and weight
            source_id: Identifier for the source text
        """
        if not relationships:
            logger.warning("No relationships provided to build graph")
            return
            
        # Generate source ID if not provided
        if not source_id:
            source_id = f"source_{len(self.source_texts)}"
            
        # Add source to tracking
        self.source_texts.add(source_id)
        
        # Process each relationship
        for rel in relationships:
            # Validate relationship data
            if not all(k in rel for k in ['source', 'target', 'type']):
                logger.warning(f"Invalid relationship structure: {rel}")
                continue
                
            # Extract and normalize entities
            source_entity = self._normalize_entity(rel['source'])
            target_entity = self._normalize_entity(rel['target'])
            relation_type = rel.get('type', 'generic')
            weight = float(rel.get('weight', 1.0))
            
            if not source_entity or not target_entity:
                continue
                
            # Add source entity
            source_type = rel.get('source_type', 'DEFAULT')
            if source_entity not in self.nodes:
                node_id = len(self.nodes)
                self.nodes[source_entity] = node_id
                self.G.add_node(
                    node_id,
                    label=source_entity,
                    type=source_type,
                    weight=1,
                    sources={source_id}
                )
                self.entity_types[source_entity] = source_type
                self.entity_counts[source_entity] = 1
            else:
                node_id = self.nodes[source_entity]
                self.G.nodes[node_id]['weight'] += 1
                self.G.nodes[node_id]['sources'].add(source_id)
                self.entity_counts[source_entity] += 1
                
            # Add target entity
            target_type = rel.get('target_type', 'DEFAULT')
            if target_entity not in self.nodes:
                node_id = len(self.nodes)
                self.nodes[target_entity] = node_id
                self.G.add_node(
                    node_id,
                    label=target_entity,
                    type=target_type,
                    weight=1,
                    sources={source_id}
                )
                self.entity_types[target_entity] = target_type
                self.entity_counts[target_entity] = 1
            else:
                node_id = self.nodes[target_entity]
                self.G.nodes[node_id]['weight'] += 1
                self.G.nodes[node_id]['sources'].add(source_id)
                self.entity_counts[target_entity] += 1
                
            # Add the edge
            source_node_id = self.nodes[source_entity]
            target_node_id = self.nodes[target_entity]
            
            edge_key = tuple(sorted([source_node_id, target_node_id]))
            
            if self.G.has_edge(source_node_id, target_node_id):
                # Update existing edge
                self.G[source_node_id][target_node_id]['weight'] += weight
                # If relation types are different, use the more specific one
                if relation_type != 'generic':
                    self.G[source_node_id][target_node_id]['type'] = relation_type
            else:
                # Create new edge
                self.G.add_edge(
                    source_node_id,
                    target_node_id,
                    weight=weight,
                    type=relation_type
                )
                self.edges[edge_key] = {
                    'source': source_entity,
                    'target': target_entity,
                    'weight': weight,
                    'type': relation_type
                }
                
            # Track relation type
            self.relation_types[relation_type] = self.relation_types.get(relation_type, 0) + 1
        
        # Update metadata
        self.graph_metadata["node_count"] = self.G.number_of_nodes()
        self.graph_metadata["edge_count"] = self.G.number_of_edges()
        self.graph_metadata["source_count"] = len(self.source_texts)
        
        logger.info(f"Built graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
    
    def add_topics(self, 
                 topics: List[Dict],
                 source_id: str = None) -> None:
        """
        Add topic nodes and connect to related entities.
        
        Args:
            topics: List of topic dictionaries with id, label, and terms
            source_id: Identifier for the source text
        """
        if not topics:
            logger.warning("No topics provided to add to graph")
            return
            
        # Generate source ID if not provided
        if not source_id:
            source_id = f"source_{len(self.source_texts)}"
            
        # Add source to tracking
        self.source_texts.add(source_id)
        
        # Process each topic
        for topic in topics:
            # Validate topic data
            if not all(k in topic for k in ['id', 'label']):
                logger.warning(f"Invalid topic structure: {topic}")
                continue
                
            # Extract topic data
            topic_id = str(topic['id'])
            topic_label = topic['label']
            topic_terms = topic.get('terms', [])
            
            # Create normalized topic name
            topic_node_name = f"TOPIC_{topic_id}_{topic_label}"
            topic_node_name = self._normalize_entity(topic_node_name)
            
            # Add topic as node
            if topic_node_name not in self.nodes:
                node_id = len(self.nodes)
                self.nodes[topic_node_name] = node_id
                self.G.add_node(
                    node_id,
                    label=topic_label,
                    type='TOPIC',
                    weight=len(topic_terms),
                    sources={source_id},
                    is_topic=True
                )
                self.entity_types[topic_node_name] = 'TOPIC'
                self.entity_counts[topic_node_name] = len(topic_terms)
            else:
                node_id = self.nodes[topic_node_name]
                self.G.nodes[node_id]['weight'] += len(topic_terms)
                self.G.nodes[node_id]['sources'].add(source_id)
                self.entity_counts[topic_node_name] += len(topic_terms)
                
            # Connect topic to terms
            for term_data in topic_terms:
                # Handle different term formats
                if isinstance(term_data, dict):
                    term = term_data.get('term') or term_data.get('word')
                    weight = float(term_data.get('weight', 1.0))
                else:
                    term = str(term_data)
                    weight = 1.0
                
                if not term:
                    continue
                    
                # Normalize term
                term = self._normalize_entity(term)
                
                # Add term as node if it doesn't exist
                if term not in self.nodes:
                    term_node_id = len(self.nodes)
                    self.nodes[term] = term_node_id
                    self.G.add_node(
                        term_node_id,
                        label=term,
                        type='CONCEPT',
                        weight=1,
                        sources={source_id}
                    )
                    self.entity_types[term] = 'CONCEPT'
                    self.entity_counts[term] = 1
                else:
                    term_node_id = self.nodes[term]
                    self.G.nodes[term_node_id]['weight'] += 1
                    self.G.nodes[term_node_id]['sources'].add(source_id)
                    self.entity_counts[term] += 1
                
                # Connect topic to term
                topic_node_id = self.nodes[topic_node_name]
                edge_key = tuple(sorted([topic_node_id, term_node_id]))
                
                if self.G.has_edge(topic_node_id, term_node_id):
                    # Update existing edge
                    self.G[topic_node_id][term_node_id]['weight'] += weight
                else:
                    # Create new edge
                    self.G.add_edge(
                        topic_node_id,
                        term_node_id,
                        weight=weight,
                        type='topic_term'
                    )
                    self.edges[edge_key] = {
                        'source': topic_node_name,
                        'target': term,
                        'weight': weight,
                        'type': 'topic_term'
                    }
        
        # Update metadata
        self.graph_metadata["node_count"] = self.G.number_of_nodes()
        self.graph_metadata["edge_count"] = self.G.number_of_edges()
        self.graph_metadata["source_count"] = len(self.source_texts)
        
        logger.info(f"Added {len(topics)} topics to graph, now with {self.G.number_of_nodes()} nodes")
    
    def merge_graph(self, other_graph: 'KnowledgeGraph') -> None:
        """
        Merge another knowledge graph into this one.
        
        Args:
            other_graph: Another KnowledgeGraph instance to merge
        """
        if not isinstance(other_graph, KnowledgeGraph):
            logger.error("Can only merge with another KnowledgeGraph instance")
            return
            
        # Merge nodes
        for entity, node_id in other_graph.nodes.items():
            if entity in self.nodes:
                # Node exists, merge attributes
                self_node_id = self.nodes[entity]
                other_node = other_graph.G.nodes[node_id]
                
                # Update weight
                self.G.nodes[self_node_id]['weight'] += other_node['weight']
                
                # Merge sources
                self.G.nodes[self_node_id]['sources'].update(other_node.get('sources', set()))
                
                # Keep more specific type if available
                if other_node['type'] != 'DEFAULT' and self.G.nodes[self_node_id]['type'] == 'DEFAULT':
                    self.G.nodes[self_node_id]['type'] = other_node['type']
                    self.entity_types[entity] = other_node['type']
            else:
                # Add new node
                new_node_id = len(self.nodes)
                self.nodes[entity] = new_node_id
                
                # Copy node attributes
                other_node = other_graph.G.nodes[node_id]
                node_attrs = {k: v for k, v in other_node.items()}
                
                # Convert sources to set if it's not already
                if 'sources' in node_attrs and not isinstance(node_attrs['sources'], set):
                    node_attrs['sources'] = set(node_attrs['sources'])
                    
                self.G.add_node(new_node_id, **node_attrs)
                
                # Update tracking
                self.entity_types[entity] = other_node.get('type', 'DEFAULT')
                self.entity_counts[entity] = other_node.get('weight', 1)
        
        # Merge edges
        for u, v, data in other_graph.G.edges(data=True):
            # Get source and target entities
            source_entity = None
            target_entity = None
            
            for entity, nid in other_graph.nodes.items():
                if nid == u:
                    source_entity = entity
                elif nid == v:
                    target_entity = entity
                    
                if source_entity and target_entity:
                    break
            
            if not source_entity or not target_entity:
                continue
                
            # Get corresponding node IDs in this graph
            if source_entity in self.nodes and target_entity in self.nodes:
                source_id = self.nodes[source_entity]
                target_id = self.nodes[target_entity]
                
                # Add or update edge
                if self.G.has_edge(source_id, target_id):
                    # Update weight
                    self.G[source_id][target_id]['weight'] += data['weight']
                    
                    # Keep more specific relation type
                    if data['type'] != 'generic' and self.G[source_id][target_id]['type'] == 'generic':
                        self.G[source_id][target_id]['type'] = data['type']
                else:
                    # Add new edge
                    self.G.add_edge(source_id, target_id, **data)
                    
                    # Update edge tracking
                    edge_key = tuple(sorted([source_id, target_id]))
                    self.edges[edge_key] = {
                        'source': source_entity,
                        'target': target_entity,
                        'weight': data['weight'],
                        'type': data['type']
                    }
                    
                    # Update relation type counts
                    rel_type = data['type']
                    self.relation_types[rel_type] = self.relation_types.get(rel_type, 0) + 1
        
        # Merge source texts
        self.source_texts.update(other_graph.source_texts)
        
        # Update metadata
        self.graph_metadata["node_count"] = self.G.number_of_nodes()
        self.graph_metadata["edge_count"] = self.G.number_of_edges()
        self.graph_metadata["source_count"] = len(self.source_texts)
        
        logger.info(f"Merged graphs, now with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
    
    def prune_graph(self, 
                  min_node_weight: float = 1.0,
                  min_edge_weight: float = 0.1,
                  max_nodes: int = None) -> None:
        """
        Prune the graph to remove weak connections and limit size.
        
        Args:
            min_node_weight: Minimum weight for nodes to keep
            min_edge_weight: Minimum weight for edges to keep
            max_nodes: Maximum number of nodes to keep (None = no limit)
        """
        if not self.G.number_of_nodes():
            logger.warning("No graph to prune")
            return
            
        original_nodes = self.G.number_of_nodes()
        original_edges = self.G.number_of_edges()
        
        # Set max_nodes if not specified
        max_nodes = max_nodes or self.max_entities
        
        # Remove weak edges
        edges_to_remove = []
        for u, v, data in self.G.edges(data=True):
            if data['weight'] < min_edge_weight:
                edges_to_remove.append((u, v))
                
        for u, v in edges_to_remove:
            self.G.remove_edge(u, v)
            edge_key = tuple(sorted([u, v]))
            if edge_key in self.edges:
                del self.edges[edge_key]
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(self.G))
        self.G.remove_nodes_from(isolated_nodes)
        
        # Remove nodes with low weight
        low_weight_nodes = [n for n, data in self.G.nodes(data=True) if data['weight'] < min_node_weight]
        self.G.remove_nodes_from(low_weight_nodes)
        
        # If still over max_nodes, keep the most important ones
        if max_nodes and self.G.number_of_nodes() > max_nodes:
            # Calculate node importance (degree centrality * weight)
            node_importance = {}
            centrality = nx.degree_centrality(self.G)
            
            for node in self.G.nodes():
                node_weight = self.G.nodes[node]['weight']
                node_importance[node] = centrality[node] * node_weight
                
            # Sort nodes by importance
            sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Keep only the top max_nodes
            nodes_to_keep = set(n for n, _ in sorted_nodes[:max_nodes])
            nodes_to_remove = [n for n in self.G.nodes() if n not in nodes_to_keep]
            self.G.remove_nodes_from(nodes_to_remove)
        
        # Rebuild node and edge mappings
        self._rebuild_mappings()
        
        # Update metadata
        self.graph_metadata["node_count"] = self.G.number_of_nodes()
        self.graph_metadata["edge_count"] = self.G.number_of_edges()
        
        logger.info(f"Pruned graph from {original_nodes} to {self.G.number_of_nodes()} nodes")
        logger.info(f"Pruned edges from {original_edges} to {self.G.number_of_edges()} edges")
    
    def _rebuild_mappings(self) -> None:
        """Rebuild node and edge mappings after graph changes."""
        # Rebuild node mapping
        new_nodes = {}
        for entity, node_id in self.nodes.items():
            if self.G.has_node(node_id):
                new_nodes[entity] = node_id
                
        self.nodes = new_nodes
        
        # Rebuild entity tracking
        new_entity_counts = Counter()
        new_entity_types = {}
        
        for entity, node_id in self.nodes.items():
            new_entity_counts[entity] = self.G.nodes[node_id]['weight']
            new_entity_types[entity] = self.G.nodes[node_id]['type']
            
        self.entity_counts = new_entity_counts
        self.entity_types = new_entity_types
        
        # Rebuild edge mapping
        new_edges = {}
        for edge_key, edge_data in self.edges.items():
            source_entity = edge_data['source']
            target_entity = edge_data['target']
            
            if (source_entity in self.nodes and 
                target_entity in self.nodes and
                self.G.has_edge(self.nodes[source_entity], self.nodes[target_entity])):
                new_edges[edge_key] = edge_data
                
        self.edges = new_edges
        
        # Rebuild relation type counts
        new_relation_types = defaultdict(int)
        for _, _, data in self.G.edges(data=True):
            rel_type = data['type']
            new_relation_types[rel_type] += 1
            
        self.relation_types = dict(new_relation_types)
    
    def get_central_entities(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most central entities in the graph.
        
        Args:
            top_n: Number of top entities to return
            
        Returns:
            List of (entity, centrality_score) tuples
        """
        if not self.G.number_of_nodes():
            return []
            
        # Calculate eigenvector centrality
        try:
            centrality = nx.eigenvector_centrality(self.G, weight='weight')
        except nx.PowerIterationFailedConvergence:
            # Fallback to degree centrality if eigenvector fails
            centrality = nx.degree_centrality(self.G)
            
        # Map node IDs to entities
        node_to_entity = {node_id: entity for entity, node_id in self.nodes.items()}
        
        # Sort by centrality
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        # Get top entities
        top_entities = []
        for node_id, score in sorted_nodes[:top_n]:
            if node_id in node_to_entity:
                top_entities.append((node_to_entity[node_id], score))
                
        return top_entities
    
    def get_communities(self, min_community_size: int = 3) -> Dict[int, List[str]]:
        """
        Detect communities in the graph.
        
        Args:
            min_community_size: Minimum size for a community
            
        Returns:
            Dictionary mapping community IDs to lists of entities
        """
        if not self.G.number_of_nodes():
            return {}
            
        # Detect communities using Louvain method
        try:
            from community import best_partition
            partition = best_partition(self.G, weight='weight')
        except ImportError:
            # Fallback to connected components
            logger.warning("python-louvain not available, using connected components instead")
            communities = list(nx.connected_components(self.G))
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
        
        # Group entities by community
        community_entities = defaultdict(list)
        node_to_entity = {node_id: entity for entity, node_id in self.nodes.items()}
        
        for node_id, community_id in partition.items():
            if node_id in node_to_entity:
                community_entities[community_id].append(node_to_entity[node_id])
        
        # Filter by minimum size
        return {cid: entities for cid, entities in community_entities.items() 
                if len(entities) >= min_community_size}
    
    def find_paths(self, 
                 source_entity: str, 
                 target_entity: str, 
                 max_length: int = 3) -> List[List[Tuple[str, str]]]:
        """
        Find paths between two entities.
        
        Args:
            source_entity: Starting entity
            target_entity: Target entity
            max_length: Maximum path length
            
        Returns:
            List of paths, each a list of (entity, relation_type) tuples
        """
        # Validate entities
        if source_entity not in self.nodes or target_entity not in self.nodes:
            logger.warning(f"Source or target entity not in graph")
            return []
            
        source_id = self.nodes[source_entity]
        target_id = self.nodes[target_entity]
        
        # Find all paths up to max_length
        try:
            paths = list(nx.all_simple_paths(
                self.G, source_id, target_id, cutoff=max_length
            ))
        except nx.NetworkXNoPath:
            return []
        
        # Convert to entity paths with relation types
        entity_paths = []
        node_to_entity = {node_id: entity for entity, node_id in self.nodes.items()}
        
        for path in paths:
            entity_path = []
            
            for i in range(len(path)):
                node_id = path[i]
                entity = node_to_entity[node_id]
                
                # Add entity to path
                if i < len(path) - 1:
                    next_node_id = path[i+1]
                    relation_type = self.G[node_id][next_node_id]['type']
                    entity_path.append((entity, relation_type))
                else:
                    # Last entity
                    entity_path.append((entity, None))
                    
            entity_paths.append(entity_path)
            
        return entity_paths
    
    def search_entities(self, 
                      query: str, 
                      max_results: int = 10) -> List[Tuple[str, str, float]]:
        """
        Search for entities matching a query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of (entity, type, relevance_score) tuples
        """
        if not query or not self.nodes:
            return []
            
        # Normalize query
        query = query.lower().strip()
        
        # Find entities that match the query
        matches = []
        
        for entity, node_id in self.nodes.items():
            entity_lower = entity.lower()
            
            # Calculate match score
            score = 0.0
            
            # Exact match
            if entity_lower == query:
                score = 1.0
            # Contains query
            elif query in entity_lower:
                score = 0.8
            # Query contains entity
            elif entity_lower in query:
                score = 0.6
            # Word overlap
            else:
                # Check for word overlap
                entity_words = set(entity_lower.split())
                query_words = set(query.split())
                overlap = entity_words.intersection(query_words)
                
                if overlap:
                    score = 0.4 * len(overlap) / max(len(entity_words), len(query_words))
            
            # Include if score is significant
            if score > 0.1:
                entity_type = self.G.nodes[node_id]['type']
                weight = self.G.nodes[node_id]['weight']
                
                # Boost score by entity weight
                final_score = score * (1 + 0.2 * min(1.0, weight / 10))
                
                matches.append((entity, entity_type, final_score))
        
        # Sort by score and return top results
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:max_results]
    
    def visualize(self, 
                output_path: str = None, 
                width: int = 1200, 
                height: int = 900,
                layout: str = 'spring',
                show_labels: bool = True,
                edge_threshold: float = 0.2,
                highlight_entities: List[str] = None) -> str:
        """
        Visualize the knowledge graph.
        
        Args:
            output_path: Path to save the visualization (None = auto-generate)
            width: Width in pixels
            height: Height in pixels
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
            show_labels: Whether to show node labels
            edge_threshold: Minimum edge weight to show
            highlight_entities: List of entities to highlight
            
        Returns:
            Path to the generated visualization file
        """
        if not self.G.number_of_nodes():
            logger.warning("No graph to visualize")
            return None
            
        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time())
            output_path = os.path.join(
                self.output_dir, 
                f"knowledge_graph_{timestamp}.png"
            )
        
        # Create figure
        plt.figure(figsize=(width/100, height/100), dpi=100)
        
        # Prepare graph for visualization
        viz_graph = self.G.copy()
        
        # Remove weak edges
        for u, v, data in list(viz_graph.edges(data=True)):
            if data['weight'] < edge_threshold:
                viz_graph.remove_edge(u, v)
        
        # Calculate node sizes based on weights
        node_weights = [data['weight'] for _, data in viz_graph.nodes(data=True)]
        max_weight = max(node_weights) if node_weights else 1
        min_weight = min(node_weights) if node_weights else 1
        
        # Normalize sizes between 100 and 2000
        node_sizes = [
            100 + 1900 * ((w - min_weight) / (max_weight - min_weight + 0.1))
            for w in node_weights
        ]
        
        # Prepare node colors based on entity types
        node_colors = []
        for _, data in viz_graph.nodes(data=True):
            entity_type = data.get('type', 'DEFAULT')
            color = self.ENTITY_COLORS.get(entity_type, self.ENTITY_COLORS['DEFAULT'])
            node_colors.append(color)
        
        # Prepare edge weights for width
        edge_weights = [data['weight'] for _, _, data in viz_graph.edges(data=True)]
        max_edge_weight = max(edge_weights) if edge_weights else 1
        min_edge_weight = min(edge_weights) if edge_weights else edge_threshold
        
        # Normalize edge widths between 1 and 5
        edge_widths = [
            1 + 4 * ((w - min_edge_weight) / (max_edge_weight - min_edge_weight + 0.1))
            for w in edge_weights
        ]
        
        # Determine positions using the specified layout
        if layout == 'circular':
            pos = nx.circular_layout(viz_graph)
        elif layout == 'kamada_kawai':
            try:
                pos = nx.kamada_kawai_layout(viz_graph)
            except Exception:
                logger.warning("Kamada-Kawai layout failed, falling back to spring layout")
                pos = nx.spring_layout(viz_graph, k=0.3, iterations=50, seed=42)
        elif layout == 'spectral':
            try:
                pos = nx.spectral_layout(viz_graph)
            except Exception:
                logger.warning("Spectral layout failed, falling back to spring layout")
                pos = nx.spring_layout(viz_graph, k=0.3, iterations=50, seed=42)
        else:
            # Default to spring layout
            pos = nx.spring_layout(viz_graph, k=0.3, iterations=50, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(
            viz_graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            viz_graph, pos,
            width=edge_widths,
            edge_color='gray',
            alpha=0.6
        )
        
        # Draw labels if requested
        if show_labels:
            # Get labels
            labels = {}
            for node, data in viz_graph.nodes(data=True):
                labels[node] = data.get('label', str(node))
                
            # Adjust label size based on node weight
            font_sizes = {}
            for node, data in viz_graph.nodes(data=True):
                weight = data['weight']
                # Scale font size from 8 to 14 based on weight
                font_sizes[node] = 8 + 6 * ((weight - min_weight) / (max_weight - min_weight + 0.1))
            
            # Draw labels in multiple calls based on font size for better visibility
            for size in range(8, 15):
                nodes_with_size = [n for n in viz_graph.nodes() if int(font_sizes[n]) == size]
                if not nodes_with_size:
                    continue
                    
                node_labels = {n: labels[n] for n in nodes_with_size}
                nx.draw_networkx_labels(
                    viz_graph, pos,
                    labels=node_labels,
                    font_size=size,
                    font_family='sans-serif',
                    font_weight='bold',
                    alpha=0.7
                )
        
        # Highlight entities if specified
        if highlight_entities:
            highlight_nodes = []
            for entity in highlight_entities:
                if entity in self.nodes and self.nodes[entity] in viz_graph:
                    highlight_nodes.append(self.nodes[entity])
                    
            if highlight_nodes:
                nx.draw_networkx_nodes(
                    viz_graph, pos,
                    nodelist=highlight_nodes,
                    node_size=[node_sizes[i] * 1.5 for i in highlight_nodes],
                    node_color='red',
                    alpha=0.9
                )
        
        # Add title and legend
        plt.title(f"Knowledge Graph ({viz_graph.number_of_nodes()} entities, {viz_graph.number_of_edges()} relationships)")
        
        # Create legend for entity types
        legend_elements = []
        for entity_type, color in self.ENTITY_COLORS.items():
            if any(data.get('type') == entity_type for _, data in viz_graph.nodes(data=True)):
                from matplotlib.patches import Patch
                legend_elements.append(
                    Patch(facecolor=color, edgecolor='gray',
                          label=entity_type)
                )
                
        if legend_elements:
            plt.legend(handles=legend_elements, loc='lower right')
        
        # Remove axis
        plt.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
        return output_path
    
    def export_graph(self, format: str = 'json', output_path: str = None) -> Optional[str]:
        """
        Export the knowledge graph to various formats.
        
        Args:
            format: Export format ('json', 'gexf', 'graphml', 'cytoscape')
            output_path: Path to save the export (None = auto-generate)
            
        Returns:
            Path to the exported file or None if export failed
        """
        if not self.G.number_of_nodes():
            logger.warning("No graph to export")
            return None
            
        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time())
            output_path = os.path.join(
                self.output_dir, 
                f"knowledge_graph_{timestamp}.{format}"
            )
        
        try:
            if format == 'json':
                # Export as JSON
                data = {
                    "metadata": self.graph_metadata,
                    "nodes": [],
                    "edges": []
                }
                
                # Add nodes
                for node_id, node_data in self.G.nodes(data=True):
                    node_entry = dict(node_data)
                    
                    # Convert sources to list for JSON serialization
                    if 'sources' in node_entry and isinstance(node_entry['sources'], set):
                        node_entry['sources'] = list(node_entry['sources'])
                        
                    # Add node ID
                    node_entry['id'] = str(node_id)
                    
                    data["nodes"].append(node_entry)
                
                # Add edges
                for source, target, edge_data in self.G.edges(data=True):
                    edge_entry = dict(edge_data)
                    edge_entry['source'] = str(source)
                    edge_entry['target'] = str(target)
                    data["edges"].append(edge_entry)
                
                # Write to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    
            elif format == 'gexf':
                # Export as GEXF (for Gephi)
                nx.write_gexf(self.G, output_path)
                
            elif format == 'graphml':
                # Export as GraphML
                nx.write_graphml(self.G, output_path)
                
            elif format == 'cytoscape':
                # Export for Cytoscape
                cytoscape_data = {
                    "elements": {
                        "nodes": [],
                        "edges": []
                    }
                }
                
                # Add nodes
                for node_id, node_data in self.G.nodes(data=True):
                    node_entry = {
                        "data": {
                            "id": str(node_id),
                            "label": node_data.get('label', str(node_id)),
                            "type": node_data.get('type', 'DEFAULT'),
                            "weight": node_data.get('weight', 1)
                        }
                    }
                    cytoscape_data["elements"]["nodes"].append(node_entry)
                
                # Add edges
                edge_id = 0
                for source, target, edge_data in self.G.edges(data=True):
                    edge_entry = {
                        "data": {
                            "id": f"e{edge_id}",
                            "source": str(source),
                            "target": str(target),
                            "weight": edge_data.get('weight', 1),
                            "type": edge_data.get('type', 'generic')
                        }
                    }
                    cytoscape_data["elements"]["edges"].append(edge_entry)
                    edge_id += 1
                
                # Write to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(cytoscape_data, f, indent=2)
                    
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
            logger.info(f"Graph exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting graph: {e}")
            return None
    
    def save(self, filepath: str = None) -> Optional[str]:
        """
        Save the knowledge graph to a file for later loading.
        
        Args:
            filepath: Path to save the graph (None = auto-generate)
            
        Returns:
            Path to the saved file or None if save failed
        """
        # Generate filepath if not provided
        if not filepath:
            timestamp = int(time.time())
            filepath = os.path.join(self.output_dir, f"knowledge_graph_{timestamp}.graphdat")
            
        try:
            # Prepare data for saving
            save_data = {
                "nodes": self.nodes,
                "edges": self.edges,
                "entity_counts": dict(self.entity_counts),
                "entity_types": self.entity_types,
                "relation_types": self.relation_types,
                "source_texts": list(self.source_texts),
                "graph_metadata": self.graph_metadata
            }
            
            # Convert graph to adjacency data
            graph_data = []
            for node_id, node_data in self.G.nodes(data=True):
                # Convert node data, ensuring sources is serializable
                node_data_copy = dict(node_data)
                if 'sources' in node_data_copy:
                    node_data_copy['sources'] = list(node_data_copy['sources'])
                    
                node_entry = {
                    'id': node_id,
                    'data': node_data_copy,
                    'edges': []
                }
                
                # Add edges
                for _, neighbor, edge_data in self.G.edges(data=True, nbunch=[node_id]):
                    if neighbor != node_id:  # Skip self-loops
                        node_entry['edges'].append({
                            'target': neighbor,
                            'data': dict(edge_data)
                        })
                        
                graph_data.append(node_entry)
                
            save_data['graph_data'] = graph_data
            
            # Save to file with thread safety
            with self.file_lock:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2)
                    
            logger.info(f"Graph saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            return None
    
    @classmethod
    def load(cls, filepath: str, output_dir: str = None) -> Optional['KnowledgeGraph']:
        """
        Load a knowledge graph from a saved file.
        
        Args:
            filepath: Path to the saved graph file
            output_dir: Output directory for the loaded graph (None = use saved value)
            
        Returns:
            Loaded KnowledgeGraph instance or None if load failed
        """
        try:
            # Load data from file
            with open(filepath, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
                
            # Create new instance
            graph = cls(output_dir=output_dir or os.path.dirname(filepath))
            
            # Restore basic data
            graph.nodes = save_data["nodes"]
            graph.edges = save_data["edges"]
            graph.entity_counts = Counter(save_data["entity_counts"])
            graph.entity_types = save_data["entity_types"]
            graph.relation_types = save_data["relation_types"]
            graph.source_texts = set(save_data["source_texts"])
            graph.graph_metadata = save_data["graph_metadata"]
            
            # Rebuild graph
            graph.G = nx.Graph()
            
            # Add nodes
            for node_entry in save_data["graph_data"]:
                node_id = node_entry['id']
                node_data = node_entry['data']
                
                # Convert sources back to set if present
                if 'sources' in node_data and isinstance(node_data['sources'], list):
                    node_data['sources'] = set(node_data['sources'])
                    
                graph.G.add_node(node_id, **node_data)
                
                # Add edges
                for edge in node_entry['edges']:
                    target = edge['target']
                    edge_data = edge['data']
                    
                    # Only add each edge once
                    if not graph.G.has_edge(node_id, target):
                        graph.G.add_edge(node_id, target, **edge_data)
            
            logger.info(f"Loaded graph with {graph.G.number_of_nodes()} nodes and {graph.G.number_of_edges()} edges")
            return graph
            
        except Exception as e:
            logger.error(f"Error loading graph from {filepath}: {e}")
            return None
    
    def generate_html_visualization(self, 
                                 output_path: str = None, 
                                 title: str = "Knowledge Graph Visualization",
                                 include_search: bool = True,
                                 width: int = 1000,
                                 height: int = 700) -> Optional[str]:
        """
        Generate interactive HTML visualization of the knowledge graph.
        
        Args:
            output_path: Path to save the HTML file (None = auto-generate)
            title: Title for the visualization
            include_search: Whether to include search functionality
            width: Width of the visualization in pixels
            height: Height of the visualization in pixels
            
        Returns:
            Path to the generated HTML file or None if generation failed
        """
        if not self.G.number_of_nodes():
            logger.warning("No graph to visualize")
            return None
            
        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time())
            output_path = os.path.join(
                self.output_dir, 
                f"knowledge_graph_{timestamp}.html"
            )
            
        try:
            # Prepare data for visualization
            vis_data = {
                "nodes": [],
                "edges": []
            }
            
            # Add nodes
            for node_id, node_data in self.G.nodes(data=True):
                # Get entity from node mapping
                entity = None
                for e, nid in self.nodes.items():
                    if nid == node_id:
                        entity = e
                        break
                        
                if not entity:
                    entity = str(node_id)
                    
                # Get node attributes
                label = node_data.get('label', entity)
                node_type = node_data.get('type', 'DEFAULT')
                weight = node_data.get('weight', 1)
                
                # Set color based on entity type
                color = self.ENTITY_COLORS.get(node_type, self.ENTITY_COLORS['DEFAULT'])
                
                # Scale size based on weight (10-50)
                max_weight = max(d.get('weight', 1) for _, d in self.G.nodes(data=True))
                size = 10 + 40 * (weight / max_weight)
                
                vis_data["nodes"].append({
                    "id": str(node_id),
                    "label": label,
                    "title": f"{label} ({node_type})<br/>Weight: {weight}",
                    "type": node_type,
                    "color": color,
                    "size": size,
                    "weight": weight
                })
            
            # Add edges
            for source, target, edge_data in self.G.edges(data=True):
                edge_type = edge_data.get('type', 'generic')
                weight = edge_data.get('weight', 1)
                
                # Scale width based on weight (1-8)
                max_edge_weight = max(d.get('weight', 1) for _, _, d in self.G.edges(data=True))
                width = 1 + 7 * (weight / max_edge_weight)
                
                vis_data["edges"].append({
                    "from": str(source),
                    "to": str(target),
                    "title": f"Type: {edge_type}<br/>Weight: {weight:.2f}",
                    "value": weight,
                    "width": width,
                    "type": edge_type
                })
                
            # Generate unique groups based on entity types
            entity_types = set(node.get('type', 'DEFAULT') for node in vis_data["nodes"])
            
            # Generate HTML
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{title}</title>
                <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/standalone/umd/vis-network.min.js"></script>
                <style type="text/css">
                    body, html {{
                        margin: 0;
                        padding: 0;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    }}
                    
                    #container {{
                        display: flex;
                        flex-direction: column;
                        height: 100vh;
                        overflow: hidden;
                    }}
                    
                    #header {{
                        background-color: #f5f5f5;
                        border-bottom: 1px solid #ddd;
                        padding: 10px 20px;
                    }}
                    
                    #controls {{
                        display: flex;
                        align-items: center;
                        gap: 20px;
                    }}
                    
                    #search {{
                        flex: 1;
                        max-width: 300px;
                        padding: 8px 12px;
                        border: 1px solid #ccc;
                        border-radius: 4px;
                    }}
                    
                    #search-results {{
                        display: none;
                        position: absolute;
                        background: white;
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        max-height: 200px;
                        overflow-y: auto;
                        z-index: 100;
                        width: 300px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    
                    #search-results div {{
                        padding: 8px 12px;
                        border-bottom: 1px solid #eee;
                        cursor: pointer;
                    }}
                    
                    #search-results div:hover {{
                        background-color: #f5f5f5;
                    }}
                    
                    #visualization {{
                        flex: 1;
                        position: relative;
                    }}
                    
                    #network {{
                        width: 100%;
                        height: 100%;
                    }}
                    
                    #legend {{
                        position: absolute;
                        bottom: 20px;
                        right: 20px;
                        background-color: rgba(255, 255, 255, 0.9);
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        padding: 10px;
                        z-index: 10;
                    }}
                    
                    .legend-item {{
                        display: flex;
                        align-items: center;
                        margin-bottom: 5px;
                    }}
                    
                    .legend-color {{
                        width: 15px;
                        height: 15px;
                        margin-right: 5px;
                        border-radius: 50%;
                    }}
                    
                    #details {{
                        position: absolute;
                        top: 20px;
                        left: 20px;
                        background-color: rgba(255, 255, 255, 0.9);
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        padding: 15px;
                        z-index: 10;
                        max-width: 300px;
                        display: none;
                    }}
                    
                    .button {{
                        background-color: #4CAF50;
                        border: none;
                        color: white;
                        padding: 8px 16px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 14px;
                        margin: 4px 2px;
                        cursor: pointer;
                        border-radius: 4px;
                    }}
                    
                    .button:hover {{
                        background-color: #45a049;
                    }}
                    
                    select {{
                        padding: 8px 12px;
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        background-color: white;
                    }}
                </style>
            </head>
            <body>
                <div id="container">
                    <div id="header">
                        <h2>{title}</h2>
                        <div id="controls">
                            {search_control}
                            <select id="layout-selector" onchange="changeLayout()">
                                <option value="standard">Standard Layout</option>
                                <option value="hierarchical">Hierarchical Layout</option>
                                <option value="physics">Physics-based Layout</option>
                            </select>
                            <button class="button" onclick="resetView()">Reset View</button>
                            <span><strong>Nodes:</strong> {node_count} | <strong>Edges:</strong> {edge_count}</span>
                        </div>
                        <div id="search-results"></div>
                    </div>
                    <div id="visualization">
                        <div id="network"></div>
                        <div id="legend">
                            <h3>Entity Types</h3>
                            <div id="legend-content"></div>
                        </div>
                        <div id="details">
                            <h3 id="detail-title">Node Details</h3>
                            <div id="detail-content"></div>
                            <button class="button" onclick="hideDetails()">Close</button>
                        </div>
                    </div>
                </div>

                <script type="text/javascript">
                    // Graph data
                    const nodes = new vis.DataSet({data_nodes});
                    const edges = new vis.DataSet({data_edges});

                    // Create a network
                    const container = document.getElementById('network');
                    const data = {{
                        nodes: nodes,
                        edges: edges
                    }};
                    
                    // Configuration
                    const options = {{
                        nodes: {{
                            shape: 'dot',
                            scaling: {{
                                label: {{
                                    min: 12,
                                    max: 24
                                }}
                            }},
                            font: {{
                                size: 14
                            }}
                        }},
                        edges: {{
                            smooth: {{
                                type: 'continuous',
                                forceDirection: 'none'
                            }},
                            color: {{
                                inherit: false,
                                color: '#bbbbbb',
                                opacity: 0.7
                            }}
                        }},
                        physics: {{
                            stabilization: true,
                            barnesHut: {{
                                gravitationalConstant: -5000,
                                centralGravity: 0.3,
                                springLength: 150,
                                springConstant: 0.05,
                                damping: 0.09
                            }}
                        }},
                        interaction: {{
                            hover: true,
                            tooltipDelay: 300,
                            navigationButtons: true,
                            keyboard: true
                        }}
                    }};
                    
                    // Create network
                    const network = new vis.Network(container, data, options);
                    
                    // Event listeners
                    network.on("click", function(params) {{
                        if (params.nodes.length > 0) {{
                            const nodeId = params.nodes[0];
                            const node = nodes.get(nodeId);
                            showNodeDetails(node);
                        }} else {{
                            hideDetails();
                        }}
                    }});
                    
                    // Create legend
                    const typeColors = {type_colors};
                    const legendContent = document.getElementById('legend-content');
                    
                    for (const [type, color] of Object.entries(typeColors)) {{
                        const item = document.createElement('div');
                        item.className = 'legend-item';
                        
                        const colorBox = document.createElement('div');
                        colorBox.className = 'legend-color';
                        colorBox.style.backgroundColor = color;
                        
                        const label = document.createElement('span');
                        label.textContent = type;
                        
                        item.appendChild(colorBox);
                        item.appendChild(label);
                        legendContent.appendChild(item);
                    }}
                    
                    // Functions
                    function showNodeDetails(node) {{
                        const detailsDiv = document.getElementById('details');
                        const titleDiv = document.getElementById('detail-title');
                        const contentDiv = document.getElementById('detail-content');
                        
                        titleDiv.textContent = node.label;
                        
                        let content = `<p><strong>Type:</strong> ${{node.type}}</p>`;
                        content += `<p><strong>Weight:</strong> ${{node.weight}}</p>`;
                        
                        // Find connected nodes
                        const connectedNodes = network.getConnectedNodes(node.id);
                        
                        if (connectedNodes.length > 0) {{
                            content += `<p><strong>Connected to:</strong></p><ul>`;
                            
                            // Get a maximum of 10 connections
                            const maxToShow = Math.min(10, connectedNodes.length);
                            
                            for (let i = 0; i < maxToShow; i++) {{
                                const connectedNode = nodes.get(connectedNodes[i]);
                                content += `<li>${{connectedNode.label}} (${{connectedNode.type}})</li>`;
                            }}
                            
                            if (connectedNodes.length > maxToShow) {{
                                content += `<li>...and ${{connectedNodes.length - maxToShow}} more</li>`;
                            }}
                            
                            content += `</ul>`;
                        }}
                        
                        contentDiv.innerHTML = content;
                        detailsDiv.style.display = 'block';
                    }}
                    
                    function hideDetails() {{
                        document.getElementById('details').style.display = 'none';
                    }}
                    
                    function resetView() {{
                        network.fit({{
                            animation: {{
                                duration: 1000,
                                easingFunction: 'easeInOutQuad'
                            }}
                        }});
                    }}
                    
                    function changeLayout() {{
                        const layoutType = document.getElementById('layout-selector').value;
                        
                        let newOptions = {{...options}};
                        
                        if (layoutType === 'hierarchical') {{
                            newOptions.layout = {{
                                hierarchical: {{
                                    direction: 'UD',
                                    sortMethod: 'directed',
                                    nodeSpacing: 150,
                                    treeSpacing: 200
                                }}
                            }};
                        }} else if (layoutType === 'physics') {{
                            newOptions.physics = {{
                                forceAtlas2Based: {{
                                    gravitationalConstant: -50,
                                    centralGravity: 0.01,
                                    springLength: 100,
                                    springConstant: 0.08,
                                    damping: 0.4
                                }},
                                maxVelocity: 50,
                                minVelocity: 0.1,
                                solver: 'forceAtlas2Based',
                                timestep: 0.35,
                                stabilization: {{
                                    enabled: true,
                                    iterations: 1000,
                                    updateInterval: 25
                                }}
                            }};
                        }} else {{
                            // Standard
                            newOptions.layout = {{}};
                            newOptions.physics = {{
                                stabilization: true,
                                barnesHut: {{
                                    gravitationalConstant: -5000,
                                    centralGravity: 0.3,
                                    springLength: 150,
                                    springConstant: 0.05,
                                    damping: 0.09
                                }}
                            }};
                        }}
                        
                        network.setOptions(newOptions);
                    }}
                    
                    {search_function}
                    
                    // Initialize
                    resetView();
                </script>
            </body>
            </html>
            """
            
            # Replace placeholders
            search_html = ""
            search_js = ""
            
            if include_search:
                search_html = '<input id="search" type="text" placeholder="Search nodes..." onkeyup="searchNodes()">'
                search_js = """
                    function searchNodes() {
                        const query = document.getElementById('search').value.toLowerCase();
                        const resultsDiv = document.getElementById('search-results');
                        
                        if (query.length < 2) {
                            resultsDiv.style.display = 'none';
                            return;
                        }
                        
                        // Find matching nodes
                        const matches = [];
                        nodes.forEach(node => {
                            if (node.label.toLowerCase().includes(query)) {
                                matches.push(node);
                            }
                        });
                        
                        // Display results
                        if (matches.length > 0) {
                            resultsDiv.innerHTML = '';
                            
                            // Get a maximum of 10 matches
                            const maxToShow = Math.min(10, matches.length);
                            
                            for (let i = 0; i < maxToShow; i++) {
                                const node = matches[i];
                                const resultItem = document.createElement('div');
                                resultItem.textContent = `${node.label} (${node.type})`;
                                resultItem.onclick = function() {
                                    network.focus(node.id, {
                                        scale: 1.5,
                                        animation: {
                                            duration: 1000,
                                            easingFunction: 'easeInOutQuad'
                                        }
                                    });
                                    network.selectNodes([node.id]);
                                    showNodeDetails(node);
                                    resultsDiv.style.display = 'none';
                                    document.getElementById('search').value = '';
                                };
                                resultsDiv.appendChild(resultItem);
                            }
                            
                            // Position the results div
                            const searchInput = document.getElementById('search');
                            const rect = searchInput.getBoundingClientRect();
                            resultsDiv.style.top = `${rect.bottom}px`;
                            resultsDiv.style.left = `${rect.left}px`;
                            
                            resultsDiv.style.display = 'block';
                        } else {
                            resultsDiv.style.display = 'none';
                        }
                    }
                    
                    // Hide search results when clicking elsewhere
                    document.addEventListener('click', function(event) {
                        if (event.target.id !== 'search' && !event.target.closest('#search-results')) {
                            document.getElementById('search-results').style.display = 'none';
                        }
                    });
                """
            
            # Generate color mapping
            type_colors_json = {}
            for entity_type in entity_types:
                type_colors_json[entity_type] = self.ENTITY_COLORS.get(
                    entity_type, self.ENTITY_COLORS['DEFAULT'])
            
            html_content = html_template.format(
                title=title,
                search_control=search_html,
                search_function=search_js,
                data_nodes=json.dumps(vis_data["nodes"]),
                data_edges=json.dumps(vis_data["edges"]),
                type_colors=json.dumps(type_colors_json),
                node_count=self.G.number_of_nodes(),
                edge_count=self.G.number_of_edges()
            )
            
            # Write HTML file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"HTML visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML visualization: {e}")
            return None
    
    def _normalize_entity(self, entity_text: str) -> str:
        """
        Normalize entity text for consistency.
        
        Args:
            entity_text: Raw entity text
            
        Returns:
            Normalized entity text
        """
        if not entity_text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', entity_text).strip()
        
        # Remove special characters except alphanumeric, space, and underscore
        text = re.sub(r'[^\w\s]', '', text)
        
        # Limit length
        if len(text) > 100:
            text = text[:97] + '...'
            
        return text


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage
    kg = KnowledgeGraph(output_dir="output")
    
    # Sample entities and relationships
    entities = [
        ("Machine Learning", "CONCEPT", 10),
        ("Neural Networks", "CONCEPT", 8),
        ("Deep Learning", "CONCEPT", 7),
        ("Computer Vision", "CONCEPT", 5),
        ("Natural Language Processing", "CONCEPT", 6),
        ("Geoffrey Hinton", "PERSON", 4),
        ("Yann LeCun", "PERSON", 3),
        ("Yoshua Bengio", "PERSON", 3),
        ("TensorFlow", "PRODUCT", 4),
        ("PyTorch", "PRODUCT", 4)
    ]
    
    relationships = [
        {"source": "Neural Networks", "target": "Deep Learning", "type": "is_related_to", "weight": 0.9},
        {"source": "Deep Learning", "target": "Machine Learning", "type": "is_part_of", "weight": 0.8},
        {"source": "Geoffrey Hinton", "target": "Neural Networks", "type": "researches", "weight": 0.7},
        {"source": "Yann LeCun", "target": "Deep Learning", "type": "researches", "weight": 0.7},
        {"source": "Yoshua Bengio", "target": "Natural Language Processing", "type": "researches", "weight": 0.6},
        {"source": "TensorFlow", "target": "Machine Learning", "type": "supports", "weight": 0.5},
        {"source": "PyTorch", "target": "Deep Learning", "type": "supports", "weight": 0.5},
        {"source": "Computer Vision", "target": "Deep Learning", "type": "uses", "weight": 0.6}
    ]
    
    # Build graph
    kg.build_from_entities(entities)
    kg.build_from_relationships(relationships)
    
    # Add topics
    topics = [
        {
            "id": 1,
            "label": "Neural Techniques",
            "terms": [
                {"term": "Neural", "weight": 0.9},
                {"term": "Network", "weight": 0.8},
                {"term": "Deep", "weight": 0.7},
                {"term": "Learning", "weight": 0.6},
                {"term": "Backpropagation", "weight": 0.5}
            ]
        },
        {
            "id": 2,
            "label": "Applications",
            "terms": [
                {"term": "Vision", "weight": 0.8},
                {"term": "Language", "weight": 0.7},
                {"term": "Processing", "weight": 0.6},
                {"term": "Recognition", "weight": 0.5},
                {"term": "Classification", "weight": 0.4}
            ]
        }
    ]
    
    kg.add_topics(topics)
    
    # Visualize
    kg.visualize(output_path="output/example_graph.png")
    
    # Generate HTML visualization
    kg.generate_html_visualization(output_path="output/example_graph.html")
    
    # Export graph
    kg.export_graph(format='json', output_path="output/example_graph.json")
    
    # Find central entities
    central_entities = kg.get_central_entities(top_n=5)
    print("Central entities:", central_entities)
    
    # Find communities
    communities = kg.get_communities()
    print("Communities:", communities)
    
    # Search for entities
    search_results = kg.search_entities("neural")
    print("Search results for 'neural':", search_results)
