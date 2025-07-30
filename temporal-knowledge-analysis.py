"""
temporal_analysis.py - Temporal analysis of knowledge evolution

This module provides tools for analyzing how concepts, topics, and relationships
evolve over time in document collections, enabling trend detection and
knowledge evolution tracking.

Design principles:
- Algorithmic elegance (Dijkstra approach)
- Performance optimization (Knuth efficiency)
- Clean code structure (Torvalds/van Rossum style)
- Comprehensive documentation (Stroustrup methodology)
- Secure implementation (Schneier principles)

Author: ototao
License: Apache License 2.0
"""

import logging
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from collections import defaultdict, Counter
import json
import os
import re
import pandas as pd
from threading import Lock
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import SUM components
from knowledge_graph import KnowledgeGraph
from Models.topic_modeling import TopicModeler

# Configure logging
logger = logging.getLogger(__name__)


class TemporalAnalysis:
    """
    Analyze temporal evolution of knowledge in document collections.
    
    This class provides tools to track how concepts, topics, and relationships
    evolve over time, enabling trend detection and knowledge evolution tracking.
    """
    
    def __init__(self, 
                output_dir: str = 'output',
                time_granularity: str = 'month',
                min_documents: int = 2,
                smoothing_window: int = 1):
        """
        Initialize the temporal analysis engine.
        
        Args:
            output_dir: Directory to save visualizations and data
            time_granularity: Time bucketization level ('day', 'week', 'month', 'year')
            min_documents: Minimum number of documents required for analysis
            smoothing_window: Window size for trend smoothing (0 = no smoothing)
        """
        self.output_dir = output_dir
        self.time_granularity = time_granularity
        self.min_documents = min_documents
        self.smoothing_window = max(0, smoothing_window)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data structures
        self.documents = []
        self.document_dates = []
        self.document_metadata = []
        self.time_periods = []
        self.period_documents = defaultdict(list)
        self.period_indices = defaultdict(list)
        
        # Thread safety for file operations
        self.file_lock = Lock()
        
        logger.info(f"TemporalAnalysis initialized with time_granularity='{time_granularity}'")
    
    def add_document(self, 
                   document: str, 
                   date_str: str, 
                   date_format: str = '%Y-%m-%d',
                   metadata: Dict = None) -> bool:
        """
        Add a document with timestamp to the analysis.
        
        Args:
            document: Document text
            date_str: Date string
            date_format: Date format string
            metadata: Optional metadata dictionary
            
        Returns:
            True if document was successfully added
        """
        try:
            # Parse date
            date = datetime.strptime(date_str, date_format)
            
            # Add document
            self.documents.append(document)
            self.document_dates.append(date)
            self.document_metadata.append(metadata or {})
            
            # Clear cached time periods (will be recalculated on next analysis)
            self.time_periods = []
            self.period_documents = defaultdict(list)
            self.period_indices = defaultdict(list)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def add_documents(self, 
                    documents: List[str],
                    dates: List[str],
                    date_format: str = '%Y-%m-%d',
                    metadata_list: List[Dict] = None) -> int:
        """
        Add multiple documents with timestamps to the analysis.
        
        Args:
            documents: List of document texts
            dates: List of date strings
            date_format: Date format string
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            Number of documents successfully added
        """
        if len(documents) != len(dates):
            logger.error("Documents and dates lists must have the same length")
            return 0
            
        if metadata_list and len(metadata_list) != len(documents):
            logger.error("Metadata list must have the same length as documents")
            return 0
            
        count = 0
        for i, (doc, date_str) in enumerate(zip(documents, dates)):
            metadata = metadata_list[i] if metadata_list else None
            if self.add_document(doc, date_str, date_format, metadata):
                count += 1
                
        return count
    
    def load_from_json(self, filepath: str) -> int:
        """
        Load documents from a JSON file.
        
        Expected format:
        {
            "documents": [
                {
                    "text": "Document text",
                    "date": "2023-01-01",
                    "metadata": { ... }
                },
                ...
            ],
            "date_format": "%Y-%m-%d"
        }
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Number of documents loaded
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, dict) or 'documents' not in data:
                logger.error("Invalid JSON format: missing 'documents' field")
                return 0
                
            documents = data['documents']
            date_format = data.get('date_format', '%Y-%m-%d')
            
            count = 0
            for doc_data in documents:
                if not isinstance(doc_data, dict):
                    continue
                    
                text = doc_data.get('text')
                date = doc_data.get('date')
                metadata = doc_data.get('metadata')
                
                if text and date:
                    if self.add_document(text, date, date_format, metadata):
                        count += 1
                        
            logger.info(f"Loaded {count} documents from {filepath}")
            return count
            
        except Exception as e:
            logger.error(f"Error loading from JSON: {e}")
            return 0
    
    def _organize_by_time_periods(self) -> bool:
        """
        Organize documents into time periods based on granularity.
        
        Returns:
            True if successful
        """
        if not self.documents:
            logger.warning("No documents to organize")
            return False
            
        if self.time_periods:
            # Already organized
            return True
            
        # Clear existing data
        self.time_periods = []
        self.period_documents = defaultdict(list)
        self.period_indices = defaultdict(list)
        
        # Define period format based on granularity
        if self.time_granularity == 'day':
            period_format = '%Y-%m-%d'
        elif self.time_granularity == 'week':
            # For weeks, we'll use the first day of the week
            pass  # Special handling below
        elif self.time_granularity == 'month':
            period_format = '%Y-%m'
        elif self.time_granularity == 'year':
            period_format = '%Y'
        else:
            logger.error(f"Invalid time granularity: {self.time_granularity}")
            return False
            
        # Group documents by period
        for i, date in enumerate(self.document_dates):
            if self.time_granularity == 'week':
                # Calculate the first day of the week (Monday)
                start_of_week = date - timedelta(days=date.weekday())
                period = start_of_week.strftime('%Y-%m-%d')
            else:
                period = date.strftime(period_format)
                
            self.period_documents[period].append(self.documents[i])
            self.period_indices[period].append(i)
            
        # Sort periods chronologically
        self.time_periods = sorted(self.period_documents.keys())
        
        # Filter periods with too few documents
        if self.min_documents > 1:
            filtered_periods = []
            for period in self.time_periods:
                if len(self.period_documents[period]) >= self.min_documents:
                    filtered_periods.append(period)
                else:
                    logger.info(f"Skipping period {period} with only {len(self.period_documents[period])} documents")
                    
            self.time_periods = filtered_periods
            
        if not self.time_periods:
            logger.warning("No time periods with sufficient documents")
            return False
            
        logger.info(f"Organized documents into {len(self.time_periods)} time periods")
        return True
    
    def analyze_topic_evolution(self, 
                             num_topics: int = 5,
                             top_n_terms: int = 10,
                             algorithm: str = 'lda') -> Dict:
        """
        Analyze how topics evolve over time periods.
        
        Args:
            num_topics: Number of topics to extract
            top_n_terms: Number of top terms per topic
            algorithm: Topic modeling algorithm ('lda', 'nmf', or 'lsa')
            
        Returns:
            Dictionary with topic evolution data
        """
        if not self._organize_by_time_periods():
            return {'error': 'Failed to organize documents by time periods'}
            
        if len(self.time_periods) < 2:
            return {'error': 'Insufficient time periods for evolution analysis'}
            
        try:
            # Initialize results structure
            results = {
                'time_periods': self.time_periods,
                'granularity': self.time_granularity,
                'topics_by_period': {},
                'topic_evolution': {},
                'topic_similarity': {},
                'trending_topics': {}
            }
            
            # Extract topics for each time period
            period_topics = {}
            
            for period in self.time_periods:
                documents = self.period_documents[period]
                
                # Create topic modeler
                topic_modeler = TopicModeler(
                    n_topics=num_topics,
                    algorithm=algorithm,
                    n_top_words=top_n_terms
                )
                
                # Fit model to documents
                topic_modeler.fit(documents)
                
                # Extract topics
                topics = topic_modeler.get_topics_summary()
                
                # Store topic data
                period_topics[period] = topics
                
                # Format for results
                formatted_topics = {}
                for topic_id, topic_data in topics.items():
                    formatted_topics[str(topic_id)] = {
                        'label': topic_data['label'],
                        'terms': list(topic_data['terms'].items()),
                        'coherence': topic_data.get('coherence')
                    }
                    
                results['topics_by_period'][period] = formatted_topics
            
            # Track topic evolution across periods
            topic_evolution = {}
            
            # For each topic in each period
            for i, period in enumerate(self.time_periods):
                current_topics = period_topics[period]
                
                for topic_id, topic_data in current_topics.items():
                    topic_key = f"{period}_topic{topic_id}"
                    topic_terms = set(topic_data['terms'].keys())
                    
                    # If this is not the first period, compare with previous
                    if i > 0:
                        prev_period = self.time_periods[i-1]
                        prev_topics = period_topics[prev_period]
                        
                        # Find most similar topic from previous period
                        max_similarity = 0
                        most_similar_topic = None
                        
                        for prev_id, prev_data in prev_topics.items():
                            prev_terms = set(prev_data['terms'].keys())
                            
                            # Calculate Jaccard similarity
                            similarity = len(topic_terms.intersection(prev_terms)) / len(topic_terms.union(prev_terms))
                            
                            if similarity > max_similarity:
                                max_similarity = similarity
                                most_similar_topic = f"{prev_period}_topic{prev_id}"
                                
                        # Store evolution data
                        if most_similar_topic and max_similarity > 0.1:
                            if most_similar_topic not in topic_evolution:
                                topic_evolution[most_similar_topic] = {
                                    'evolution': [{'period': prev_period, 'topic_id': prev_id}]
                                }
                                
                            topic_evolution[most_similar_topic]['evolution'].append({
                                'period': period,
                                'topic_id': topic_id,
                                'similarity': max_similarity
                            })
                            
                    # Initialize evolution tracking if new topic
                    if topic_key not in topic_evolution:
                        topic_evolution[topic_key] = {
                            'evolution': [{'period': period, 'topic_id': topic_id}]
                        }
            
            # Format evolution data for results
            for topic_key, data in topic_evolution.items():
                evolution = data['evolution']
                
                # Only include topics that span multiple periods
                if len(evolution) > 1:
                    # Get the first period's topic data for reference
                    first_period = evolution[0]['period']
                    first_topic_id = evolution[0]['topic_id']
                    topic_data = period_topics[first_period][first_topic_id]
                    
                    # Create evolution series
                    evolution_series = []
                    for entry in evolution:
                        entry_period = entry['period']
                        entry_topic_id = entry['topic_id']
                        entry_data = period_topics[entry_period][entry_topic_id]
                        
                        evolution_series.append({
                            'period': entry_period,
                            'topic_id': entry_topic_id,
                            'label': entry_data['label'],
                            'terms': list(entry_data['terms'].items())[:5],  # Top 5 terms only
                            'similarity': entry.get('similarity', 1.0)
                        })
                        
                    results['topic_evolution'][topic_key] = {
                        'label': topic_data['label'],
                        'evolution': evolution_series
                    }
            
            # Calculate period-to-period topic similarity
            period_similarity = {}
            
            for i in range(len(self.time_periods) - 1):
                curr_period = self.time_periods[i]
                next_period = self.time_periods[i + 1]
                
                # Get all topic terms for each period
                curr_terms = set()
                for topic in period_topics[curr_period].values():
                    curr_terms.update(topic['terms'].keys())
                    
                next_terms = set()
                for topic in period_topics[next_period].values():
                    next_terms.update(topic['terms'].keys())
                    
                # Calculate Jaccard similarity
                similarity = len(curr_terms.intersection(next_terms)) / len(curr_terms.union(next_terms))
                
                period_similarity[f"{curr_period}_to_{next_period}"] = similarity
                
            results['topic_similarity'] = period_similarity
            
            # Identify trending topics
            if len(self.time_periods) >= 3:
                trending_topics = self._identify_trending_topics(period_topics)
                results['trending_topics'] = trending_topics
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing topic evolution: {e}", exc_info=True)
            return {'error': f"Error analyzing topic evolution: {str(e)}"}
    
    def _identify_trending_topics(self, period_topics: Dict) -> Dict:
        """
        Identify trending and declining topics.
        
        Args:
            period_topics: Topics by period
            
        Returns:
            Dictionary of trending topics
        """
        # Track term frequencies across periods
        term_frequencies = defaultdict(list)
        
        for period in self.time_periods:
            # Count terms across all topics in this period
            period_terms = Counter()
            
            for topic_data in period_topics[period].values():
                for term, weight in topic_data['terms'].items():
                    period_terms[term] += weight
                    
            # Normalize frequencies
            total_weight = sum(period_terms.values())
            if total_weight > 0:
                normalized_terms = {term: weight / total_weight for term, weight in period_terms.items()}
            else:
                normalized_terms = {term: 0 for term in period_terms}
                
            # Store for each term
            for term, freq in normalized_terms.items():
                term_frequencies[term].append(freq)
                
            # Ensure all terms have values for all periods
            for term in term_frequencies:
                if term not in normalized_terms:
                    term_frequencies[term].append(0)
        
        # Calculate trends for terms with data in at least half the periods
        min_periods = max(2, len(self.time_periods) // 2)
        trending_up = []
        trending_down = []
        
        for term, frequencies in term_frequencies.items():
            # Skip terms that don't appear in enough periods
            if sum(freq > 0 for freq in frequencies) < min_periods:
                continue
                
            # Calculate trend (simple linear regression slope)
            x = np.arange(len(frequencies))
            slope, _ = np.polyfit(x, frequencies, 1)
            
            # Smooth trend if enabled
            if self.smoothing_window > 0 and len(frequencies) > self.smoothing_window:
                smooth_freq = np.convolve(
                    frequencies, 
                    np.ones(self.smoothing_window) / self.smoothing_window, 
                    mode='valid'
                )
                
                # Recalculate trend with smoothed data
                x_smooth = np.arange(len(smooth_freq))
                slope, _ = np.polyfit(x_smooth, smooth_freq, 1)
            
            # Classify trend
            if slope > 0.01:  # Threshold for trending up
                trending_up.append((term, slope, frequencies))
            elif slope < -0.01:  # Threshold for trending down
                trending_down.append((term, slope, frequencies))
        
        # Sort by trend strength
        trending_up.sort(key=lambda x: x[1], reverse=True)
        trending_down.sort(key=lambda x: x[1])
        
        # Format results
        result = {
            'trending_up': [
                {
                    'term': term,
                    'trend': float(slope),
                    'frequency': [float(f) for f in freq]
                }
                for term, slope, freq in trending_up[:10]  # Top 10
            ],
            'trending_down': [
                {
                    'term': term,
                    'trend': float(slope),
                    'frequency': [float(f) for f in freq]
                }
                for term, slope, freq in trending_down[:10]  # Top 10
            ]
        }
        
        return result
    
    def analyze_entity_evolution(self, use_spacy: bool = True) -> Dict:
        """
        Analyze how entities and their relationships evolve over time.
        
        Args:
            use_spacy: Whether to use spaCy for entity extraction
            
        Returns:
            Dictionary with entity evolution data
        """
        if not self._organize_by_time_periods():
            return {'error': 'Failed to organize documents by time periods'}
            
        try:
            # Initialize summarizer with entity recognition
            from advanced_sum import AdvancedSUM
            summarizer = AdvancedSUM(use_entities=True)
            
            # Initialize results structure
            results = {
                'time_periods': self.time_periods,
                'granularity': self.time_granularity,
                'entities_by_period': {},
                'entity_trends': {},
                'relationship_evolution': {},
                'entity_graphs': {}
            }
            
            # Extract entities for each time period
            period_entities = {}
            period_entity_counts = {}
            
            for period in self.time_periods:
                documents = self.period_documents[period]
                period_text = " ".join(documents)
                
                # Process text to extract entities
                analysis_result = summarizer.process_text(period_text)
                
                if 'entities' in analysis_result and analysis_result['entities']:
                    entities = analysis_result['entities']
                    
                    # Count and organize entities
                    entity_dict = {}
                    entity_counter = Counter()
                    
                    for entity, entity_type, count in entities:
                        entity_dict[entity] = {
                            'type': entity_type,
                            'count': count
                        }
                        entity_counter[entity] = count
                        
                    # Store data
                    period_entities[period] = entity_dict
                    period_entity_counts[period] = entity_counter
                    
                    # Format for results
                    results['entities_by_period'][period] = [
                        {'entity': entity, 'type': data['type'], 'count': data['count']}
                        for entity, data in entity_dict.items()
                    ]
                else:
                    # No entities found
                    period_entities[period] = {}
                    period_entity_counts[period] = Counter()
                    results['entities_by_period'][period] = []
            
            # Create knowledge graphs for each period
            period_graphs = {}
            
            for period in self.time_periods:
                kg = KnowledgeGraph(output_dir=self.output_dir)
                
                # Extract entities for the graph
                entity_list = [
                    (entity, data['type'], data['count'])
                    for entity, data in period_entities[period].items()
                ]
                
                if entity_list:
                    # Build graph from entities
                    kg.build_from_entities(entity_list)
                    
                    # Generate relationships from co-occurrence
                    documents = self.period_documents[period]
                    relationships = self._extract_relationships(documents, entity_list)
                    
                    if relationships:
                        kg.build_from_relationships(relationships)
                        
                    # Prune graph to keep it manageable
                    kg.prune_graph()
                    
                # Store graph
                period_graphs[period] = kg
                
                # Generate visualization for the period
                viz_path = None
                if kg.G.number_of_nodes() > 0:
                    viz_path = kg.visualize(
                        output_path=os.path.join(self.output_dir, f"entity_graph_{period}.png"),
                        show_labels=True
                    )
                    
                    # Export graph data
                    graph_data_path = kg.export_graph(
                        format='json',
                        output_path=os.path.join(self.output_dir, f"entity_graph_{period}.json")
                    )
                    
                # Add to results
                if kg.G.number_of_nodes() > 0:
                    results['entity_graphs'][period] = {
                        'nodes': kg.G.number_of_nodes(),
                        'edges': kg.G.number_of_edges(),
                        'visualization': os.path.basename(viz_path) if viz_path else None,
                        'central_entities': [
                            {'entity': entity, 'centrality': float(score)}
                            for entity, score in kg.get_central_entities(top_n=10)
                        ]
                    }
            
            # Track entity trends across periods
            all_entities = set()
            for entities in period_entity_counts.values():
                all_entities.update(entities.keys())
                
            entity_trends = {}
            
            for entity in all_entities:
                # Get counts across periods
                counts = []
                for period in self.time_periods:
                    counts.append(period_entity_counts[period].get(entity, 0))
                    
                # Calculate trend
                if sum(counts) >= len(counts):  # At least average of 1 per period
                    x = np.arange(len(counts))
                    slope, _ = np.polyfit(x, counts, 1)
                    
                    # Smooth trend if enabled
                    if self.smoothing_window > 0 and len(counts) > self.smoothing_window:
                        smooth_counts = np.convolve(
                            counts, 
                            np.ones(self.smoothing_window) / self.smoothing_window, 
                            mode='valid'
                        )
                        
                        # Recalculate trend with smoothed data
                        x_smooth = np.arange(len(smooth_counts))
                        slope, _ = np.polyfit(x_smooth, smooth_counts, 1)
                    
                    # Get entity type
                    entity_type = None
                    for period_data in period_entities.values():
                        if entity in period_data:
                            entity_type = period_data[entity]['type']
                            break
                            
                    entity_trends[entity] = {
                        'type': entity_type,
                        'counts': counts,
                        'trend': float(slope)
                    }
            
            # Sort entities by trend and add to results
            trending_up = sorted(
                [(entity, data) for entity, data in entity_trends.items() if data['trend'] > 0],
                key=lambda x: x[1]['trend'],
                reverse=True
            )
            
            trending_down = sorted(
                [(entity, data) for entity, data in entity_trends.items() if data['trend'] < 0],
                key=lambda x: x[1]['trend']
            )
            
            results['entity_trends'] = {
                'trending_up': [
                    {
                        'entity': entity,
                        'type': data['type'],
                        'counts': data['counts'],
                        'trend': data['trend']
                    }
                    for entity, data in trending_up[:20]  # Top 20
                ],
                'trending_down': [
                    {
                        'entity': entity,
                        'type': data['type'],
                        'counts': data['counts'],
                        'trend': data['trend']
                    }
                    for entity, data in trending_down[:20]  # Top 20
                ]
            }
            
            # Analyze relationship evolution
            relationship_evolution = self._analyze_relationship_evolution(period_graphs)
            results['relationship_evolution'] = relationship_evolution
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing entity evolution: {e}", exc_info=True)
            return {'error': f"Error analyzing entity evolution: {str(e)}"}
    
    def _extract_relationships(self, 
                             documents: List[str], 
                             entities: List[Tuple[str, str, int]]) -> List[Dict]:
        """
        Extract entity relationships from documents.
        
        Args:
            documents: List of documents
            entities: List of (entity, type, count) tuples
            
        Returns:
            List of relationship dictionaries
        """
        # Create entity sets for faster lookup
        entity_names = {entity[0].lower() for entity in entities}
        entity_types = {entity[0].lower(): entity[1] for entity in entities}
        
        # Create relationship index
        relationships = defaultdict(float)
        
        # Analyze documents for co-occurrence
        for doc in documents:
            # Split into sentences
            sentences = re.split(r'[.!?]+', doc)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Find entities in this sentence
                found_entities = set()
                
                for entity in entity_names:
                    if entity in sentence_lower:
                        found_entities.add(entity)
                
                # Create relationships for co-occurring entities
                for entity1 in found_entities:
                    for entity2 in found_entities:
                        if entity1 != entity2:
                            # Get entity types
                            type1 = entity_types.get(entity1, 'DEFAULT')
                            type2 = entity_types.get(entity2, 'DEFAULT')
                            
                            # Create relationship key
                            rel_key = tuple(sorted([entity1, entity2]))
                            
                            # Increment relationship strength
                            relationships[rel_key] += 1
        
        # Format relationships
        result = []
        
        for (entity1, entity2), weight in relationships.items():
            # Get entity types
            type1 = entity_types.get(entity1, 'DEFAULT')
            type2 = entity_types.get(entity2, 'DEFAULT')
            
            # Only include relationships with sufficient strength
            if weight >= 2:
                result.append({
                    'source': entity1,
                    'source_type': type1,
                    'target': entity2,
                    'target_type': type2,
                    'type': 'co-occurrence',
                    'weight': weight
                })
        
        return result
    
    def _analyze_relationship_evolution(self, period_graphs: Dict[str, KnowledgeGraph]) -> Dict:
        """
        Analyze how relationships evolve over time.
        
        Args:
            period_graphs: Dictionary of knowledge graphs by period
            
        Returns:
            Dictionary with relationship evolution data
        """
        # Track relationships across periods
        relationship_data = {}
        
        # For each pair of consecutive periods
        for i in range(len(self.time_periods) - 1):
            curr_period = self.time_periods[i]
            next_period = self.time_periods[i + 1]
            
            curr_graph = period_graphs[curr_period]
            next_graph = period_graphs[next_period]
            
            # Skip if either graph is empty
            if curr_graph.G.number_of_nodes() == 0 or next_graph.G.number_of_nodes() == 0:
                continue
                
            # Find common entities
            curr_entities = set(curr_graph.nodes.keys())
            next_entities = set(next_graph.nodes.keys())
            common_entities = curr_entities.intersection(next_entities)
            
            # Find new and disappeared entities
            new_entities = next_entities - curr_entities
            disappeared_entities = curr_entities - next_entities
            
            # Find edges for common entities
            curr_edges = {}
            next_edges = {}
            
            # Get edges in current period
            for u, v, data in curr_graph.G.edges(data=True):
                # Get entity names
                source = None
                target = None
                
                for entity, node_id in curr_graph.nodes.items():
                    if node_id == u:
                        source = entity
                    elif node_id == v:
                        target = entity
                        
                    if source and target:
                        break
                        
                if source and target:
                    edge_key = tuple(sorted([source, target]))
                    curr_edges[edge_key] = data['weight']
            
            # Get edges in next period
            for u, v, data in next_graph.G.edges(data=True):
                # Get entity names
                source = None
                target = None
                
                for entity, node_id in next_graph.nodes.items():
                    if node_id == u:
                        source = entity
                    elif node_id == v:
                        target = entity
                        
                    if source and target:
                        break
                        
                if source and target:
                    edge_key = tuple(sorted([source, target]))
                    next_edges[edge_key] = data['weight']
            
            # Find common, strengthened, weakened, new, and disappeared edges
            common_edges = set(curr_edges.keys()).intersection(set(next_edges.keys()))
            new_edges = set(next_edges.keys()) - set(curr_edges.keys())
            disappeared_edges = set(curr_edges.keys()) - set(next_edges.keys())
            
            strengthened_edges = []
            weakened_edges = []
            
            for edge in common_edges:
                curr_weight = curr_edges[edge]
                next_weight = next_edges[edge]
                
                if next_weight > curr_weight:
                    strengthened_edges.append((edge, curr_weight, next_weight))
                elif next_weight < curr_weight:
                    weakened_edges.append((edge, curr_weight, next_weight))
            
            # Store evolution data
            relationship_data[f"{curr_period}_to_{next_period}"] = {
                'common_entities': len(common_entities),
                'new_entities': [entity for entity in new_entities][:20],  # Limit to 20
                'disappeared_entities': [entity for entity in disappeared_entities][:20],  # Limit to 20
                'common_relationships': len(common_edges),
                'new_relationships': [
                    {
                        'source': edge[0], 
                        'target': edge[1], 
                        'weight': next_edges[edge]
                    }
                    for edge in new_edges
                ][:20],  # Limit to 20
                'disappeared_relationships': [
                    {
                        'source': edge[0], 
                        'target': edge[1], 
                        'weight': curr_edges[edge]
                    }
                    for edge in disappeared_edges
                ][:20],  # Limit to 20
                'strengthened_relationships': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'old_weight': old_weight,
                        'new_weight': new_weight,
                        'change': new_weight - old_weight
                    }
                    for edge, old_weight, new_weight in sorted(
                        strengthened_edges, 
                        key=lambda x: x[2] - x[1], 
                        reverse=True
                    )
                ][:20],  # Limit to 20
                'weakened_relationships': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'old_weight': old_weight,
                        'new_weight': new_weight,
                        'change': old_weight - new_weight
                    }
                    for edge, old_weight, new_weight in sorted(
                        weakened_edges, 
                        key=lambda x: x[1] - x[2], 
                        reverse=True
                    )
                ][:20]  # Limit to 20
            }
        
        return relationship_data
    
    def analyze_sentiment_evolution(self) -> Dict:
        """
        Analyze sentiment evolution over time.
        
        Returns:
            Dictionary with sentiment evolution data
        """
        if not self._organize_by_time_periods():
            return {'error': 'Failed to organize documents by time periods'}
            
        try:
            from textblob import TextBlob
            
            # Initialize results structure
            results = {
                'time_periods': self.time_periods,
                'granularity': self.time_granularity,
                'sentiment_by_period': {},
                'entity_sentiment': {},
                'sentiment_trends': {}
            }
            
            # Track sentiment by period
            period_sentiment = {}
            
            for period in self.time_periods:
                documents = self.period_documents[period]
                
                # Calculate sentiment for each document
                doc_sentiments = []
                
                for doc in documents:
                    blob = TextBlob(doc)
                    doc_sentiments.append(blob.sentiment.polarity)
                
                # Calculate average and variance
                avg_sentiment = np.mean(doc_sentiments)
                sentiment_variance = np.var(doc_sentiments)
                
                # Calculate sentiment distribution
                negative = sum(1 for s in doc_sentiments if s < -0.1)
                neutral = sum(1 for s in doc_sentiments if -0.1 <= s <= 0.1)
                positive = sum(1 for s in doc_sentiments if s > 0.1)
                
                # Store data
                period_sentiment[period] = {
                    'average': float(avg_sentiment),
                    'variance': float(sentiment_variance),
                    'distribution': {
                        'negative': negative,
                        'neutral': neutral,
                        'positive': positive
                    }
                }
                
                # Add to results
                results['sentiment_by_period'][period] = period_sentiment[period]
            
            # Calculate entity sentiment if possible
            try:
                # Run entity analysis first
                entity_evolution = self.analyze_entity_evolution()
                
                if 'error' not in entity_evolution:
                    entity_sentiment = self._analyze_entity_sentiment(entity_evolution)
                    results['entity_sentiment'] = entity_sentiment
            except Exception as e:
                logger.warning(f"Entity sentiment analysis failed: {e}")
            
            # Calculate sentiment trends
            sentiment_trends = []
            
            # Extract sentiment values in chronological order
            sentiment_values = [period_sentiment[period]['average'] for period in self.time_periods]
            
            # Calculate trend (simple linear regression)
            x = np.arange(len(sentiment_values))
            slope, intercept = np.polyfit(x, sentiment_values, 1)
            
            # Smooth sentiment if enabled
            if self.smoothing_window > 0 and len(sentiment_values) > self.smoothing_window:
                smooth_sentiment = np.convolve(
                    sentiment_values, 
                    np.ones(self.smoothing_window) / self.smoothing_window, 
                    mode='valid'
                )
                
                # Add to results
                results['sentiment_trends'] = {
                    'overall_trend': float(slope),
                    'sentiment_values': sentiment_values,
                    'smoothed_values': smooth_sentiment.tolist() if isinstance(smooth_sentiment, np.ndarray) else smooth_sentiment
                }
            else:
                results['sentiment_trends'] = {
                    'overall_trend': float(slope),
                    'sentiment_values': sentiment_values
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment evolution: {e}", exc_info=True)
            return {'error': f"Error analyzing sentiment evolution: {str(e)}"}
    
    def _analyze_entity_sentiment(self, entity_evolution: Dict) -> Dict:
        """
        Analyze sentiment evolution for specific entities.
        
        Args:
            entity_evolution: Entity evolution data
            
        Returns:
            Dictionary with entity sentiment data
        """
        from textblob import TextBlob
        
        # Get trending entities
        trending_up = [item['entity'] for item in entity_evolution.get('entity_trends', {}).get('trending_up', [])]
        trending_down = [item['entity'] for item in entity_evolution.get('entity_trends', {}).get('trending_down', [])]
        
        # Combine trending entities
        trending_entities = trending_up + trending_down
        
        # Calculate sentiment for trending entities in each period
        entity_sentiment = {}
        
        for entity in trending_entities:
            entity_sentiment[entity] = {}
            
            for period in self.time_periods:
                documents = self.period_documents[period]
                
                # Find sentences containing the entity
                entity_sentences = []
                
                for doc in documents:
                    sentences = re.split(r'[.!?]+', doc)
                    
                    for sentence in sentences:
                        if re.search(r'\b' + re.escape(entity) + r'\b', sentence, re.IGNORECASE):
                            entity_sentences.append(sentence)
                
                # Calculate sentiment for entity sentences
                if entity_sentences:
                    sentiments = []
                    
                    for sentence in entity_sentences:
                        blob = TextBlob(sentence)
                        sentiments.append(blob.sentiment.polarity)
                        
                    avg_sentiment = np.mean(sentiments)
                    
                    entity_sentiment[entity][period] = {
                        'sentiment': float(avg_sentiment),
                        'mentions': len(entity_sentences)
                    }
                else:
                    entity_sentiment[entity][period] = {
                        'sentiment': 0.0,
                        'mentions': 0
                    }
        
        # Filter to entities with data in at least half the periods
        min_periods = max(2, len(self.time_periods) // 2)
        filtered_entities = {}
        
        for entity, period_data in entity_sentiment.items():
            valid_periods = sum(1 for data in period_data.values() if data['mentions'] > 0)
            
            if valid_periods >= min_periods:
                filtered_entities[entity] = period_data
                
                # Calculate trend
                sentiments = [data['sentiment'] for data in period_data.values()]
                x = np.arange(len(sentiments))
                slope, _ = np.polyfit(x, sentiments, 1)
                
                filtered_entities[entity]['trend'] = float(slope)
        
        return filtered_entities
    
    def visualize_topic_evolution(self, 
                                topic_evolution: Dict, 
                                output_path: str = None,
                                width: int = 1200,
                                height: int = 900) -> str:
        """
        Visualize topic evolution over time.
        
        Args:
            topic_evolution: Topic evolution data
            output_path: Path to save visualization
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Path to the saved visualization
        """
        if 'error' in topic_evolution:
            logger.error(f"Cannot visualize topic evolution: {topic_evolution['error']}")
            return None
            
        if 'time_periods' not in topic_evolution or 'topics_by_period' not in topic_evolution:
            logger.error("Invalid topic evolution data")
            return None
            
        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"topic_evolution_{timestamp}.png")
            
        try:
            time_periods = topic_evolution['time_periods']
            
            # Extract trending topics
            trending_up = []
            trending_down = []
            
            if 'trending_topics' in topic_evolution:
                trending_up = topic_evolution['trending_topics'].get('trending_up', [])
                trending_down = topic_evolution['trending_topics'].get('trending_down', [])
            
            # Create figure
            plt.figure(figsize=(width/100, height/100), dpi=100)
            
            # Create grid layout
            n_rows = 3
            n_cols = 2
            
            # Plot 1: Topic count by period
            topic_counts = []
            for period in time_periods:
                topic_counts.append(len(topic_evolution['topics_by_period'][period]))
                
            plt.subplot(n_rows, n_cols, 1)
            plt.plot(time_periods, topic_counts, marker='o', linestyle='-', color='blue')
            plt.title('Topics per Period', fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Plot 2: Topic similarity between periods
            if 'topic_similarity' in topic_evolution:
                similarities = []
                periods = []
                
                for key, similarity in topic_evolution['topic_similarity'].items():
                    from_period, to_period = key.split('_to_')
                    periods.append(from_period)
                    similarities.append(similarity)
                    
                plt.subplot(n_rows, n_cols, 2)
                plt.plot(periods, similarities, marker='o', linestyle='-', color='green')
                plt.title('Topic Similarity Between Periods', fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            # Plot 3: Trending Terms (Up)
            if trending_up:
                plt.subplot(n_rows, n_cols, 3)
                
                # Prepare data for each trending term
                for term_data in trending_up[:5]:  # Top 5
                    term = term_data['term']
                    freq = term_data['frequency']
                    
                    # If frequency data missing for some periods, pad with zeros
                    if len(freq) < len(time_periods):
                        freq = freq + [0] * (len(time_periods) - len(freq))
                        
                    plt.plot(time_periods, freq, marker='o', linestyle='-', label=term)
                    
                plt.title('Trending Terms (Up)', fontsize=14)
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
            
            # Plot 4: Trending Terms (Down)
            if trending_down:
                plt.subplot(n_rows, n_cols, 4)
                
                # Prepare data for each trending term
                for term_data in trending_down[:5]:  # Top 5
                    term = term_data['term']
                    freq = term_data['frequency']
                    
                    # If frequency data missing for some periods, pad with zeros
                    if len(freq) < len(time_periods):
                        freq = freq + [0] * (len(time_periods) - len(freq))
                        
                    plt.plot(time_periods, freq, marker='o', linestyle='-', label=term)
                    
                plt.title('Trending Terms (Down)', fontsize=14)
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
            
            # Plot 5: Topic Evolution
            if 'topic_evolution' in topic_evolution:
                plt.subplot(n_rows, n_cols, 5)
                
                # Count topics that evolve across periods
                evolution_counts = defaultdict(int)
                
                for topic_key, data in topic_evolution['topic_evolution'].items():
                    evolution = data['evolution']
                    for entry in evolution:
                        evolution_counts[entry['period']] += 1
                        
                periods = sorted(evolution_counts.keys())
                counts = [evolution_counts[period] for period in periods]
                
                plt.bar(periods, counts, color='purple')
                plt.title('Topics Evolving Across Periods', fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            # Plot 6: Topic Coherence
            coherence_data = {}
            for period in time_periods:
                period_topics = topic_evolution['topics_by_period'][period]
                coherence_values = []
                
                for topic_id, topic_data in period_topics.items():
                    if 'coherence' in topic_data and topic_data['coherence'] is not None:
                        coherence_values.append(topic_data['coherence'])
                        
                if coherence_values:
                    coherence_data[period] = {
                        'mean': np.mean(coherence_values),
                        'max': np.max(coherence_values),
                        'min': np.min(coherence_values)
                    }
            
            if coherence_data:
                plt.subplot(n_rows, n_cols, 6)
                
                periods = list(coherence_data.keys())
                mean_coherence = [coherence_data[period]['mean'] for period in periods]
                max_coherence = [coherence_data[period]['max'] for period in periods]
                min_coherence = [coherence_data[period]['min'] for period in periods]
                
                plt.plot(periods, mean_coherence, marker='o', linestyle='-', color='red', label='Mean')
                plt.plot(periods, max_coherence, marker='^', linestyle='--', color='green', label='Max')
                plt.plot(periods, min_coherence, marker='v', linestyle='--', color='orange', label='Min')
                
                plt.title('Topic Coherence', fontsize=14)
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
            
            # Adjust layout and save
            plt.tight_layout(pad=3.0)
            
            with self.file_lock:
                plt.savefig(output_path, bbox_inches='tight')
                
            plt.close()
            
            logger.info(f"Topic evolution visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing topic evolution: {e}", exc_info=True)
            return None
    
    def visualize_entity_evolution(self, 
                                entity_evolution: Dict, 
                                output_path: str = None,
                                width: int = 1200,
                                height: int = 900) -> str:
        """
        Visualize entity evolution over time.
        
        Args:
            entity_evolution: Entity evolution data
            output_path: Path to save visualization
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Path to the saved visualization
        """
        if 'error' in entity_evolution:
            logger.error(f"Cannot visualize entity evolution: {entity_evolution['error']}")
            return None
            
        if 'time_periods' not in entity_evolution or 'entities_by_period' not in entity_evolution:
            logger.error("Invalid entity evolution data")
            return None
            
        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"entity_evolution_{timestamp}.png")
            
        try:
            time_periods = entity_evolution['time_periods']
            
            # Create figure
            plt.figure(figsize=(width/100, height/100), dpi=100)
            
            # Create grid layout
            n_rows = 3
            n_cols = 2
            
            # Plot 1: Entity count by period
            entity_counts = []
            for period in time_periods:
                entity_counts.append(len(entity_evolution['entities_by_period'][period]))
                
            plt.subplot(n_rows, n_cols, 1)
            plt.plot(time_periods, entity_counts, marker='o', linestyle='-', color='blue')
            plt.title('Entities per Period', fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Plot 2: Entity type distribution
            entity_types = defaultdict(lambda: defaultdict(int))
            
            for period in time_periods:
                period_entities = entity_evolution['entities_by_period'][period]
                
                for entity_data in period_entities:
                    entity_type = entity_data['type']
                    entity_types[entity_type][period] += 1
            
            if entity_types:
                plt.subplot(n_rows, n_cols, 2)
                
                # Select top 5 most common entity types
                type_counts = Counter()
                for entity_type, period_counts in entity_types.items():
                    type_counts[entity_type] = sum(period_counts.values())
                    
                top_types = [type for type, _ in type_counts.most_common(5)]
                
                for entity_type in top_types:
                    type_data = entity_types[entity_type]
                    counts = [type_data.get(period, 0) for period in time_periods]
                    plt.plot(time_periods, counts, marker='o', linestyle='-', label=entity_type)
                    
                plt.title('Entity Types Over Time', fontsize=14)
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
            
            # Plot 3: Trending Entities (Up)
            if 'entity_trends' in entity_evolution and 'trending_up' in entity_evolution['entity_trends']:
                trending_up = entity_evolution['entity_trends']['trending_up']
                
                if trending_up:
                    plt.subplot(n_rows, n_cols, 3)
                    
                    # Prepare data for each trending entity
                    for entity_data in trending_up[:5]:  # Top 5
                        entity = entity_data['entity']
                        counts = entity_data['counts']
                        
                        plt.plot(time_periods, counts, marker='o', linestyle='-', label=entity)
                        
                    plt.title('Trending Entities (Up)', fontsize=14)
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.tight_layout()
            
            # Plot 4: Trending Entities (Down)
            if 'entity_trends' in entity_evolution and 'trending_down' in entity_evolution['entity_trends']:
                trending_down = entity_evolution['entity_trends']['trending_down']
                
                if trending_down:
                    plt.subplot(n_rows, n_cols, 4)
                    
                    # Prepare data for each trending entity
                    for entity_data in trending_down[:5]:  # Top 5
                        entity = entity_data['entity']
                        counts = entity_data['counts']
                        
                        plt.plot(time_periods, counts, marker='o', linestyle='-', label=entity)
                        
                    plt.title('Trending Entities (Down)', fontsize=14)
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.tight_layout()
            
            # Plot 5: Relationship Evolution
            if 'relationship_evolution' in entity_evolution:
                plt.subplot(n_rows, n_cols, 5)
                
                # Extract relationship counts
                new_rel_counts = []
                disappeared_rel_counts = []
                periods = []
                
                for key, data in entity_evolution['relationship_evolution'].items():
                    from_period, to_period = key.split('_to_')
                    periods.append(from_period)
                    
                    new_rel_counts.append(len(data.get('new_relationships', [])))
                    disappeared_rel_counts.append(len(data.get('disappeared_relationships', [])))
                
                if periods:
                    width = 0.35
                    x = np.arange(len(periods))
                    
                    plt.bar(x - width/2, new_rel_counts, width, label='New Relationships', color='green')
                    plt.bar(x + width/2, disappeared_rel_counts, width, label='Disappeared Relationships', color='red')
                    
                    plt.title('Relationship Evolution', fontsize=14)
                    plt.xticks(x, periods, rotation=45)
                    plt.legend()
                    plt.tight_layout()
            
            # Plot 6: Entity Graph Metrics
            if 'entity_graphs' in entity_evolution:
                plt.subplot(n_rows, n_cols, 6)
                
                periods = []
                nodes = []
                edges = []
                
                for period, graph_data in entity_evolution['entity_graphs'].items():
                    periods.append(period)
                    nodes.append(graph_data.get('nodes', 0))
                    edges.append(graph_data.get('edges', 0))
                
                if periods:
                    width = 0.35
                    x = np.arange(len(periods))
                    
                    plt.bar(x - width/2, nodes, width, label='Nodes', color='blue')
                    plt.bar(x + width/2, edges, width, label='Edges', color='orange')
                    
                    plt.title('Knowledge Graph Metrics', fontsize=14)
                    plt.xticks(x, periods, rotation=45)
                    plt.legend()
                    plt.tight_layout()
            
            # Adjust layout and save
            plt.tight_layout(pad=3.0)
            
            with self.file_lock:
                plt.savefig(output_path, bbox_inches='tight')
                
            plt.close()
            
            logger.info(f"Entity evolution visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing entity evolution: {e}", exc_info=True)
            return None
    
    def visualize_sentiment_evolution(self, 
                                    sentiment_evolution: Dict, 
                                    output_path: str = None,
                                    width: int = 1200,
                                    height: int = 900) -> str:
        """
        Visualize sentiment evolution over time.
        
        Args:
            sentiment_evolution: Sentiment evolution data
            output_path: Path to save visualization
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Path to the saved visualization
        """
        if 'error' in sentiment_evolution:
            logger.error(f"Cannot visualize sentiment evolution: {sentiment_evolution['error']}")
            return None
            
        if 'time_periods' not in sentiment_evolution or 'sentiment_by_period' not in sentiment_evolution:
            logger.error("Invalid sentiment evolution data")
            return None
            
        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"sentiment_evolution_{timestamp}.png")
            
        try:
            time_periods = sentiment_evolution['time_periods']
            
            # Create figure
            plt.figure(figsize=(width/100, height/100), dpi=100)
            
            # Create grid layout
            n_rows = 2
            n_cols = 2
            
            # Plot 1: Average sentiment by period
            avg_sentiment = []
            for period in time_periods:
                avg_sentiment.append(sentiment_evolution['sentiment_by_period'][period]['average'])
                
            plt.subplot(n_rows, n_cols, 1)
            
            # Plot raw sentiment
            plt.plot(time_periods, avg_sentiment, marker='o', linestyle='-', color='blue', label='Raw')
            
            # Plot smoothed sentiment if available
            if 'sentiment_trends' in sentiment_evolution and 'smoothed_values' in sentiment_evolution['sentiment_trends']:
                smoothed = sentiment_evolution['sentiment_trends']['smoothed_values']
                
                # Create appropriate x values for smoothed data
                if len(smoothed) < len(time_periods):
                    offset = (len(time_periods) - len(smoothed)) // 2
                    smoothed_periods = time_periods[offset:offset+len(smoothed)]
                else:
                    smoothed_periods = time_periods
                    
                plt.plot(smoothed_periods, smoothed, marker='s', linestyle='--', color='red', label='Smoothed')
            
            # Add trend line
            if 'sentiment_trends' in sentiment_evolution and 'overall_trend' in sentiment_evolution['sentiment_trends']:
                trend = sentiment_evolution['sentiment_trends']['overall_trend']
                x = np.arange(len(time_periods))
                y = trend * x + avg_sentiment[0]
                plt.plot(time_periods, y, linestyle='-.', color='green', label=f'Trend ({trend:.3f})')
            
            plt.title('Average Sentiment by Period', fontsize=14)
            plt.xticks(rotation=45)
            plt.ylabel('Sentiment (-1 to 1)')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            
            # Plot 2: Sentiment distribution by period
            plt.subplot(n_rows, n_cols, 2)
            
            negative = []
            neutral = []
            positive = []
            
            for period in time_periods:
                distribution = sentiment_evolution['sentiment_by_period'][period]['distribution']
                negative.append(distribution['negative'])
                neutral.append(distribution['neutral'])
                positive.append(distribution['positive'])
            
            x = np.arange(len(time_periods))
            width = 0.25
            
            plt.bar(x - width, negative, width, label='Negative', color='red')
            plt.bar(x, neutral, width, label='Neutral', color='gray')
            plt.bar(x + width, positive, width, label='Positive', color='green')
            
            plt.title('Sentiment Distribution by Period', fontsize=14)
            plt.xticks(x, time_periods, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Plot 3: Entity sentiment (if available)
            if 'entity_sentiment' in sentiment_evolution and sentiment_evolution['entity_sentiment']:
                plt.subplot(n_rows, n_cols, 3)
                
                # Select top entities with most sentiment change
                entity_sentiment_change = []
                
                for entity, data in sentiment_evolution['entity_sentiment'].items():
                    if 'trend' in data:
                        trend = data['trend']
                        entity_sentiment_change.append((entity, abs(trend), trend))
                
                # Sort by sentiment change magnitude
                entity_sentiment_change.sort(key=lambda x: x[1], reverse=True)
                
                # Plot top entities
                for entity, _, trend in entity_sentiment_change[:5]:  # Top 5
                    sentiment_values = []
                    
                    for period in time_periods:
                        if period in sentiment_evolution['entity_sentiment'][entity]:
                            sentiment_values.append(sentiment_evolution['entity_sentiment'][entity][period]['sentiment'])
                        else:
                            sentiment_values.append(0)
                            
                    plt.plot(time_periods, sentiment_values, marker='o', linestyle='-', label=f"{entity} ({trend:.3f})")
                
                plt.title('Entity Sentiment Over Time', fontsize=14)
                plt.xticks(rotation=45)
                plt.ylabel('Sentiment (-1 to 1)')
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                plt.legend()
                plt.tight_layout()
            
            # Plot 4: Sentiment variance
            plt.subplot(n_rows, n_cols, 4)
            
            variance = []
            for period in time_periods:
                variance.append(sentiment_evolution['sentiment_by_period'][period]['variance'])
                
            plt.plot(time_periods, variance, marker='o', linestyle='-', color='purple')
            plt.title('Sentiment Variance by Period', fontsize=14)
            plt.xticks(rotation=45)
            plt.ylabel('Variance')
            plt.tight_layout()
            
            # Adjust layout and save
            plt.tight_layout(pad=3.0)
            
            with self.file_lock:
                plt.savefig(output_path, bbox_inches='tight')
                
            plt.close()
            
            logger.info(f"Sentiment evolution visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing sentiment evolution: {e}", exc_info=True)
            return None
    
    def create_evolution_report(self, 
                              output_path: str = None, 
                              include_topics: bool = True,
                              include_entities: bool = True,
                              include_sentiment: bool = True) -> str:
        """
        Create a comprehensive evolution report combining all analyses.
        
        Args:
            output_path: Path to save the HTML report
            include_topics: Whether to include topic evolution analysis
            include_entities: Whether to include entity evolution analysis
            include_sentiment: Whether to include sentiment evolution analysis
            
        Returns:
            Path to the saved report
        """
        if not self.documents:
            logger.error("No documents to analyze")
            return None
            
        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"evolution_report_{timestamp}.html")
            
        try:
            # Perform analyses
            topic_evolution = None
            entity_evolution = None
            sentiment_evolution = None
            
            if include_topics:
                logger.info("Analyzing topic evolution...")
                topic_evolution = self.analyze_topic_evolution()
                
            if include_entities:
                logger.info("Analyzing entity evolution...")
                entity_evolution = self.analyze_entity_evolution()
                
            if include_sentiment:
                logger.info("Analyzing sentiment evolution...")
                sentiment_evolution = self.analyze_sentiment_evolution()
            
            # Create visualizations
            topic_viz_path = None
            entity_viz_path = None
            sentiment_viz_path = None
            
            if topic_evolution and 'error' not in topic_evolution:
                topic_viz_path = self.visualize_topic_evolution(topic_evolution)
                
            if entity_evolution and 'error' not in entity_evolution:
                entity_viz_path = self.visualize_entity_evolution(entity_evolution)
                
            if sentiment_evolution and 'error' not in sentiment_evolution:
                sentiment_viz_path = self.visualize_sentiment_evolution(sentiment_evolution)
            
            # Generate HTML report
            html_content = self._generate_html_report(
                topic_evolution, entity_evolution, sentiment_evolution,
                topic_viz_path, entity_viz_path, sentiment_viz_path
            )
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"Evolution report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating evolution report: {e}", exc_info=True)
            return None
    
    def _generate_html_report(self, 
                            topic_evolution: Dict, 
                            entity_evolution: Dict,
                            sentiment_evolution: Dict,
                            topic_viz_path: str,
                            entity_viz_path: str,
                            sentiment_viz_path: str) -> str:
        """
        Generate HTML content for the evolution report.
        
        Args:
            topic_evolution: Topic evolution data
            entity_evolution: Entity evolution data
            sentiment_evolution: Sentiment evolution data
            topic_viz_path: Path to topic visualization
            entity_viz_path: Path to entity visualization
            sentiment_viz_path: Path to sentiment visualization
            
        Returns:
            HTML content as string
        """
        # Extract time periods
        time_periods = []
        if topic_evolution and 'time_periods' in topic_evolution:
            time_periods = topic_evolution['time_periods']
        elif entity_evolution and 'time_periods' in entity_evolution:
            time_periods = entity_evolution['time_periods']
        elif sentiment_evolution and 'time_periods' in sentiment_evolution:
            time_periods = sentiment_evolution['time_periods']
            
        # Extract granularity
        granularity = None
        if topic_evolution and 'granularity' in topic_evolution:
            granularity = topic_evolution['granularity']
        elif entity_evolution and 'granularity' in entity_evolution:
            granularity = entity_evolution['granularity']
        elif sentiment_evolution and 'granularity' in sentiment_evolution:
            granularity = sentiment_evolution['granularity']
            
        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Temporal Knowledge Evolution Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                
                h1, h2, h3, h4 {{
                    margin-top: 0;
                    color: #2c3e50;
                }}
                
                header h1 {{
                    color: white;
                }}
                
                .section {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    margin-bottom: 30px;
                }}
                
                .section-header {{
                    background-color: #34495e;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px 5px 0 0;
                    margin: -20px -20px 20px -20px;
                }}
                
                .section-header h2 {{
                    color: white;
                    margin: 0;
                }}
                
                .visualization {{
                    text-align: center;
                    margin: 20px 0;
                }}
                
                .visualization img {{
                    max-width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                
                th {{
                    background-color: #f8f9fa;
                    font-weight: 600;
                }}
                
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                
                .card {{
                    background-color: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                
                .card-header {{
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                    margin-bottom: 10px;
                    font-weight: 600;
                }}
                
                .trend-up {{
                    color: #27ae60;
                }}
                
                .trend-down {{
                    color: #e74c3c;
                }}
                
                .flex-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .flex-item {{
                    flex: 1;
                    min-width: 300px;
                }}
                
                footer {{
                    text-align: center;
                    padding: 20px;
                    margin-top: 30px;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
                
                @media (max-width: 768px) {{
                    .flex-container {{
                        flex-direction: column;
                    }}
                    
                    .flex-item {{
                        min-width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>Temporal Knowledge Evolution Report</h1>
                <p>Analysis of knowledge evolution across {len(time_periods)} time periods ({granularity if granularity else 'custom'} granularity)</p>
            </header>
            
            <div class="container">
                <div class="section">
                    <div class="section-header">
                        <h2>Overview</h2>
                    </div>
                    
                    <p>This report presents the evolution of knowledge, topics, entities, and sentiment over time based on the analysis of {len(self.documents)} documents spanning {len(time_periods)} time periods.</p>
                    
                    <h3>Time Periods</h3>
                    <table>
                        <tr>
                            <th>Period</th>
                            <th>Documents</th>
                        </tr>
        """
        
        # Add time period information
        for period in time_periods:
            doc_count = len(self.period_documents.get(period, []))
            html += f"""
                        <tr>
                            <td>{period}</td>
                            <td>{doc_count}</td>
                        </tr>
            """
            
        html += """
                    </table>
                </div>
        """
        
        # Topic Evolution Section
        if topic_evolution and 'error' not in topic_evolution:
            html += """
                <div class="section">
                    <div class="section-header">
                        <h2>Topic Evolution</h2>
                    </div>
            """
            
            # Add visualization
            if topic_viz_path:
                viz_filename = os.path.basename(topic_viz_path)
                html += f"""
                    <div class="visualization">
                        <img src="{viz_filename}" alt="Topic Evolution Visualization">
                    </div>
                """
            
            # Add trending topics
            if 'trending_topics' in topic_evolution and (
                topic_evolution['trending_topics'].get('trending_up') or
                topic_evolution['trending_topics'].get('trending_down')
            ):
                html += """
                    <div class="flex-container">
                """
                
                # Trending Up
                if topic_evolution['trending_topics'].get('trending_up'):
                    html += """
                        <div class="flex-item">
                            <div class="card">
                                <div class="card-header">Trending Up Topics</div>
                                <table>
                                    <tr>
                                        <th>Term</th>
                                        <th>Trend</th>
                                    </tr>
                    """
                    
                    for term_data in topic_evolution['trending_topics']['trending_up']:
                        html += f"""
                                    <tr>
                                        <td>{term_data['term']}</td>
                                        <td class="trend-up">+{term_data['trend']:.3f}</td>
                                    </tr>
                        """
                        
                    html += """
                                </table>
                            </div>
                        </div>
                    """
                
                # Trending Down
                if topic_evolution['trending_topics'].get('trending_down'):
                    html += """
                        <div class="flex-item">
                            <div class="card">
                                <div class="card-header">Trending Down Topics</div>
                                <table>
                                    <tr>
                                        <th>Term</th>
                                        <th>Trend</th>
                                    </tr>
                    """
                    
                    for term_data in topic_evolution['trending_topics']['trending_down']:
                        html += f"""
                                    <tr>
                                        <td>{term_data['term']}</td>
                                        <td class="trend-down">{term_data['trend']:.3f}</td>
                                    </tr>
                        """
                        
                    html += """
                                </table>
                            </div>
                        </div>
                    """
                    
                html += """
                    </div>
                """
            
            # Add topic evolution details
            if 'topic_evolution' in topic_evolution and topic_evolution['topic_evolution']:
                html += """
                    <h3>Topic Evolution Chains</h3>
                    <div class="flex-container">
                """
                
                # Add evolution chains
                for topic_key, data in list(topic_evolution['topic_evolution'].items())[:4]:  # Limit to 4
                    evolution = data['evolution']
                    
                    html += f"""
                        <div class="flex-item">
                            <div class="card">
                                <div class="card-header">{data['label']}</div>
                                <table>
                                    <tr>
                                        <th>Period</th>
                                        <th>Top Terms</th>
                                        <th>Similarity</th>
                                    </tr>
                    """
                    
                    for entry in evolution:
                        terms = ", ".join([term for term, _ in entry['terms'][:3]])
                        similarity = entry.get('similarity', 1.0)
                        similarity_str = f"{similarity:.2f}" if similarity < 1.0 else "—"
                        
                        html += f"""
                                    <tr>
                                        <td>{entry['period']}</td>
                                        <td>{terms}</td>
                                        <td>{similarity_str}</td>
                                    </tr>
                        """
                        
                    html += """
                                </table>
                            </div>
                        </div>
                    """
                    
                html += """
                    </div>
                """
                
            html += """
                </div>
            """
        
        # Entity Evolution Section
        if entity_evolution and 'error' not in entity_evolution:
            html += """
                <div class="section">
                    <div class="section-header">
                        <h2>Entity Evolution</h2>
                    </div>
            """
            
            # Add visualization
            if entity_viz_path:
                viz_filename = os.path.basename(entity_viz_path)
                html += f"""
                    <div class="visualization">
                        <img src="{viz_filename}" alt="Entity Evolution Visualization">
                    </div>
                """
            
            # Add trending entities
            if 'entity_trends' in entity_evolution and (
                entity_evolution['entity_trends'].get('trending_up') or
                entity_evolution['entity_trends'].get('trending_down')
            ):
                html += """
                    <div class="flex-container">
                """
                
                # Trending Up
                if entity_evolution['entity_trends'].get('trending_up'):
                    html += """
                        <div class="flex-item">
                            <div class="card">
                                <div class="card-header">Trending Up Entities</div>
                                <table>
                                    <tr>
                                        <th>Entity</th>
                                        <th>Type</th>
                                        <th>Trend</th>
                                    </tr>
                    """
                    
                    for entity_data in entity_evolution['entity_trends']['trending_up'][:10]:  # Top 10
                        html += f"""
                                    <tr>
                                        <td>{entity_data['entity']}</td>
                                        <td>{entity_data['type']}</td>
                                        <td class="trend-up">+{entity_data['trend']:.3f}</td>
                                    </tr>
                        """
                        
                    html += """
                                </table>
                            </div>
                        </div>
                    """
                
                # Trending Down
                if entity_evolution['entity_trends'].get('trending_down'):
                    html += """
                        <div class="flex-item">
                            <div class="card">
                                <div class="card-header">Trending Down Entities</div>
                                <table>
                                    <tr>
                                        <th>Entity</th>
                                        <th>Type</th>
                                        <th>Trend</th>
                                    </tr>
                    """
                    
                    for entity_data in entity_evolution['entity_trends']['trending_down'][:10]:  # Top 10
                        html += f"""
                                    <tr>
                                        <td>{entity_data['entity']}</td>
                                        <td>{entity_data['type']}</td>
                                        <td class="trend-down">{entity_data['trend']:.3f}</td>
                                    </tr>
                        """
                        
                    html += """
                                </table>
                            </div>
                        </div>
                    """
                    
                html += """
                    </div>
                """
            
            # Add relationship evolution
            if 'relationship_evolution' in entity_evolution and entity_evolution['relationship_evolution']:
                html += """
                    <h3>Relationship Evolution</h3>
                    <table>
                        <tr>
                            <th>Transition</th>
                            <th>Common Entities</th>
                            <th>New Entities</th>
                            <th>Disappeared Entities</th>
                            <th>Relationship Changes</th>
                        </tr>
                """
                
                for transition, data in entity_evolution['relationship_evolution'].items():
                    from_period, to_period = transition.split('_to_')
                    
                    new_entities = len(data.get('new_entities', []))
                    disappeared_entities = len(data.get('disappeared_entities', []))
                    new_rels = len(data.get('new_relationships', []))
                    disappeared_rels = len(data.get('disappeared_relationships', []))
                    
                    html += f"""
                        <tr>
                            <td>{from_period} → {to_period}</td>
                            <td>{data.get('common_entities', 0)}</td>
                            <td>{new_entities}</td>
                            <td>{disappeared_entities}</td>
                            <td>+{new_rels} / -{disappeared_rels}</td>
                        </tr>
                    """
                    
                html += """
                    </table>
                """
                
            # Add entity graph visualizations
            if 'entity_graphs' in entity_evolution and entity_evolution['entity_graphs']:
                html += """
                    <h3>Knowledge Graphs by Period</h3>
                    <div class="flex-container">
                """
                
                for period, graph_data in entity_evolution['entity_graphs'].items():
                    viz_path = graph_data.get('visualization')
                    
                    if viz_path:
                        html += f"""
                            <div class="flex-item">
                                <div class="card">
                                    <div class="card-header">{period} ({graph_data['nodes']} nodes, {graph_data['edges']} edges)</div>
                                    <div class="visualization">
                                        <img src="{viz_path}" alt="Knowledge Graph {period}">
                                    </div>
                                </div>
                            </div>
                        """
                        
                html += """
                    </div>
                """
                
            html += """
                </div>
            """
        
        # Sentiment Evolution Section
        if sentiment_evolution and 'error' not in sentiment_evolution:
            html += """
                <div class="section">
                    <div class="section-header">
                        <h2>Sentiment Evolution</h2>
                    </div>
            """
            
            # Add visualization
            if sentiment_viz_path:
                viz_filename = os.path.basename(sentiment_viz_path)
                html += f"""
                    <div class="visualization">
                        <img src="{viz_filename}" alt="Sentiment Evolution Visualization">
                    </div>
                """
            
            # Add sentiment overview table
            html += """
                <h3>Sentiment by Period</h3>
                <table>
                    <tr>
                        <th>Period</th>
                        <th>Average Sentiment</th>
                        <th>Variance</th>
                        <th>Positive</th>
                        <th>Neutral</th>
                        <th>Negative</th>
                    </tr>
            """
            
            for period in time_periods:
                if period in sentiment_evolution['sentiment_by_period']:
                    period_data = sentiment_evolution['sentiment_by_period'][period]
                    avg = period_data['average']
                    var = period_data['variance']
                    pos = period_data['distribution']['positive']
                    neu = period_data['distribution']['neutral']
                    neg = period_data['distribution']['negative']
                    
                    sentiment_class = "trend-up" if avg > 0.1 else "trend-down" if avg < -0.1 else ""
                    
                    html += f"""
                        <tr>
                            <td>{period}</td>
                            <td class="{sentiment_class}">{avg:.3f}</td>
                            <td>{var:.3f}</td>
                            <td>{pos}</td>
                            <td>{neu}</td>
                            <td>{neg}</td>
                        </tr>
                    """
                    
            html += """
                </table>
            """
            
            # Add entity sentiment if available
            if 'entity_sentiment' in sentiment_evolution and sentiment_evolution['entity_sentiment']:
                html += """
                    <h3>Entity Sentiment Evolution</h3>
                    
                    <div class="flex-container">
                """
                
                # Get entities with strongest sentiment trends
                entity_trends = []
                for entity, data in sentiment_evolution['entity_sentiment'].items():
                    if 'trend' in data:
                        entity_trends.append((entity, data['trend']))
                        
                # Sort by absolute trend value
                entity_trends.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Create entity sentiment cards
                for entity, trend in entity_trends[:4]:  # Top 4
                    entity_data = sentiment_evolution['entity_sentiment'][entity]
                    trend_class = "trend-up" if trend > 0 else "trend-down"
                    
                    html += f"""
                        <div class="flex-item">
                            <div class="card">
                                <div class="card-header">{entity} <span class="{trend_class}">({trend:.3f})</span></div>
                                <table>
                                    <tr>
                                        <th>Period</th>
                                        <th>Sentiment</th>
                                        <th>Mentions</th>
                                    </tr>
                    """
                    
                    for period in time_periods:
                        if period in entity_data:
                            period_sentiment = entity_data[period]['sentiment']
                            mentions = entity_data[period]['mentions']
                            sentiment_class = "trend-up" if period_sentiment > 0.1 else "trend-down" if period_sentiment < -0.1 else ""
                            
                            html += f"""
                                    <tr>
                                        <td>{period}</td>
                                        <td class="{sentiment_class}">{period_sentiment:.3f}</td>
                                        <td>{mentions}</td>
                                    </tr>
                            """
                            
                    html += """
                                </table>
                            </div>
                        </div>
                    """
                    
                html += """
                    </div>
                """
                
            html += """
                </div>
            """
        
        # Footer
        html += f"""
                <footer>
                    <p>Generated by TemporalAnalysis on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Sample test
    analyzer = TemporalAnalysis(time_granularity='month')
    
    # Sample documents by period
    docs_by_period = {
        '2023-01': [
            "Machine learning has seen significant advancements in recent years. Deep learning models are becoming increasingly sophisticated.",
            "Artificial intelligence applications are expanding across various domains including healthcare and finance."
        ],
        '2023-02': [
            "Natural language processing models continue to improve, with new transformer architectures being developed.",
            "Computer vision systems are now capable of real-time object detection with greater accuracy than ever before."
        ],
        '2023-03': [
            "Ethical concerns about AI are growing as these systems become more integrated into society.",
            "Explainable AI is an emerging field focused on making AI decisions more transparent and understandable."
        ],
        '2023-04': [
            "Reinforcement learning is showing promise in robotics and autonomous systems development.",
            "Federated learning approaches enable privacy-preserving machine learning across distributed datasets."
        ]
    }
    
    # Add documents to analyzer
    for period, documents in docs_by_period.items():
        for doc in documents:
            analyzer.add_document(doc, f"{period}-15", '%Y-%m-%d')
    
    # Run full analysis and create report
    report_path = analyzer.create_evolution_report()
    
    if report_path:
        logger.info(f"Evolution report generated: {report_path}")
    else:
        logger.error("Failed to generate evolution report")
