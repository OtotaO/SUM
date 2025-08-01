"""
kg_web_interface.py - Interactive web interface for knowledge graph exploration

This module provides a Flask-based web interface for visualizing, exploring,
and interacting with knowledge graphs. It allows users to navigate complex
information networks through an intuitive UI.

Design principles:
- Separation of concerns (MVC architecture)
- Responsive design (mobile-first approach)
- Progressive enhancement (graceful degradation)
- Security by design (Schneier principles)
- Optimized performance (Knuth efficiency)

Author: ototao
License: Apache License 2.0
"""

import os
import json
import time
import logging
import uuid
import tempfile
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from threading import Lock

from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename

# Import SUM components
from knowledge_graph import KnowledgeGraph
from Models.topic_modeling import TopicModeler
from advanced_sum import AdvancedSUM

# Configure logging
logger = logging.getLogger(__name__)


class KnowledgeGraphUI:
    """
    Web interface for interacting with knowledge graphs.
    
    This class provides a Flask-based web application for visualizing,
    exploring, and analyzing knowledge graphs through an intuitive UI.
    """
    
    def __init__(self, 
                app: Flask,
                output_dir: str = 'output',
                upload_dir: str = None,
                max_upload_size: int = 16 * 1024 * 1024,  # 16MB
                allowed_extensions: List[str] = None):
        """
        Initialize the knowledge graph web interface.
        
        Args:
            app: Flask application instance
            output_dir: Directory for saving visualizations
            upload_dir: Directory for file uploads
            max_upload_size: Maximum file upload size in bytes
            allowed_extensions: List of allowed file extensions
        """
        self.app = app
        self.output_dir = output_dir
        self.upload_dir = upload_dir or os.path.join(tempfile.gettempdir(), 'kg_uploads')
        
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Configure app
        self.app.config['MAX_CONTENT_LENGTH'] = max_upload_size
        self.app.config['UPLOAD_FOLDER'] = self.upload_dir
        self.app.config['ALLOWED_EXTENSIONS'] = allowed_extensions or {'txt', 'json', 'csv', 'md'}
        
        # Set up storage for active graphs
        self.active_graphs = {}
        self.graph_lock = Lock()
        
        # Register routes
        self._register_routes()
        
        logger.info("KnowledgeGraphUI initialized")
    
    def _register_routes(self) -> None:
        """Register Flask routes for the application."""
        app = self.app
        
        @app.route('/kg')
        def kg_index():
            """Render the knowledge graph exploration interface."""
            return render_template('kg_index.html')
        
        @app.route('/kg/static/<path:path>')
        def kg_static(path):
            """Serve static files for the knowledge graph UI."""
            return send_from_directory('static/kg', path)
        
        @app.route('/kg/visualizations/<path:path>')
        def kg_visualizations(path):
            """Serve generated visualizations."""
            return send_from_directory(self.output_dir, path)
        
        @app.route('/kg/api/create', methods=['POST'])
        def kg_create():
            """Create a new knowledge graph."""
            try:
                # Check if a file was uploaded
                if 'file' not in request.files and 'data' not in request.form:
                    return jsonify({'error': 'No file or data provided'}), 400
                    
                # Process file upload
                if 'file' in request.files:
                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({'error': 'No file selected'}), 400
                        
                    if not self._allowed_file(file.filename):
                        return jsonify({'error': 'File type not allowed'}), 400
                        
                    # Save file securely
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    try:
                        # Process file based on type
                        file_type = self._get_file_type(filepath)
                        
                        if file_type == 'json':
                            # Load JSON data
                            with open(filepath, 'r', encoding='utf-8') as f:
                                graph_data = json.load(f)
                                
                            # Check if this is a saved knowledge graph
                            if self._is_kg_data(graph_data):
                                # Load directly
                                kg = KnowledgeGraph.load(filepath, output_dir=self.output_dir)
                            else:
                                # Process as entity data
                                kg = self._create_kg_from_json(graph_data)
                        else:
                            # For other file types, perform text analysis
                            kg = self._create_kg_from_text_file(filepath)
                    finally:
                        # Clean up uploaded file
                        try:
                            os.unlink(filepath)
                        except Exception as e:
                            logger.warning(f"Failed to delete uploaded file {filepath}: {e}")
                
                # Process direct data input
                elif 'data' in request.form:
                    data_text = request.form['data']
                    
                    try:
                        # Parse as JSON
                        data = json.loads(data_text)
                        kg = self._create_kg_from_json(data)
                    except json.JSONDecodeError:
                        # Treat as plain text
                        kg = self._create_kg_from_text(data_text)
                
                if not kg:
                    return jsonify({'error': 'Failed to create knowledge graph'}), 500
                    
                # Register the graph
                graph_id = str(uuid.uuid4())
                
                with self.graph_lock:
                    self.active_graphs[graph_id] = kg
                
                # Generate visualization
                visualization_path = kg.generate_html_visualization(
                    title=f"Knowledge Graph - {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                if not visualization_path:
                    return jsonify({'error': 'Failed to generate visualization'}), 500
                    
                # Extract filename from path
                visualization_filename = os.path.basename(visualization_path)
                
                # Return response
                return jsonify({
                    'graph_id': graph_id,
                    'node_count': kg.G.number_of_nodes(),
                    'edge_count': kg.G.number_of_edges(),
                    'visualization_url': f'/kg/visualizations/{visualization_filename}'
                })
                
            except Exception as e:
                logger.error(f"Error creating knowledge graph: {e}", exc_info=True)
                return jsonify({'error': f'Error: {str(e)}'}), 500
        
        @app.route('/kg/api/analyze/<graph_id>', methods=['GET'])
        def kg_analyze(graph_id):
            """Analyze a knowledge graph and return insights."""
            try:
                # Check if graph exists
                if graph_id not in self.active_graphs:
                    return jsonify({'error': 'Graph not found'}), 404
                    
                kg = self.active_graphs[graph_id]
                
                # Perform analysis
                central_entities = kg.get_central_entities(top_n=10)
                communities = kg.get_communities(min_community_size=3)
                
                # Format for response
                central_entities_data = [
                    {'entity': entity, 'centrality': float(score)}
                    for entity, score in central_entities
                ]
                
                communities_data = [
                    {
                        'id': community_id,
                        'size': len(entities),
                        'entities': entities[:10],  # Limit to 10 per community
                        'has_more': len(entities) > 10
                    }
                    for community_id, entities in communities.items()
                ]
                
                # Get entity type distribution
                entity_types = {}
                for _, data in kg.G.nodes(data=True):
                    entity_type = data.get('type', 'DEFAULT')
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                # Get relation type distribution
                relation_types = {}
                for _, _, data in kg.G.edges(data=True):
                    relation_type = data.get('type', 'generic')
                    relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
                
                return jsonify({
                    'graph_id': graph_id,
                    'node_count': kg.G.number_of_nodes(),
                    'edge_count': kg.G.number_of_edges(),
                    'central_entities': central_entities_data,
                    'communities': communities_data,
                    'entity_types': entity_types,
                    'relation_types': relation_types
                })
                
            except Exception as e:
                logger.error(f"Error analyzing knowledge graph: {e}", exc_info=True)
                return jsonify({'error': f'Error: {str(e)}'}), 500
        
        @app.route('/kg/api/search/<graph_id>', methods=['GET'])
        def kg_search(graph_id):
            """Search for entities in a knowledge graph."""
            try:
                # Check if graph exists
                if graph_id not in self.active_graphs:
                    return jsonify({'error': 'Graph not found'}), 404
                    
                # Get query parameter
                query = request.args.get('q', '')
                if not query:
                    return jsonify({'error': 'No search query provided'}), 400
                    
                kg = self.active_graphs[graph_id]
                
                # Perform search
                results = kg.search_entities(query, max_results=20)
                
                # Format for response
                search_results = [
                    {'entity': entity, 'type': entity_type, 'relevance': float(score)}
                    for entity, entity_type, score in results
                ]
                
                return jsonify({
                    'graph_id': graph_id,
                    'query': query,
                    'results': search_results
                })
                
            except Exception as e:
                logger.error(f"Error searching knowledge graph: {e}", exc_info=True)
                return jsonify({'error': f'Error: {str(e)}'}), 500
        
        @app.route('/kg/api/paths/<graph_id>', methods=['GET'])
        def kg_paths(graph_id):
            """Find paths between entities in a knowledge graph."""
            try:
                # Check if graph exists
                if graph_id not in self.active_graphs:
                    return jsonify({'error': 'Graph not found'}), 404
                    
                # Get query parameters
                source = request.args.get('source', '')
                target = request.args.get('target', '')
                max_length = int(request.args.get('max_length', '3'))
                
                if not source or not target:
                    return jsonify({'error': 'Source and target entities required'}), 400
                    
                kg = self.active_graphs[graph_id]
                
                # Find matching entities if not exact
                if source not in kg.nodes:
                    source_matches = kg.search_entities(source, max_results=1)
                    if source_matches:
                        source = source_matches[0][0]
                    else:
                        return jsonify({'error': f'Source entity not found: {source}'}), 404
                
                if target not in kg.nodes:
                    target_matches = kg.search_entities(target, max_results=1)
                    if target_matches:
                        target = target_matches[0][0]
                    else:
                        return jsonify({'error': f'Target entity not found: {target}'}), 404
                
                # Find paths
                paths = kg.find_paths(source, target, max_length=max_length)
                
                # Format for response
                path_data = []
                for path in paths:
                    formatted_path = []
                    for entity, relation in path:
                        if relation:
                            formatted_path.append({
                                'entity': entity,
                                'relation': relation
                            })
                        else:
                            formatted_path.append({
                                'entity': entity
                            })
                    path_data.append(formatted_path)
                
                return jsonify({
                    'graph_id': graph_id,
                    'source': source,
                    'target': target,
                    'paths': path_data
                })
                
            except Exception as e:
                logger.error(f"Error finding paths in knowledge graph: {e}", exc_info=True)
                return jsonify({'error': f'Error: {str(e)}'}), 500
        
        @app.route('/kg/api/export/<graph_id>', methods=['GET'])
        def kg_export(graph_id):
            """Export a knowledge graph in various formats."""
            try:
                # Check if graph exists
                if graph_id not in self.active_graphs:
                    return jsonify({'error': 'Graph not found'}), 404
                    
                # Get format parameter
                export_format = request.args.get('format', 'json')
                if export_format not in ['json', 'gexf', 'graphml', 'cytoscape']:
                    return jsonify({'error': f'Unsupported export format: {export_format}'}), 400
                    
                kg = self.active_graphs[graph_id]
                
                # Export graph
                export_path = kg.export_graph(format=export_format)
                
                if not export_path:
                    return jsonify({'error': 'Failed to export graph'}), 500
                    
                # Extract filename from path
                export_filename = os.path.basename(export_path)
                
                return jsonify({
                    'graph_id': graph_id,
                    'format': export_format,
                    'download_url': f'/kg/visualizations/{export_filename}'
                })
                
            except Exception as e:
                logger.error(f"Error exporting knowledge graph: {e}", exc_info=True)
                return jsonify({'error': f'Error: {str(e)}'}), 500
        
        @app.route('/kg/api/delete/<graph_id>', methods=['POST'])
        def kg_delete(graph_id):
            """Delete a knowledge graph from memory."""
            try:
                # Check if graph exists
                if graph_id not in self.active_graphs:
                    return jsonify({'error': 'Graph not found'}), 404
                    
                # Remove graph
                with self.graph_lock:
                    del self.active_graphs[graph_id]
                
                return jsonify({
                    'success': True,
                    'message': f'Graph {graph_id} deleted'
                })
                
            except Exception as e:
                logger.error(f"Error deleting knowledge graph: {e}", exc_info=True)
                return jsonify({'error': f'Error: {str(e)}'}), 500
    
    def _allowed_file(self, filename: str) -> bool:
        """
        Check if a file has an allowed extension.
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if the file extension is allowed
        """
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.app.config['ALLOWED_EXTENSIONS']
    
    def _get_file_type(self, filepath: str) -> str:
        """
        Determine the type of a file based on extension.
        
        Args:
            filepath: Path to the file
            
        Returns:
            File type as string
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.json':
            return 'json'
        elif ext == '.csv':
            return 'csv'
        elif ext == '.md':
            return 'markdown'
        else:
            return 'text'
    
    def _is_kg_data(self, data: Any) -> bool:
        """
        Check if JSON data represents a saved knowledge graph.
        
        Args:
            data: Loaded JSON data
            
        Returns:
            True if the data appears to be a saved knowledge graph
        """
        # Check for expected structure of a saved knowledge graph
        required_keys = ['nodes', 'edges', 'graph_data', 'entity_counts']
        return isinstance(data, dict) and all(key in data for key in required_keys)
    
    def _create_kg_from_json(self, data: Dict) -> Optional[KnowledgeGraph]:
        """
        Create a knowledge graph from JSON data.
        
        Args:
            data: JSON data with entities and/or relationships
            
        Returns:
            KnowledgeGraph instance or None if creation failed
        """
        try:
            kg = KnowledgeGraph(output_dir=self.output_dir)
            
            # Check for different JSON structures
            
            # Case 1: Entities and relationships structure
            if 'entities' in data and 'relationships' in data:
                entities = data['entities']
                relationships = data['relationships']
                
                if entities:
                    kg.build_from_entities(entities)
                    
                if relationships:
                    kg.build_from_relationships(relationships)
            
            # Case 2: Nodes and edges structure
            elif 'nodes' in data and 'edges' in data:
                nodes = data['nodes']
                edges = data['edges']
                
                # Convert to entities and relationships
                entities = []
                for node in nodes:
                    if isinstance(node, dict):
                        label = node.get('label') or node.get('name')
                        node_type = node.get('type', 'DEFAULT')
                        weight = node.get('weight', 1)
                        
                        if label:
                            entities.append((label, node_type, weight))
                
                relationships = []
                for edge in edges:
                    if isinstance(edge, dict):
                        source = edge.get('source') or edge.get('from')
                        target = edge.get('target') or edge.get('to')
                        edge_type = edge.get('type', 'generic')
                        weight = edge.get('weight', 1)
                        
                        if source and target:
                            relationships.append({
                                'source': source,
                                'target': target,
                                'type': edge_type,
                                'weight': weight
                            })
                
                if entities:
                    kg.build_from_entities(entities)
                    
                if relationships:
                    kg.build_from_relationships(relationships)
            
            # Case 3: SUM-specific format with entries
            elif 'entries' in data:
                entries = data['entries']
                summarizer = AdvancedSUM()
                
                entities_list = []
                
                for entry in entries:
                    if isinstance(entry, dict) and 'content' in entry:
                        # Extract entities from content
                        content = entry['content']
                        
                        try:
                            # Process with AdvancedSUM to extract entities
                            result = summarizer.process_text(content)
                            
                            if 'entities' in result:
                                for entity, entity_type, count in result['entities']:
                                    entities_list.append((entity, entity_type, count))
                        except Exception as e:
                            logger.warning(f"Error processing entry: {e}")
                
                if entities_list:
                    kg.build_from_entities(entities_list)
            
            # Check if we successfully built a graph
            if kg.G.number_of_nodes() == 0:
                logger.warning("Failed to create knowledge graph from JSON: no nodes created")
                return None
                
            # Prune the graph to remove weak connections
            kg.prune_graph()
            
            return kg
            
        except Exception as e:
            logger.error(f"Error creating knowledge graph from JSON: {e}", exc_info=True)
            return None
    
    def _create_kg_from_text_file(self, filepath: str) -> Optional[KnowledgeGraph]:
        """
        Create a knowledge graph from a text file.
        
        Args:
            filepath: Path to the text file
            
        Returns:
            KnowledgeGraph instance or None if creation failed
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                
            return self._create_kg_from_text(text)
            
        except Exception as e:
            logger.error(f"Error reading text file: {e}", exc_info=True)
            return None
    
    def _create_kg_from_text(self, text: str) -> Optional[KnowledgeGraph]:
        """
        Create a knowledge graph from text using NLP analysis.
        
        Args:
            text: Text content to analyze
            
        Returns:
            KnowledgeGraph instance or None if creation failed
        """
        try:
            # Initialize the knowledge graph
            kg = KnowledgeGraph(output_dir=self.output_dir)
            
            # Initialize NLP components
            summarizer = AdvancedSUM()
            topic_modeler = TopicModeler(n_topics=5)
            
            # Process text to extract entities
            result = summarizer.process_text(text)
            
            # Extract entities
            entities_list = []
            if 'entities' in result and result['entities']:
                for entity, entity_type, count in result['entities']:
                    entities_list.append((entity, entity_type, count))
            
            # Extract topics
            try:
                topic_modeler.fit([text])
                topics = []
                
                for topic_idx in range(topic_modeler.n_topics):
                    terms = topic_modeler.get_topic_terms(topic_idx)
                    
                    topic_terms = []
                    for term, weight in terms:
                        topic_terms.append({
                            'term': term,
                            'weight': float(weight)
                        })
                    
                    topics.append({
                        'id': topic_idx,
                        'label': f"Topic {topic_idx + 1}",
                        'terms': topic_terms
                    })
                    
            except Exception as e:
                logger.warning(f"Error performing topic modeling: {e}")
                topics = []
            
            # Build the knowledge graph
            if entities_list:
                kg.build_from_entities(entities_list)
                
            if topics:
                kg.add_topics(topics)
                
            # If no entities were found, try to extract keywords
            if kg.G.number_of_nodes() == 0 and 'tags' in result and result['tags']:
                keyword_entities = [
                    (keyword, 'CONCEPT', 1) for keyword in result['tags']
                ]
                kg.build_from_entities(keyword_entities)
            
            # Check if we successfully built a graph
            if kg.G.number_of_nodes() == 0:
                logger.warning("Failed to create knowledge graph from text: no nodes created")
                return None
                
            # Prune the graph to remove weak connections
            kg.prune_graph()
            
            return kg
            
        except Exception as e:
            logger.error(f"Error creating knowledge graph from text: {e}", exc_info=True)
            return None


def create_kg_templates():
    """Create HTML templates for the knowledge graph UI."""
    # Base directory for templates
    template_dir = 'templates'
    os.makedirs(template_dir, exist_ok=True)
    
    # Knowledge graph index template
    kg_index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SUM Knowledge Graph Explorer</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="/kg/static/css/kg_styles.css">
    </head>
    <body>
        <div class="container">
            <header>
                <h1>SUM Knowledge Graph Explorer</h1>
                <p>Visualize and analyze knowledge networks extracted from text data</p>
            </header>
            
            <div class="tabs">
                <button class="tab-button active" data-tab="input">Create Graph</button>
                <button class="tab-button" data-tab="explore" disabled>Explore Graph</button>
                <button class="tab-button" data-tab="analyze" disabled>Analyze Graph</button>
            </div>
            
            <div class="tab-content" id="input-tab">
                <div class="input-section">
                    <h2>Create Knowledge Graph</h2>
                    <p>Upload a file or enter text to create a knowledge graph.</p>
                    
                    <div class="input-toggle">
                        <button class="toggle-button active" data-input="file">Upload File</button>
                        <button class="toggle-button" data-input="text">Enter Text</button>
                    </div>
                    
                    <div class="input-container" id="file-input">
                        <form id="file-form" enctype="multipart/form-data">
                            <div class="file-drop-area">
                                <span class="file-msg">Drag & drop file here or click to browse</span>
                                <input class="file-input" type="file" name="file" accept=".txt,.json,.csv,.md">
                            </div>
                            <div class="selected-file"></div>
                            <button type="submit" class="primary-button">Create Graph</button>
                        </form>
                    </div>
                    
                    <div class="input-container" id="text-input" style="display:none;">
                        <form id="text-form">
                            <textarea name="data" placeholder="Enter text to analyze..." rows="10"></textarea>
                            <button type="submit" class="primary-button">Create Graph</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="tab-content" id="explore-tab" style="display:none;">
                <div class="loading-indicator">Loading graph...</div>
                <div class="visualization-container">
                    <iframe id="graph-viz" src="" frameborder="0"></iframe>
                </div>
                <div class="action-buttons">
                    <button id="export-json" class="secondary-button">Export as JSON</button>
                    <button id="export-gexf" class="secondary-button">Export for Gephi</button>
                    <button id="new-graph" class="danger-button">New Graph</button>
                </div>
            </div>
            
            <div class="tab-content" id="analyze-tab" style="display:none;">
                <div class="analysis-container">
                    <div class="loading-indicator">Analyzing graph...</div>
                    
                    <div class="analysis-grid">
                        <div class="analysis-card">
                            <h3>Graph Statistics</h3>
                            <div class="stats-container">
                                <div class="stat">
                                    <span class="stat-value" id="node-count">0</span>
                                    <span class="stat-label">Nodes</span>
                                </div>
                                <div class="stat">
                                    <span class="stat-value" id="edge-count">0</span>
                                    <span class="stat-label">Edges</span>
                                </div>
                                <div class="stat">
                                    <span class="stat-value" id="community-count">0</span>
                                    <span class="stat-label">Communities</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="analysis-card">
                            <h3>Entity Types</h3>
                            <div id="entity-types-chart" class="chart-container"></div>
                        </div>
                        
                        <div class="analysis-card">
                            <h3>Central Entities</h3>
                            <div id="central-entities"></div>
                        </div>
                        
                        <div class="analysis-card">
                            <h3>Communities</h3>
                            <div id="communities"></div>
                        </div>
                    </div>
                    
                    <div class="path-finder">
                        <h3>Find Paths Between Entities</h3>
                        <div class="path-form">
                            <input type="text" id="source-entity" placeholder="Source entity">
                            <input type="text" id="target-entity" placeholder="Target entity">
                            <button id="find-paths" class="primary-button">Find Paths</button>
                        </div>
                        <div id="paths-result"></div>
                    </div>
                </div>
            </div>
            
            <div class="notification" id="notification"></div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.0/dist/chart.min.js"></script>
        <script src="/kg/static/js/kg_main.js"></script>
    </body>
    </html>
    """
    
    with open(os.path.join(template_dir, 'kg_index.html'), 'w', encoding='utf-8') as f:
        f.write(kg_index_html)
    
    # Create static directories
    static_dir = 'static/kg'
    os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
    
    # CSS file
    kg_css = """
    /* Knowledge Graph Explorer Styles */
    
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #3498db;
        --accent-color: #e74c3c;
        --light-color: #ecf0f1;
        --dark-color: #34495e;
        --success-color: #2ecc71;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
        --shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        --transition: all 0.3s ease;
    }
    
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }
    
    body {
        font-family: 'Roboto', sans-serif;
        line-height: 1.6;
        color: var(--dark-color);
        background-color: #f5f5f5;
    }
    
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    header h1 {
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    header p {
        color: var(--dark-color);
        font-size: 1.1rem;
    }
    
    /* Tabs */
    
    .tabs {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    
    .tab-button {
        padding: 0.8rem 1.5rem;
        background-color: var(--light-color);
        border: none;
        border-bottom: 3px solid transparent;
        cursor: pointer;
        font-size: 1rem;
        font-weight: 500;
        transition: var(--transition);
        margin: 0 0.5rem;
    }
    
    .tab-button:hover:not([disabled]) {
        background-color: rgba(236, 240, 241, 0.8);
    }
    
    .tab-button.active {
        border-bottom: 3px solid var(--secondary-color);
        color: var(--secondary-color);
    }
    
    .tab-button[disabled] {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    .tab-content {
        background-color: white;
        border-radius: 0.5rem;
        padding: 2rem;
        box-shadow: var(--shadow);
    }
    
    /* Input Section */
    
    .input-section h2 {
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    .input-section p {
        margin-bottom: 1.5rem;
        color: var(--dark-color);
    }
    
    .input-toggle {
        display: flex;
        margin-bottom: 1.5rem;
    }
    
    .toggle-button {
        padding: 0.6rem 1.2rem;
        background-color: var(--light-color);
        border: none;
        cursor: pointer;
        transition: var(--transition);
        flex: 1;
        text-align: center;
    }
    
    .toggle-button:first-child {
        border-radius: 0.3rem 0 0 0.3rem;
    }
    
    .toggle-button:last-child {
        border-radius: 0 0.3rem 0.3rem 0;
    }
    
    .toggle-button.active {
        background-color: var(--secondary-color);
        color: white;
    }
    
    .file-drop-area {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 3rem;
        text-align: center;
        margin-bottom: 1.5rem;
        cursor: pointer;
        transition: var(--transition);
    }
    
    .file-drop-area:hover {
        border-color: var(--secondary-color);
    }
    
    .file-input {
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: pointer;
    }
    
    .selected-file {
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    textarea {
        width: 100%;
        padding: 1rem;
        border: 1px solid #ccc;
        border-radius: 0.5rem;
        font-family: inherit;
        resize: vertical;
        margin-bottom: 1.5rem;
    }
    
    .primary-button, .secondary-button, .danger-button {
        padding: 0.8rem 1.5rem;
        border: none;
        border-radius: 0.3rem;
        font-weight: 500;
        cursor: pointer;
        transition: var(--transition);
    }
    
    .primary-button {
        background-color: var(--secondary-color);
        color: white;
    }
    
    .primary-button:hover {
        background-color: #2980b9;
    }
    
    .secondary-button {
        background-color: var(--light-color);
        color: var(--dark-color);
        margin-right: 0.5rem;
    }
    
    .secondary-button:hover {
        background-color: #bdc3c7;
    }
    
    .danger-button {
        background-color: var(--danger-color);
        color: white;
    }
    
    .danger-button:hover {
        background-color: #c0392b;
    }
    
    /* Visualization */
    
    .visualization-container {
        height: 70vh;
        min-height: 500px;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    
    iframe {
        width: 100%;
        height: 100%;
        border: none;
    }
    
    .action-buttons {
        display: flex;
        justify-content: flex-end;
    }
    
    /* Analysis */
    
    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .analysis-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: var(--shadow);
    }
    
    .analysis-card h3 {
        margin-bottom: 1rem;
        color: var(--primary-color);
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
    }
    
    .stat {
        text-align: center;
    }
    
    .stat-value {
        display: block;
        font-size: 2rem;
        font-weight: 700;
        color: var(--secondary-color);
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: var(--dark-color);
    }
    
    .chart-container {
        height: 200px;
    }
    
    #central-entities, #communities {
        max-height: 300px;
        overflow-y: auto;
    }
    
    .entity-item, .community-item {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 0.3rem;
        background-color: #f5f5f5;
    }
    
    .entity-name {
        font-weight: 500;
    }
    
    .entity-score {
        float: right;
        font-size: 0.9rem;
        color: var(--dark-color);
    }
    
    .community-entities {
        margin-top: 0.5rem;
        padding-left: 1rem;
        font-size: 0.9rem;
    }
    
    /* Path finder */
    
    .path-finder {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: var(--shadow);
    }
    
    .path-finder h3 {
        margin-bottom: 1rem;
        color: var(--primary-color);
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
    }
    
    .path-form {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .path-form input {
        flex: 1;
        padding: 0.8rem;
        border: 1px solid #ccc;
        border-radius: 0.3rem;
    }
    
    .path-result {
        margin-top: 1rem;
    }
    
    .path {
        margin-bottom: 1rem;
        padding: 1rem;
        background-color: #f5f5f5;
        border-radius: 0.3rem;
    }
    
    .path-step {
        display: flex;
        align-items: center;
    }
    
    .path-entity {
        padding: 0.3rem 0.6rem;
        background-color: var(--light-color);
        border-radius: 0.3rem;
        margin-right: 0.5rem;
    }
    
    .path-relation {
        font-size: 0.9rem;
        color: var(--dark-color);
        margin: 0 0.5rem;
    }
    
    /* Loading */
    
    .loading-indicator {
        display: none;
        text-align: center;
        padding: 2rem;
        font-weight: 500;
        color: var(--dark-color);
    }
    
    /* Notification */
    
    .notification {
        position: fixed;
        bottom: -100px;
        left: 50%;
        transform: translateX(-50%);
        padding: 1rem 2rem;
        background-color: white;
        color: var(--dark-color);
        border-radius: 0.5rem;
        box-shadow: var(--shadow);
        transition: bottom 0.3s ease;
        z-index: 1000;
    }
    
    .notification.success {
        background-color: var(--success-color);
        color: white;
    }
    
    .notification.error {
        background-color: var(--danger-color);
        color: white;
    }
    
    .notification.show {
        bottom: 20px;
    }
    
    /* Responsive */
    
    @media (max-width: 768px) {
        .container {
            padding: 1rem;
        }
        
        .tabs {
            flex-direction: column;
        }
        
        .tab-button {
            margin-bottom: 0.5rem;
        }
        
        .path-form {
            flex-direction: column;
        }
        
        .stats-container {
            flex-direction: column;
            gap: 1rem;
        }
    }
    """
    
    with open(os.path.join(static_dir, 'css', 'kg_styles.css'), 'w', encoding='utf-8') as f:
        f.write(kg_css)
    
    # JavaScript file
    kg_js = """
    // Knowledge Graph Explorer Main JavaScript
    
    document.addEventListener('DOMContentLoaded', function() {
        // Variables
        let currentGraphId = null;
        let analysisData = null;
        
        // DOM Elements
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        const toggleButtons = document.querySelectorAll('.toggle-button');
        const inputContainers = document.querySelectorAll('.input-container');
        const fileForm = document.getElementById('file-form');
        const textForm = document.getElementById('text-form');
        const fileDropArea = document.querySelector('.file-drop-area');
        const fileInput = document.querySelector('.file-input');
        const selectedFile = document.querySelector('.selected-file');
        const graphViz = document.getElementById('graph-viz');
        const exportJsonBtn = document.getElementById('export-json');
        const exportGexfBtn = document.getElementById('export-gexf');
        const newGraphBtn = document.getElementById('new-graph');
        const nodeCount = document.getElementById('node-count');
        const edgeCount = document.getElementById('edge-count');
        const communityCount = document.getElementById('community-count');
        const entityTypesChart = document.getElementById('entity-types-chart');
        const centralEntities = document.getElementById('central-entities');
        const communities = document.getElementById('communities');
        const sourceEntity = document.getElementById('source-entity');
        const targetEntity = document.getElementById('target-entity');
        const findPathsBtn = document.getElementById('find-paths');
        const pathsResult = document.getElementById('paths-result');
        const loadingIndicators = document.querySelectorAll('.loading-indicator');
        const notification = document.getElementById('notification');
        
        // Tab Switching
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                if (button.disabled) return;
                
                // Deactivate all tabs
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.style.display = 'none');
                
                // Activate selected tab
                button.classList.add('active');
                const tabId = button.getAttribute('data-tab');
                document.getElementById(`${tabId}-tab`).style.display = 'block';
            });
        });
        
        // Input Toggle
        toggleButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Deactivate all toggles
                toggleButtons.forEach(btn => btn.classList.remove('active'));
                inputContainers.forEach(container => container.style.display = 'none');
                
                // Activate selected toggle
                button.classList.add('active');
                const inputId = button.getAttribute('data-input');
                document.getElementById(`${inputId}-input`).style.display = 'block';
            });
        });
        
        // File Drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            fileDropArea.style.borderColor = '#3498db';
        }
        
        function unhighlight() {
            fileDropArea.style.borderColor = '#ccc';
        }
        
        fileDropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                updateFileInfo();
            }
        }
        
        fileInput.addEventListener('change', updateFileInfo);
        
        function updateFileInfo() {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                selectedFile.textContent = `Selected file: ${file.name} (${formatFileSize(file.size)})`;
            } else {
                selectedFile.textContent = '';
            }
        }
        
        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' bytes';
            else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
            else return (bytes / 1048576).toFixed(1) + ' MB';
        }
        
        // Create Graph from File
        fileForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            if (!fileInput.files.length) {
                showNotification('Please select a file first', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            createGraph(formData);
        });
        
        // Create Graph from Text
        textForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const text = textForm.elements.data.value.trim();
            if (!text) {
                showNotification('Please enter some text', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('data', text);
            
            createGraph(formData);
        });
        
        // Create Graph
        function createGraph(formData) {
            // Show loading
            showLoading();
            
            fetch('/kg/api/create', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Error creating graph');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Hide loading
                hideLoading();
                
                // Store graph ID
                currentGraphId = data.graph_id;
                
                // Set visualization
                graphViz.src = data.visualization_url;
                
                // Enable tabs
                enableTabs();
                
                // Switch to explore tab
                document.querySelector('[data-tab="explore"]').click();
                
                // Show success notification
                showNotification('Graph created successfully', 'success');
                
                // Load analysis data
                loadAnalysisData();
            })
            .catch(error => {
                // Hide loading
                hideLoading();
                
                // Show error notification
                showNotification(error.message, 'error');
            });
        }
        
        // Load Analysis Data
        function loadAnalysisData() {
            if (!currentGraphId) return;
            
            // Show loading
            document.querySelector('#analyze-tab .loading-indicator').style.display = 'block';
            
            fetch(`/kg/api/analyze/${currentGraphId}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Error analyzing graph');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Hide loading
                document.querySelector('#analyze-tab .loading-indicator').style.display = 'none';
                
                // Store analysis data
                analysisData = data;
                
                // Update UI with analysis data
                updateAnalysisUI(data);
            })
            .catch(error => {
                // Hide loading
                document.querySelector('#analyze-tab .loading-indicator').style.display = 'none';
                
                // Show error notification
                showNotification(error.message, 'error');
            });
        }
        
        // Update Analysis UI
        function updateAnalysisUI(data) {
            // Update stats
            nodeCount.textContent = data.node_count;
            edgeCount.textContent = data.edge_count;
            communityCount.textContent = data.communities.length;
            
            // Entity types chart
            if (data.entity_types && Object.keys(data.entity_types).length > 0) {
                const canvas = document.createElement('canvas');
                entityTypesChart.innerHTML = '';
                entityTypesChart.appendChild(canvas);
                
                const labels = Object.keys(data.entity_types);
                const values = Object.values(data.entity_types);
                
                const colors = [
                    '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd', '#8c564b',
                    '#e377c2', '#bcbd22', '#17becf', '#d62728', '#9edae5',
                    '#c49c94', '#dbdb8d', '#c7c7c7', '#7f7f7f'
                ];
                
                new Chart(canvas, {
                    type: 'pie',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: values,
                            backgroundColor: colors.slice(0, labels.length)
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'right'
                            }
                        }
                    }
                });
            } else {
                entityTypesChart.innerHTML = '<p>No entity type data available</p>';
            }
            
            // Central entities
            if (data.central_entities && data.central_entities.length > 0) {
                centralEntities.innerHTML = '';
                
                data.central_entities.forEach(entity => {
                    const entityItem = document.createElement('div');
                    entityItem.className = 'entity-item';
                    
                    const entityName = document.createElement('span');
                    entityName.className = 'entity-name';
                    entityName.textContent = entity.entity;
                    
                    const entityScore = document.createElement('span');
                    entityScore.className = 'entity-score';
                    entityScore.textContent = entity.centrality.toFixed(3);
                    
                    entityItem.appendChild(entityName);
                    entityItem.appendChild(entityScore);
                    centralEntities.appendChild(entityItem);
                });
            } else {
                centralEntities.innerHTML = '<p>No central entities data available</p>';
            }
            
            // Communities
            if (data.communities && data.communities.length > 0) {
                communities.innerHTML = '';
                
                data.communities.forEach(community => {
                    const communityItem = document.createElement('div');
                    communityItem.className = 'community-item';
                    
                    const communityHeader = document.createElement('div');
                    communityHeader.innerHTML = `<strong>Community ${community.id + 1}</strong> (${community.size} entities)`;
                    
                    const communityEntities = document.createElement('div');
                    communityEntities.className = 'community-entities';
                    
                    community.entities.forEach(entity => {
                        const entityEl = document.createElement('div');
                        entityEl.textContent = entity;
                        communityEntities.appendChild(entityEl);
                    });
                    
                    if (community.has_more) {
                        const moreEl = document.createElement('div');
                        moreEl.textContent = `...and ${community.size - community.entities.length} more`;
                        communityEntities.appendChild(moreEl);
                    }
                    
                    communityItem.appendChild(communityHeader);
                    communityItem.appendChild(communityEntities);
                    communities.appendChild(communityItem);
                });
            } else {
                communities.innerHTML = '<p>No community data available</p>';
            }
        }
        
        // Find Paths
        findPathsBtn.addEventListener('click', function() {
            const source = sourceEntity.value.trim();
            const target = targetEntity.value.trim();
            
            if (!source || !target) {
                showNotification('Please enter both source and target entities', 'error');
                return;
            }
            
            // Show loading
            pathsResult.innerHTML = '<p>Searching for paths...</p>';
            
            fetch(`/kg/api/paths/${currentGraphId}?source=${encodeURIComponent(source)}&target=${encodeURIComponent(target)}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Error finding paths');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Show paths
                if (data.paths && data.paths.length > 0) {
                    pathsResult.innerHTML = `<h4>Found ${data.paths.length} path(s) from "${data.source}" to "${data.target}":</h4>`;
                    
                    data.paths.forEach((path, index) => {
                        const pathDiv = document.createElement('div');
                        pathDiv.className = 'path';
                        
                        const pathHeader = document.createElement('div');
                        pathHeader.innerHTML = `<strong>Path ${index + 1}</strong> (${path.length} steps)`;
                        pathDiv.appendChild(pathHeader);
                        
                        const pathContent = document.createElement('div');
                        pathContent.className = 'path-content';
                        
                        path.forEach((step, stepIndex) => {
                            const stepDiv = document.createElement('div');
                            stepDiv.className = 'path-step';
                            
                            const entity = document.createElement('span');
                            entity.className = 'path-entity';
                            entity.textContent = step.entity;
                            stepDiv.appendChild(entity);
                            
                            if (step.relation && stepIndex < path.length - 1) {
                                const relation = document.createElement('span');
                                relation.className = 'path-relation';
                                relation.textContent = `--[ ${step.relation} ]-->`;
                                stepDiv.appendChild(relation);
                            }
                            
                            pathContent.appendChild(stepDiv);
                        });
                        
                        pathDiv.appendChild(pathContent);
                        pathsResult.appendChild(pathDiv);
                    });
                } else {
                    pathsResult.innerHTML = `<p>No paths found between "${data.source}" and "${data.target}"</p>`;
                }
            })
            .catch(error => {
                // Show error
                pathsResult.innerHTML = `<p class="error">Error: ${error.message}</p>`;
            });
        });
        
        // Export Buttons
        exportJsonBtn.addEventListener('click', function() {
            if (!currentGraphId) return;
            
            fetch(`/kg/api/export/${currentGraphId}?format=json`)
            .then(response => response.json())
            .then(data => {
                if (data.download_url) {
                    window.open(data.download_url, '_blank');
                }
            })
            .catch(error => {
                showNotification('Error exporting graph: ' + error.message, 'error');
            });
        });
        
        exportGexfBtn.addEventListener('click', function() {
            if (!currentGraphId) return;
            
            fetch(`/kg/api/export/${currentGraphId}?format=gexf`)
            .then(response => response.json())
            .then(data => {
                if (data.download_url) {
                    window.open(data.download_url, '_blank');
                }
            })
            .catch(error => {
                showNotification('Error exporting graph: ' + error.message, 'error');
            });
        });
        
        // New Graph Button
        newGraphBtn.addEventListener('click', function() {
            // Reset state
            currentGraphId = null;
            analysisData = null;
            
            // Reset forms
            fileForm.reset();
            textForm.reset();
            selectedFile.textContent = '';
            
            // Disable tabs
            disableTabs();
            
            // Switch to input tab
            document.querySelector('[data-tab="input"]').click();
            
            // Delete graph on server
            if (currentGraphId) {
                fetch(`/kg/api/delete/${currentGraphId}`, {
                    method: 'POST'
                }).catch(error => {
                    console.error('Error deleting graph:', error);
                });
            }
        });
        
        // Helper Functions
        function enableTabs() {
            tabButtons.forEach(button => {
                button.disabled = false;
            });
        }
        
        function disableTabs() {
            tabButtons.forEach(button => {
                if (button.getAttribute('data-tab') !== 'input') {
                    button.disabled = true;
                }
            });
        }
        
        function showLoading() {
            loadingIndicators.forEach(indicator => {
                indicator.style.display = 'block';
            });
        }
        
        function hideLoading() {
            loadingIndicators.forEach(indicator => {
                indicator.style.display = 'none';
            });
        }
        
        function showNotification(message, type = 'info') {
            notification.textContent = message;
            notification.className = 'notification';
            notification.classList.add(type);
            notification.classList.add('show');
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
    });
    """
    
    with open(os.path.join(static_dir, 'js', 'kg_main.js'), 'w', encoding='utf-8') as f:
        f.write(kg_js)
    
    logger.info("Knowledge Graph UI templates created successfully")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create Flask app
    app = Flask(__name__)
    
    # Create KnowledgeGraphUI instance
    kg_ui = KnowledgeGraphUI(app)
    
    # Create templates and static files
    create_kg_templates()
    
    # Start the app
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting KnowledgeGraphUI on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
