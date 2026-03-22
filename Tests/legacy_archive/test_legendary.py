"""
Test suite for legendary features
"""

import pytest
import sys
import os
import asyncio
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rag_crystallizer import GraphRAGCrystallizer
from multi_agent_orchestrator import MultiAgentOrchestrator, AgentRole
from raptor_hierarchical import RAPTORBuilder, RAPTORQueryEngine


class TestGraphRAG:
    """Test GraphRAG crystallization"""
    
    def setup_method(self):
        self.crystallizer = GraphRAGCrystallizer()
        self.test_documents = [
            "Apple Inc. is developing new AI features for the iPhone. Tim Cook announced partnerships.",
            "Microsoft and OpenAI are collaborating on Azure cloud services. AI is transformative.",
            "Google's Gemini model competes with OpenAI's GPT series. Multimodal capabilities are key.",
            "Meta's Llama models are open source. Open AI development is the future."
        ]
    
    def test_corpus_crystallization(self):
        """Test basic corpus crystallization"""
        result = self.crystallizer.crystallize_corpus(self.test_documents)
        
        assert result is not None
        assert result.global_summary is not None
        assert len(result.communities) > 0
        assert result.metadata['num_documents'] == 4
    
    def test_global_questions(self):
        """Test answering global questions"""
        result = self.crystallizer.crystallize_corpus(self.test_documents)
        
        # Test theme extraction
        answer = self.crystallizer.answer_global_question(result, "What are the main themes?")
        assert answer is not None
        assert len(answer) > 0
        
        # Test relationship extraction
        answer = self.crystallizer.answer_global_question(result, "What are the key relationships?")
        assert answer is not None
    
    def test_entity_extraction(self):
        """Test entity extraction and graph building"""
        entities, relations = self.crystallizer.graph_builder.extract_entities_and_relations(
            self.test_documents
        )
        
        assert len(entities) > 0
        assert any('Apple' in e.text for e in entities)
        assert any('Microsoft' in e.text for e in entities)
    
    def test_community_detection(self):
        """Test community detection in knowledge graph"""
        entities, relations = self.crystallizer.graph_builder.extract_entities_and_relations(
            self.test_documents
        )
        graph = self.crystallizer.graph_builder.build_graph(entities, relations)
        
        partition = self.crystallizer.community_detector.detect(graph)
        assert len(partition) > 0
        
        # Check hierarchical detection
        hierarchies = self.crystallizer.community_detector.hierarchical_detection(graph)
        assert len(hierarchies) > 0


class TestMultiAgent:
    """Test multi-agent orchestration"""
    
    def setup_method(self):
        self.orchestrator = MultiAgentOrchestrator(max_workers=5)
        self.test_text = """
        Artificial intelligence is rapidly transforming industries worldwide. 
        Companies like OpenAI and Google are developing sophisticated models.
        These models can understand context and generate human-like text.
        """
    
    @pytest.mark.asyncio
    async def test_orchestration(self):
        """Test basic orchestration"""
        result = await self.orchestrator.orchestrate_crystallization(self.test_text)
        
        assert result is not None
        assert result.summary is not None
        assert result.essence is not None
        assert result.quality_score > 0
        assert result.consensus_score > 0
    
    @pytest.mark.asyncio
    async def test_agent_roles(self):
        """Test individual agent roles"""
        result = await self.orchestrator.orchestrate_crystallization(self.test_text)
        
        # Check that all agents contributed
        assert len(result.agent_contributions) > 0
        assert 'essence' in result.agent_contributions
        assert 'style' in result.agent_contributions
        assert 'facts' in result.agent_contributions
    
    @pytest.mark.asyncio
    async def test_style_variations(self):
        """Test style variations"""
        result = await self.orchestrator.orchestrate_crystallization(
            self.test_text,
            {'style': 'executive'}
        )
        
        assert result.style_variations is not None
        if result.style_variations:
            assert len(result.style_variations) > 0
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        agents = self.orchestrator._initialize_agents()
        
        assert len(agents) == len(AgentRole)
        assert AgentRole.ESSENCE_EXTRACTOR in agents
        assert AgentRole.FACT_CHECKER in agents


class TestRAPTOR:
    """Test RAPTOR hierarchical trees"""
    
    def setup_method(self):
        self.builder = RAPTORBuilder(max_chunk_size=50)
        self.query_engine = RAPTORQueryEngine()
        self.test_text = """
        Machine learning has revolutionized many industries. Deep learning uses neural networks.
        These networks can learn complex patterns from data. Transformers are very successful.
        Models like GPT and BERT use transformer architecture. They understand context well.
        The attention mechanism is key to transformer success. It focuses on relevant parts.
        Self-attention enables understanding relationships. Multi-head attention provides subspaces.
        Training requires vast amounts of data. Pre-training followed by fine-tuning is standard.
        Transfer learning allows knowledge application. This makes AI more accessible.
        """
    
    def test_tree_building(self):
        """Test RAPTOR tree construction"""
        tree = self.builder.build_tree(self.test_text)
        
        assert tree is not None
        assert tree.root is not None
        assert tree.metadata['total_levels'] > 0
        assert tree.metadata['total_nodes'] > 0
    
    def test_hierarchical_levels(self):
        """Test hierarchical level generation"""
        tree = self.builder.build_tree(self.test_text)
        
        assert len(tree.levels) > 0
        
        # Check that higher levels have fewer nodes
        if len(tree.levels) > 1:
            level_sizes = [len(nodes) for nodes in tree.levels.values()]
            # Generally, higher levels should have fewer nodes
            assert max(level_sizes) >= min(level_sizes)
    
    def test_query_engine(self):
        """Test querying the RAPTOR tree"""
        tree = self.builder.build_tree(self.test_text)
        
        # Test basic query
        results = self.query_engine.query(tree, "attention mechanism", top_k=3)
        
        assert len(results) > 0
        assert all(score >= 0 for _, score in results)
    
    def test_context_window(self):
        """Test context window extraction"""
        tree = self.builder.build_tree(self.test_text)
        
        context = self.query_engine.get_context_window(
            tree, 
            "transformers",
            max_tokens=50
        )
        
        assert context is not None
        assert len(context) > 0
        assert len(context.split()) <= 50
    
    def test_chunking(self):
        """Test initial text chunking"""
        chunks = self.builder._create_initial_chunks(self.test_text)
        
        assert len(chunks) > 0
        assert all(chunk.level == 0 for chunk in chunks)
        assert all(chunk.text for chunk in chunks)


class TestIntegration:
    """Test integration of all legendary features"""
    
    def setup_method(self):
        self.graphrag = GraphRAGCrystallizer()
        self.orchestrator = MultiAgentOrchestrator()
        self.raptor = RAPTORBuilder()
        self.test_documents = [
            "AI is transforming how we work and live.",
            "Machine learning models are becoming more sophisticated.",
            "Natural language processing enables human-like communication."
        ]
    
    @pytest.mark.asyncio
    async def test_combined_pipeline(self):
        """Test combining all technologies"""
        # GraphRAG for corpus understanding
        graphrag_result = self.graphrag.crystallize_corpus(self.test_documents)
        
        # RAPTOR for hierarchical structure
        combined_text = ' '.join(self.test_documents)
        raptor_tree = self.raptor.build_tree(combined_text)
        
        # Multi-agent for quality
        agent_result = await self.orchestrator.orchestrate_crystallization(
            raptor_tree.root.summary
        )
        
        # Verify all components produced results
        assert graphrag_result.global_summary is not None
        assert raptor_tree.root.summary is not None
        assert agent_result.summary is not None
        
        # Check quality metrics
        assert agent_result.quality_score > 0
        assert agent_result.consensus_score > 0


def test_imports():
    """Test that all modules can be imported"""
    try:
        from graph_rag_crystallizer import GraphRAGCrystallizer
        from multi_agent_orchestrator import MultiAgentOrchestrator
        from raptor_hierarchical import RAPTORBuilder
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])