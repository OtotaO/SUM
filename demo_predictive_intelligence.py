#!/usr/bin/env python3
"""
demo_predictive_intelligence.py - Comprehensive demonstration of the Predictive Intelligence System

This script showcases all the capabilities of the new predictive intelligence system,
demonstrating how SUM transforms from reactive to proactive knowledge management.

Features demonstrated:
- Context awareness and pattern recognition
- Proactive suggestion generation
- Intelligent scheduling and spaced repetition
- Knowledge graph construction and serendipitous connections
- Integration with existing Knowledge OS

Author: ototao
License: Apache License 2.0
"""

import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import the predictive intelligence system
try:
    from predictive_intelligence import PredictiveIntelligenceSystem
    from predictive_knowledge_os_integration import PredictiveKnowledgeOS
    PREDICTIVE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Predictive Intelligence not available: {e}")
    PREDICTIVE_AVAILABLE = False


def print_header(title: str, width: int = 80):
    """Print a beautiful header."""
    print("\n" + "="*width)
    print(f"🧠 {title}")
    print("="*width)


def print_section(title: str, width: int = 60):
    """Print a section header."""
    print(f"\n📋 {title}")
    print("-" * width)


def simulate_thinking_session(pi_system: PredictiveIntelligenceSystem) -> List[str]:
    """Simulate a realistic thinking session with various topics."""
    sample_thoughts = [
        "I'm fascinated by how machine learning models can identify patterns in data that humans might miss. The emergence of transformers has revolutionized natural language processing.",
        
        "Reading about cognitive science today. The dual-process theory suggests we have two modes of thinking - fast intuitive and slow deliberate. This reminds me of system 1 and system 2 from Kahneman.",
        
        "Working on a new project involving knowledge graphs. I wonder how we can automatically extract relationships between concepts from unstructured text. Graph neural networks might be key here.",
        
        "The intersection of AI and cognitive science is fascinating. Both fields study intelligence, but from different angles. AI focuses on building intelligent systems, while cognitive science studies how intelligence works.",
        
        "Just realized that spaced repetition is essentially optimizing the forgetting curve. Hermann Ebbinghaus discovered this pattern in the 1880s, and now we use it in modern learning systems.",
        
        "Thinking about knowledge management systems. Most are just digital filing cabinets. What if they could anticipate what information you need before you ask for it? Predictive intelligence for knowledge work.",
        
        "Graph databases like Neo4j are powerful for storing relationships, but querying them efficiently is still challenging. The complexity grows exponentially with the depth of relationships you want to explore.",
        
        "Pattern recognition in human thinking is remarkable. We can see faces in clouds, find meaningful connections between disparate ideas, and generate insights through seemingly random associations.",
        
        "The concept of emergence is everywhere - from consciousness arising from neurons to intelligence emerging from simple rules. Complex systems theory explains how simple interactions create sophisticated behaviors.",
        
        "Active learning in machine learning is like how humans learn best - by asking good questions rather than passive consumption. The system identifies the most informative examples to learn from."
    ]
    
    thought_ids = []
    
    print_section("🤔 Simulating Thinking Session")
    
    for i, thought_content in enumerate(sample_thoughts, 1):
        thought_id = f"demo_thought_{i}_{int(time.time())}"
        
        print(f"\n💭 Thought {i}: {thought_content[:100]}...")
        
        # Process through predictive intelligence
        result = pi_system.process_new_thought(thought_content, thought_id)
        
        print(f"   🎯 Generated {result['suggestions_generated']} insights")
        if result['new_concepts']:
            print(f"   🔍 Concepts: {', '.join(result['new_concepts'][:3])}")
        
        thought_ids.append(thought_id)
        time.sleep(0.5)  # Small delay for realism
    
    return thought_ids


def demonstrate_proactive_insights(pi_system: PredictiveIntelligenceSystem):
    """Demonstrate the proactive insight generation."""
    print_section("💡 Proactive Insights Generation")
    
    insights = pi_system.get_proactive_insights(5)
    
    if insights:
        print(f"Generated {len(insights)} proactive insights:")
        
        for i, insight in enumerate(insights, 1):
            print(f"\n{i}. 🔮 {insight['content']}")
            print(f"   Type: {insight['insight_type']} | Confidence: {insight.get('confidence', 0):.1%}")
            
            if insight.get('actions'):
                print(f"   🎯 Suggested actions:")
                for action in insight['actions'][:2]:
                    print(f"      • {action['label']}")
    else:
        print("No insights ready for delivery at this time.")


def demonstrate_pattern_recognition(pi_system: PredictiveIntelligenceSystem):
    """Demonstrate thinking pattern analysis."""
    print_section("📊 Thinking Pattern Analysis")
    
    analysis = pi_system.analyze_thinking_patterns()
    
    if analysis.get('status') == 'insufficient_data':
        print(f"⚠️  {analysis['message']}")
        return
    
    print(f"🧠 Overall Assessment:")
    print(f"   {analysis['overall_assessment']}")
    
    # Show cognitive patterns
    cognitive = analysis['thinking_patterns'].get('cognitive_patterns', {})
    if cognitive:
        print(f"\n🎯 Cognitive Style:")
        print(f"   Dominant: {cognitive.get('dominant_style', 'balanced').title()}")
        print(f"   Strength: {cognitive.get('style_strength', 0):.1%}")
        print(f"   Balanced: {'Yes' if cognitive.get('is_balanced') else 'No'}")
    
    # Show improvement suggestions
    if analysis.get('improvement_suggestions'):
        print(f"\n✨ Personalized Suggestions:")
        for suggestion in analysis['improvement_suggestions'][:3]:
            print(f"   • {suggestion.get('suggestion', 'Continue exploring')}")


def demonstrate_intelligent_scheduling(pi_system: PredictiveIntelligenceSystem):
    """Demonstrate intelligent scheduling capabilities."""
    print_section("📅 Intelligent Scheduling")
    
    schedule = pi_system.get_intelligent_schedule(3)
    
    print("Next 3 days:")
    for date_key, daily_plan in list(schedule['daily_recommendations'].items())[:3]:
        print(f"\n📆 {daily_plan['date']}:")
        
        if daily_plan.get('scheduled_reviews'):
            reviews = [r['concept'].replace('_', ' ') for r in daily_plan['scheduled_reviews']]
            print(f"   📚 Knowledge reviews: {', '.join(reviews)}")
        
        if daily_plan.get('synthesis_opportunities'):
            synthesis = [s['thread_title'] for s in daily_plan['synthesis_opportunities']]
            print(f"   🎯 Synthesis opportunities: {', '.join(synthesis)}")
        
        if daily_plan.get('peak_hours'):
            hours = [f"{h}:00" for h in daily_plan['peak_hours'][:3]]
            print(f"   ⚡ Peak productivity: {', '.join(hours)}")
    
    # Show weekly goals
    if schedule.get('weekly_goals'):
        print(f"\n🎯 Weekly Learning Goals:")
        for goal in schedule['weekly_goals'][:3]:
            print(f"   • {goal['goal']} for {goal['thread']} ({goal['priority']} priority)")


def demonstrate_knowledge_graph(pi_system: PredictiveIntelligenceSystem):
    """Demonstrate knowledge graph capabilities."""
    print_section("🕸️ Knowledge Graph Insights")
    
    # Get graph insights
    graph_insights = pi_system.knowledge_graph.get_graph_insights()
    
    if graph_insights.get('status') == 'empty_graph':
        print("Knowledge graph is still building...")
        return
    
    graph_size = graph_insights['graph_size']
    print(f"📊 Graph Structure:")
    print(f"   Concepts: {graph_size['nodes']}")
    print(f"   Connections: {graph_size['edges']}")
    print(f"   Density: {graph_size.get('density', 0):.2%}")
    
    # Show central concepts
    central_concepts = graph_insights.get('central_concepts', [])
    if central_concepts:
        print(f"\n🌟 Most Central Concepts:")
        for concept_info in central_concepts[:5]:
            concept = concept_info['concept'].replace('_', ' ')
            score = concept_info['centrality_score']
            print(f"   • {concept} (centrality: {score:.2f})")
    
    # Show emerging clusters
    clusters = pi_system.knowledge_graph.detect_emerging_clusters()
    if clusters:
        print(f"\n🔗 Emerging Knowledge Clusters:")
        for cluster in clusters[:3]:
            print(f"   • {cluster['suggested_name']} ({cluster['size']} concepts)")
            print(f"     Emergence score: {cluster['emergence_score']:.1%}")


def demonstrate_contextual_recommendations(pi_system: PredictiveIntelligenceSystem):
    """Demonstrate contextual recommendations."""
    print_section("🎯 Contextual Recommendations")
    
    recommendations = pi_system.get_contextual_recommendations()
    
    # Immediate actions
    if recommendations.get('immediate_actions'):
        print("⚡ Immediate Opportunities:")
        for action in recommendations['immediate_actions'][:2]:
            print(f"   • {action['description']}")
            if action.get('best_time'):
                print(f"     ⏰ Optimal time: {action['best_time']}")
    
    # Suggested explorations
    if recommendations.get('suggested_explorations'):
        print(f"\n🔍 Suggested Explorations:")
        for exploration in recommendations['suggested_explorations'][:2]:
            print(f"   • {exploration['description']}")
    
    # Connection opportunities
    if recommendations.get('connection_opportunities'):
        print(f"\n🔗 Connection Opportunities:")
        for connection in recommendations['connection_opportunities'][:2]:
            print(f"   • {connection['description']}")
    
    # Learning optimizations
    if recommendations.get('learning_optimizations'):
        print(f"\n📈 Learning Optimizations:")
        for optimization in recommendations['learning_optimizations'][:2]:
            print(f"   • {optimization['description']}")


def demonstrate_system_status(pi_system: PredictiveIntelligenceSystem):
    """Show comprehensive system status."""
    print_section("⚡ System Status")
    
    status = pi_system.get_system_status()
    
    print(f"🧠 Context Engine:")
    print(f"   Active threads: {status['context_engine']['active_threads']}")
    print(f"   Concepts tracked: {status['context_engine']['concepts_tracked']}")
    
    print(f"\n💡 Suggestion System:")
    print(f"   Insights queued: {status['suggestion_system']['insights_queued']}")
    print(f"   Insights delivered: {status['suggestion_system']['delivered_insights']}")
    
    print(f"\n📅 Scheduling Engine:")
    print(f"   Concepts scheduled: {status['scheduling_engine']['concepts_scheduled']}")
    print(f"   Reviews due: {status['scheduling_engine']['due_reviews']}")
    
    print(f"\n🕸️ Knowledge Graph:")
    kg_status = status['knowledge_graph']
    if kg_status.get('graph_size'):
        size = kg_status['graph_size']
        print(f"   Nodes: {size['nodes']}, Edges: {size['edges']}")
        print(f"   Density: {size.get('density', 0):.2%}")
    
    print(f"\n⚡ Overall Health: {status['system_health'].upper()}")


def main():
    """Main demonstration function."""
    if not PREDICTIVE_AVAILABLE:
        print("❌ Predictive Intelligence System not available")
        return
    
    print_header("SUM Predictive Intelligence System Demonstration", 80)
    print("Transforming from reactive to proactive knowledge management")
    
    # Initialize system
    print("\n🚀 Initializing Predictive Intelligence System...")
    pi_system = PredictiveIntelligenceSystem("demo_predictive_data")
    print("✅ System initialized successfully!")
    
    try:
        # 1. Simulate thinking session
        thought_ids = simulate_thinking_session(pi_system)
        
        # 2. Demonstrate proactive insights
        demonstrate_proactive_insights(pi_system)
        
        # 3. Show pattern recognition
        demonstrate_pattern_recognition(pi_system)
        
        # 4. Show intelligent scheduling
        demonstrate_intelligent_scheduling(pi_system)
        
        # 5. Show knowledge graph
        demonstrate_knowledge_graph(pi_system)
        
        # 6. Show contextual recommendations
        demonstrate_contextual_recommendations(pi_system)
        
        # 7. Show system status
        demonstrate_system_status(pi_system)
        
        # Final summary
        print_header("🎉 Demonstration Complete", 80)
        print("The Predictive Intelligence System has successfully:")
        print("✅ Analyzed your thinking patterns")
        print("✅ Generated proactive insights")
        print("✅ Built a knowledge graph of your concepts")
        print("✅ Created an intelligent learning schedule")
        print("✅ Identified knowledge gaps and connections")
        print("✅ Provided contextual recommendations")
        
        print(f"\n🚀 SUM is now truly proactive, anticipating your needs")
        print(f"   and enhancing your cognitive capabilities!")
        
    except Exception as e:
        print(f"\n❌ Demonstration error: {e}")
        import traceback
        traceback.print_exc()


def demo_integration():
    """Demonstrate integration with Knowledge OS."""
    print_header("🔗 Knowledge OS Integration Demo", 80)
    
    try:
        # Initialize integrated system
        print("🚀 Initializing Enhanced Knowledge OS...")
        enhanced_os = PredictiveKnowledgeOS(
            data_dir="demo_integrated_data", 
            enable_predictive=True
        )
        print("✅ Enhanced Knowledge OS ready!")
        
        # Demonstrate enhanced capture
        print_section("📝 Enhanced Thought Capture")
        
        sample_thoughts = [
            "Exploring the relationship between consciousness and artificial intelligence",
            "How do neural networks actually learn? The backpropagation algorithm is fascinating",
            "I'm noticing patterns in how I think about complex problems - always breaking them down first"
        ]
        
        for i, thought in enumerate(sample_thoughts, 1):
            print(f"\n💭 Capturing: {thought[:60]}...")
            
            # Get intelligent prompt
            prompt = enhanced_os.get_capture_prompt()
            print(f"🎯 AI Prompt: {prompt}")
            
            # Capture thought
            captured = enhanced_os.capture_thought(thought)
            print(f"✨ Captured as: {captured.id}")
            
            # Get immediate insights
            insights = enhanced_os.get_proactive_insights(1)
            if insights:
                print(f"💡 Immediate insight: {insights[0]['content'][:80]}...")
        
        # Show enhanced insights
        print_section("🔮 Enhanced System Insights")
        
        system_insights = enhanced_os.get_system_insights()
        print(f"📊 {system_insights.get('beautiful_summary', 'Analysis in progress...')}")
        
        if system_insights.get('predictive_intelligence'):
            pi_info = system_insights['predictive_intelligence']
            print(f"\n🤖 Predictive Intelligence Status:")
            print(f"   Knowledge Graph: {pi_info['knowledge_graph_size']['nodes']} concepts")
            print(f"   Proactive Insights: {pi_info['proactive_insights_available']} ready")
        
        print(f"\n🎉 Integration successful! Knowledge OS is now predictively intelligent.")
        
    except Exception as e:
        print(f"❌ Integration demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    
    # Also demo integration
    demo_integration()