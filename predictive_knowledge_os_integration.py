#!/usr/bin/env python3
"""
predictive_knowledge_os_integration.py - Integration layer for Predictive Intelligence System with Knowledge OS

This module seamlessly integrates the new Predictive Intelligence System with the existing
Knowledge OS, creating a unified experience that transforms SUM from reactive to proactive.

Features:
- Seamless integration with existing Knowledge OS workflow
- Enhanced thought capture with predictive insights
- Proactive suggestions during user sessions
- Intelligent scheduling integrated with capture sessions
- Beautiful notification system that respects user flow

Author: ototao
License: Apache License 2.0  
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Import existing Knowledge OS components
try:
    from knowledge_os import (
        KnowledgeOperatingSystem, Thought, KnowledgeCluster, 
        IntuitiveCaptureEngine, BackgroundIntelligenceEngine
    )
except ImportError:
    # Fallback for testing
    print("Knowledge OS not found, running with limited functionality")

# Import the new Predictive Intelligence System
from predictive_intelligence import PredictiveIntelligenceSystem, UserProfile

logger = logging.getLogger('PredictiveKnowledgeOS')


class EnhancedIntuitiveCaptureEngine(IntuitiveCaptureEngine):
    """
    Enhanced capture engine that integrates predictive intelligence
    with the existing beautiful capture experience.
    """
    
    def __init__(self, predictive_system: PredictiveIntelligenceSystem):
        super().__init__()
        self.predictive_system = predictive_system
        self.last_insight_delivery = datetime.now()
        
    def get_capture_prompt(self, context: Dict[str, Any] = None) -> str:
        """Enhanced capture prompt with predictive context awareness."""
        # Get base prompt from original system
        base_prompt = super().get_capture_prompt(context)
        
        # Add predictive intelligence enhancements
        contextual_recommendations = self.predictive_system.get_contextual_recommendations()
        
        # Check if we should add a contextual hint
        if context and context.get('recent_thoughts', 0) > 2:
            # Get emerging interests for contextual prompts
            context_engine = self.predictive_system.context_engine
            emerging_interests = context_engine.get_emerging_interests()
            
            if emerging_interests:
                interest = emerging_interests[0].replace('_', ' ')
                return f"You've been exploring {interest} lately. {base_prompt}"
        
        # Check for connection opportunities
        if contextual_recommendations['connection_opportunities']:
            opportunity = contextual_recommendations['connection_opportunities'][0]
            if self.should_suggest_connection():
                return f"I notice connections forming around {opportunity['cluster_name']}. What's emerging in your thinking?"
        
        return base_prompt
    
    def should_suggest_connection(self) -> bool:
        """Determine if now is a good time to suggest connections."""
        # Respect user's interruption tolerance
        user_profile = self.predictive_system.context_engine.user_profile
        
        # Only suggest occasionally to avoid overwhelming
        time_since_last = (datetime.now() - self.last_insight_delivery).total_seconds() / 60
        
        return (time_since_last > 15 and  # At least 15 minutes since last insight
                user_profile.interruption_tolerance > 0.5)  # User is okay with interruptions
    
    def capture_thought(self, content: str, source: str = "direct") -> Thought:
        """Enhanced thought capture with predictive intelligence processing."""
        # Capture thought using original system
        thought = super().capture_thought(content, source)
        
        if not thought:
            return None
        
        # Process through predictive intelligence system
        try:
            pi_result = self.predictive_system.process_new_thought(content, thought.id)
            
            # Store predictive intelligence metadata in thought
            if hasattr(thought, 'metadata'):
                thought.metadata.update({
                    'pi_concepts': pi_result.get('new_concepts', []),
                    'pi_suggestions_generated': pi_result.get('suggestions_generated', 0),
                    'pi_graph_updated': pi_result.get('graph_updated', False)
                })
            
        except Exception as e:
            logger.error(f"Predictive intelligence processing failed: {e}")
            # Continue gracefully without PI features
        
        return thought


class PredictiveKnowledgeOS(KnowledgeOperatingSystem):
    """
    Enhanced Knowledge OS that integrates predictive intelligence capabilities
    while maintaining the beautiful, intuitive experience of the original system.
    """
    
    def __init__(self, data_dir: str = "knowledge_os_data", enable_predictive: bool = True):
        # Initialize base Knowledge OS
        super().__init__(data_dir)
        
        # Initialize predictive intelligence system
        self.enable_predictive = enable_predictive
        if enable_predictive:
            pi_data_dir = os.path.join(data_dir, "predictive_intelligence")
            self.predictive_system = PredictiveIntelligenceSystem(pi_data_dir)
            
            # Replace capture engine with enhanced version
            self.capture_engine = EnhancedIntuitiveCaptureEngine(self.predictive_system)
            
        logger.info(f"Predictive Knowledge OS initialized (Predictive: {enable_predictive})")
    
    def capture_thought(self, content: str, source: str = "direct") -> Thought:
        """Enhanced thought capture with predictive intelligence."""
        thought = super().capture_thought(content, source)
        
        if not thought or not self.enable_predictive:
            return thought
        
        # Process through predictive system if enabled
        try:
            # The enhanced capture engine already processes through PI
            # Just ensure thought is properly stored with PI data
            pass
            
        except Exception as e:
            logger.error(f"Predictive processing error: {e}")
        
        return thought
    
    def get_capture_prompt(self) -> str:
        """Get intelligent, context-aware capture prompt."""
        if not self.enable_predictive:
            return super().get_capture_prompt()
        
        # Build enhanced context
        context = {
            'recent_thoughts': len([t for t in self.active_thoughts.values() 
                                  if (datetime.now() - t.timestamp).seconds < 3600]),
            'last_topic': self._get_recent_topic(),
            'active_threads': len(self.predictive_system.context_engine.active_threads) if self.enable_predictive else 0
        }
        
        return self.capture_engine.get_capture_prompt(context)
    
    def get_proactive_insights(self, max_insights: int = 3) -> List[Dict[str, Any]]:
        """Get proactive insights ready for delivery."""
        if not self.enable_predictive:
            return []
        
        return self.predictive_system.get_proactive_insights(max_insights)
    
    def get_intelligent_recommendations(self) -> Dict[str, Any]:
        """Get contextual recommendations for the current session."""
        if not self.enable_predictive:
            return {
                'immediate_actions': [],
                'suggested_explorations': [], 
                'connection_opportunities': [],
                'learning_optimizations': []
            }
        
        return self.predictive_system.get_contextual_recommendations()
    
    def analyze_thinking_patterns(self) -> Dict[str, Any]:
        """Analyze user's thinking patterns and provide insights."""
        if not self.enable_predictive:
            return {'status': 'predictive_intelligence_disabled'}
        
        return self.predictive_system.analyze_thinking_patterns()
    
    def get_intelligent_schedule(self, days: int = 7) -> Dict[str, Any]:
        """Get AI-optimized schedule for learning and knowledge work."""
        if not self.enable_predictive:
            return {'status': 'predictive_intelligence_disabled'}
        
        return self.predictive_system.get_intelligent_schedule(days)
    
    def check_densification_opportunities(self) -> List[Dict[str, Any]]:
        """Enhanced densification check with predictive intelligence."""
        # Get base opportunities from original system
        base_opportunities = super().check_densification_opportunities()
        
        if not self.enable_predictive:
            return base_opportunities
        
        # Enhance with predictive intelligence insights
        enhanced_opportunities = []
        
        for opportunity in base_opportunities:
            enhanced_opp = opportunity.copy()
            
            # Add predictive insights about timing and approach
            concept = opportunity['concept']
            
            # Get learning curve analysis
            try:
                learning_curve = self.predictive_system.scheduling_engine.get_learning_curve_optimization(concept)
                enhanced_opp['learning_phase'] = learning_curve.get('phase', 'unknown')
                enhanced_opp['pi_recommendation'] = learning_curve.get('recommendation', '')
                enhanced_opp['optimal_approach'] = learning_curve.get('suggested_next_action', {})
                
            except Exception as e:
                logger.warning(f"Could not enhance opportunity with PI: {e}")
            
            enhanced_opportunities.append(enhanced_opp)
        
        return enhanced_opportunities
    
    def get_system_insights(self) -> Dict[str, Any]:
        """Enhanced system insights with predictive intelligence."""
        # Get base insights from original system
        base_insights = super().get_system_insights()
        
        if not self.enable_predictive:
            return base_insights
        
        # Add predictive intelligence insights
        try:
            pi_patterns = self.predictive_system.analyze_thinking_patterns()
            pi_status = self.predictive_system.get_system_status()
            
            # Merge insights
            enhanced_insights = base_insights.copy()
            enhanced_insights.update({
                'predictive_intelligence': {
                    'thinking_patterns': pi_patterns,
                    'system_status': pi_status,
                    'proactive_insights_available': len(self.predictive_system.active_insights),
                    'knowledge_graph_size': pi_status['knowledge_graph']['graph_size']
                }
            })
            
            # Enhanced beautiful summary
            if pi_patterns.get('status') != 'insufficient_data':
                pi_assessment = pi_patterns.get('overall_assessment', '')
                original_summary = base_insights.get('beautiful_summary', '')
                
                enhanced_insights['beautiful_summary'] = f"{original_summary} {pi_assessment}"
            
            return enhanced_insights
            
        except Exception as e:
            logger.error(f"Could not enhance insights with PI: {e}")
            return base_insights
    
    def start_enhanced_session(self) -> Dict[str, Any]:
        """Start an enhanced thinking session with predictive intelligence."""
        # Start regular session
        session = self.capture_engine.start_session()
        
        if not self.enable_predictive:
            return {'session': session, 'recommendations': []}
        
        # Get initial recommendations
        recommendations = self.get_intelligent_recommendations()
        
        # Get any pending insights
        pending_insights = self.get_proactive_insights(2)
        
        return {
            'session': session,
            'recommendations': recommendations,
            'pending_insights': pending_insights,
            'capture_prompt': self.get_capture_prompt()
        }
    
    def end_enhanced_session(self) -> Dict[str, Any]:
        """End session with predictive intelligence summary."""
        # End regular session
        completed_session = self.capture_engine.end_session()
        
        if not self.enable_predictive or not completed_session:
            return {'session': completed_session}
        
        # Generate session insights
        session_insights = {
            'thoughts_captured': completed_session.thoughts_captured,
            'new_concepts_discovered': 0,
            'connections_made': 0,
            'synthesis_opportunities': 0
        }
        
        try:
            # Analyze session impact on knowledge graph
            pi_status = self.predictive_system.get_system_status()
            recent_growth = pi_status['knowledge_graph']['growth_metrics']
            
            session_insights.update({
                'new_concepts_discovered': recent_growth.get('recent_concept_growth', 0),
                'connections_made': recent_growth.get('recent_connection_growth', 0)
            })
            
            # Check for synthesis opportunities
            recommendations = self.get_intelligent_recommendations()
            session_insights['synthesis_opportunities'] = len(recommendations.get('immediate_actions', []))
            
        except Exception as e:
            logger.warning(f"Could not generate PI session insights: {e}")
        
        return {
            'session': completed_session,
            'session_insights': session_insights,
            'next_steps': self._generate_next_steps()
        }
    
    def _generate_next_steps(self) -> List[str]:
        """Generate intelligent next steps for the user."""
        if not self.enable_predictive:
            return ["Continue capturing your thoughts as they flow."]
        
        try:
            recommendations = self.get_intelligent_recommendations()
            next_steps = []
            
            # Add immediate actions
            for action in recommendations.get('immediate_actions', [])[:2]:
                if action['type'] == 'synthesis_opportunity':
                    next_steps.append(f"Consider synthesizing your thoughts on {action['thread']} - they're ready!")
            
            # Add exploration suggestions
            for exploration in recommendations.get('suggested_explorations', [])[:1]:
                next_steps.append(f"Explore {exploration['suggested_area']} - it connects interestingly to your current thinking")
            
            # Add learning optimizations
            for optimization in recommendations.get('learning_optimizations', [])[:1]:
                next_steps.append(f"Your peak thinking time is {optimization['time']} - consider scheduling important work then")
            
            return next_steps[:3] if next_steps else ["Continue capturing your thoughts as they flow."]
            
        except Exception as e:
            logger.warning(f"Could not generate next steps: {e}")
            return ["Continue capturing your thoughts as they flow."]


def create_enhanced_cli():
    """Create a beautiful CLI that showcases the enhanced predictive Knowledge OS."""
    print("\n" + "="*80)
    print("🧠✨ Enhanced SUM Knowledge OS with Predictive Intelligence")
    print("    From reactive to proactive - your AI-powered thinking companion")
    print("="*80)
    
    # Initialize enhanced system
    print("\n🚀 Initializing Enhanced Knowledge OS...")
    try:
        knowledge_os = PredictiveKnowledgeOS(enable_predictive=True)
        print("✅ Predictive Intelligence System: ACTIVE")
    except Exception as e:
        print(f"⚠️  Predictive Intelligence initialization failed: {e}")
        print("📝 Falling back to standard Knowledge OS")
        knowledge_os = PredictiveKnowledgeOS(enable_predictive=False)
    
    # Start enhanced session
    session_info = knowledge_os.start_enhanced_session()
    print(f"\n🎯 Session started: {session_info['session'].session_id}")
    
    # Show initial insights if available
    if session_info.get('pending_insights'):
        print(f"\n💡 Welcome back! I have {len(session_info['pending_insights'])} insights for you:")
        for insight in session_info['pending_insights']:
            print(f"   • {insight['content'][:100]}...")
    
    print(f"\n{session_info.get('capture_prompt', 'What\'s on your mind?')}")
    print("\nCommands: 'insights', 'patterns', 'schedule', 'recommendations', 'densify', 'status', 'quit'")
    
    while True:
        try:
            user_input = input("\n💭 ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                # End session gracefully
                session_summary = knowledge_os.end_enhanced_session()
                print(f"\n✨ Session completed!")
                
                if session_summary.get('session_insights'):
                    insights = session_summary['session_insights']
                    print(f"   📊 {insights['thoughts_captured']} thoughts captured")
                    if insights['new_concepts_discovered']:
                        print(f"   🔍 {insights['new_concepts_discovered']} new concepts discovered")
                    if insights['connections_made']:
                        print(f"   🔗 {insights['connections_made']} new connections made")
                
                if session_summary.get('next_steps'):
                    print(f"\n🎯 Suggested next steps:")
                    for step in session_summary['next_steps']:
                        print(f"   • {step}")
                
                print(f"\n✨ Your thoughts continue to evolve. Until next time!")
                break
            
            elif user_input.lower() == 'insights':
                insights = knowledge_os.get_proactive_insights(3)
                if insights:
                    print(f"\n🔮 {len(insights)} proactive insights ready:")
                    for i, insight in enumerate(insights, 1):
                        print(f"\n{i}. {insight['content']}")
                        print(f"   💡 Type: {insight['insight_type']} | Confidence: {insight.get('confidence', 0):.1%}")
                        if insight.get('actions'):
                            print(f"   🎯 Actions: {', '.join([a['label'] for a in insight['actions'][:2]])}")
                else:
                    print("\n✨ No new insights right now. Keep thinking to generate more!")
            
            elif user_input.lower() == 'patterns':
                analysis = knowledge_os.analyze_thinking_patterns()
                if analysis.get('status') == 'insufficient_data':
                    print(f"\n📊 {analysis.get('message', 'Need more thoughts for pattern analysis')}")
                elif analysis.get('status') == 'predictive_intelligence_disabled':
                    print(f"\n📊 Predictive intelligence is not enabled")
                else:
                    print(f"\n📊 Your Thinking Patterns:")
                    print(f"   {analysis.get('overall_assessment', 'Analysis in progress...')}")
                    
                    if analysis.get('improvement_suggestions'):
                        print(f"\n🎯 Personalized Suggestions:")
                        for suggestion in analysis['improvement_suggestions'][:3]:
                            print(f"   • {suggestion.get('suggestion', 'Continue exploring')}")
            
            elif user_input.lower() == 'schedule':
                schedule = knowledge_os.get_intelligent_schedule(3)
                if schedule.get('status') == 'predictive_intelligence_disabled':
                    print(f"\n📅 Intelligent scheduling requires predictive intelligence")
                else:
                    print(f"\n📅 Your Intelligent Schedule (Next 3 Days):")
                    
                    daily_plans = schedule.get('daily_recommendations', {})
                    for date_key, plan in list(daily_plans.items())[:3]:
                        print(f"\n📆 {plan['date']}:")
                        
                        if plan.get('scheduled_reviews'):
                            reviews = [r['concept'].replace('_', ' ') for r in plan['scheduled_reviews']]
                            print(f"   📚 Reviews: {', '.join(reviews)}")
                        
                        if plan.get('synthesis_opportunities'):
                            synthesis = [s['thread_title'] for s in plan['synthesis_opportunities']]
                            print(f"   🎯 Synthesis: {', '.join(synthesis)}")
                        
                        if not plan.get('scheduled_reviews') and not plan.get('synthesis_opportunities'):
                            print(f"   ✨ Perfect day for open exploration")
                    
                    if schedule.get('weekly_goals'):
                        print(f"\n🎯 This Week's Focus:")
                        for goal in schedule['weekly_goals'][:3]:
                            print(f"   • {goal['goal']} for {goal['thread']}")
            
            elif user_input.lower() == 'recommendations':
                recommendations = knowledge_os.get_intelligent_recommendations()
                print(f"\n🎯 Contextual Recommendations:")
                
                if recommendations.get('immediate_actions'):
                    print(f"\n⚡ Immediate Opportunities:")
                    for action in recommendations['immediate_actions'][:2]:
                        print(f"   • {action['description']}")
                        if action.get('best_time'):
                            print(f"     ⏰ Best time: {action['best_time']}")
                
                if recommendations.get('suggested_explorations'):
                    print(f"\n🔍 Suggested Explorations:")
                    for exploration in recommendations['suggested_explorations'][:2]:
                        print(f"   • {exploration['description']}")
                
                if recommendations.get('connection_opportunities'):
                    print(f"\n🔗 Connection Opportunities:")
                    for connection in recommendations['connection_opportunities'][:2]:
                        print(f"   • {connection['description']}")
                
                if not any(recommendations.values()):
                    print(f"   ✨ No specific recommendations right now. Continue exploring!")
            
            elif user_input.lower() == 'densify':
                opportunities = knowledge_os.check_densification_opportunities()
                if opportunities:
                    print(f"\n🎯 {len(opportunities)} Densification Opportunities:")
                    for opp in opportunities[:3]:
                        concept = opp['concept'].replace('-', ' ').title()
                        print(f"\n📦 {concept}:")
                        print(f"   • {len(opp['thoughts'])} thoughts ready for synthesis")
                        print(f"   • {opp['analysis']['suggestion']}")
                        
                        if opp.get('pi_recommendation'):
                            print(f"   • 🤖 AI Insight: {opp['pi_recommendation']}")
                else:
                    print(f"\n✨ Your thoughts are well-organized! No densification needed right now.")
            
            elif user_input.lower() == 'status':
                insights = knowledge_os.get_system_insights()
                print(f"\n⚡ System Status:")
                print(f"   🧠 Thoughts: {insights['thinking_patterns']['total_thoughts']}")
                print(f"   🎯 Processing: {insights['thinking_patterns']['processed_thoughts']}")
                print(f"   ⭐ Avg Importance: {insights['thinking_patterns']['average_importance']:.1%}")
                print(f"   ⏰ Peak Hour: {insights['thinking_patterns']['peak_description']}")
                
                if insights.get('predictive_intelligence'):
                    pi_info = insights['predictive_intelligence']
                    graph_size = pi_info['system_status']['knowledge_graph']['graph_size']
                    print(f"\n🤖 Predictive Intelligence:")
                    print(f"   📊 Knowledge Graph: {graph_size['nodes']} concepts, {graph_size['edges']} connections")
                    print(f"   💡 Insights Available: {pi_info['proactive_insights_available']}")
                    print(f"   📈 System Health: {pi_info['system_status']['system_health'].title()}")
            
            else:
                # Capture as a thought
                thought = knowledge_os.capture_thought(user_input)
                
                if thought:
                    print(f"✨ Captured: {thought.id}")
                    
                    # Show any immediate insights generated
                    immediate_insights = knowledge_os.get_proactive_insights(1)
                    if immediate_insights:
                        insight = immediate_insights[0]
                        print(f"\n💡 {insight['content']}")
                    
                    # Show new prompt
                    print(f"\n{knowledge_os.get_capture_prompt()}")
        
        except KeyboardInterrupt:
            print(f"\n\n✨ Session paused. Your thoughts are safely stored.")
            break
        except Exception as e:
            print(f"\n❌ Something went wrong: {e}")
            continue


if __name__ == "__main__":
    create_enhanced_cli()