#!/usr/bin/env python3
"""
ai_enhanced_interface.py - AI-Enhanced Interface Components

Strategic AI placement for optimal user experience:
- Smart suggestions powered by local AI
- Contextual help and insights
- Predictive email filtering rules
- Intelligent auto-categorization
- Real-time processing optimization

Author: ototao
License: Apache License 2.0
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from ollama_manager import OllamaManager, ProcessingRequest
from summail_engine import SumMailEngine, EmailCategory, Priority

logger = logging.getLogger(__name__)


@dataclass
class AIInsight:
    """AI-generated insight for interface enhancement."""
    type: str  # 'suggestion', 'warning', 'optimization', 'pattern'
    title: str
    description: str
    confidence: float
    action: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class SmartSuggestionEngine:
    """AI-powered suggestion engine for optimal user experience."""
    
    def __init__(self, ollama_manager: OllamaManager):
        self.ollama_manager = ollama_manager
        self.user_patterns = defaultdict(dict)
        self.suggestion_cache = {}
        
    def analyze_user_behavior(self, user_id: str, actions: List[Dict[str, Any]]) -> List[AIInsight]:
        """Analyze user behavior and generate intelligent suggestions."""
        insights = []
        
        # Pattern detection
        action_patterns = self._detect_patterns(actions)
        
        # Time-based optimization
        time_insights = self._analyze_timing_patterns(actions)
        insights.extend(time_insights)
        
        # Workflow optimization
        workflow_insights = self._suggest_workflow_improvements(action_patterns)
        insights.extend(workflow_insights)
        
        # AI-powered content insights
        if self.ollama_manager and self.ollama_manager.available_models:
            ai_insights = self._generate_ai_insights(actions)
            insights.extend(ai_insights)
        
        return insights
    
    def _detect_patterns(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in user actions."""
        patterns = {
            'frequent_categories': Counter(),
            'processing_times': [],
            'preferred_modes': Counter(),
            'error_patterns': [],
            'success_patterns': []
        }
        
        for action in actions:
            if 'category' in action:
                patterns['frequent_categories'][action['category']] += 1
            
            if 'mode' in action:
                patterns['preferred_modes'][action['mode']] += 1
            
            if 'processing_time' in action:
                patterns['processing_times'].append(action['processing_time'])
            
            if action.get('success', True):
                patterns['success_patterns'].append(action)
            else:
                patterns['error_patterns'].append(action)
        
        return patterns
    
    def _analyze_timing_patterns(self, actions: List[Dict[str, Any]]) -> List[AIInsight]:
        """Analyze timing patterns and suggest optimizations."""
        insights = []
        
        if not actions:
            return insights
        
        # Analyze processing times
        processing_times = [a.get('processing_time', 0) for a in actions if a.get('processing_time')]
        
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            if avg_time > 5.0:  # If average processing time > 5 seconds
                insights.append(AIInsight(
                    type='optimization',
                    title='Processing Speed Optimization',
                    description=f'Your average processing time is {avg_time:.1f}s. Consider using smaller models for faster results.',
                    confidence=0.8,
                    action='switch_to_fast_model',
                    data={'current_avg_time': avg_time, 'suggested_models': ['llama3.2:1b', 'qwen2:1.5b']}
                ))
        
        # Analyze usage patterns by time of day
        hourly_usage = defaultdict(int)
        for action in actions:
            if 'timestamp' in action:
                hour = datetime.fromtimestamp(action['timestamp']).hour
                hourly_usage[hour] += 1
        
        if hourly_usage:
            peak_hour = max(hourly_usage.items(), key=lambda x: x[1])[0]
            insights.append(AIInsight(
                type='pattern',
                title='Usage Pattern Detected',
                description=f'You typically process most content around {peak_hour}:00. Consider scheduling batch processing during this time.',
                confidence=0.7,
                data={'peak_hour': peak_hour, 'usage_distribution': dict(hourly_usage)}
            ))
        
        return insights
    
    def _suggest_workflow_improvements(self, patterns: Dict[str, Any]) -> List[AIInsight]:
        """Suggest workflow improvements based on patterns."""
        insights = []
        
        # Suggest batch processing for frequent users
        if patterns['frequent_categories']:
            most_common = patterns['frequent_categories'].most_common(1)[0]
            if most_common[1] > 10:  # More than 10 items of same category
                insights.append(AIInsight(
                    type='suggestion',
                    title='Batch Processing Opportunity',
                    description=f'You frequently process {most_common[0]} content. Consider using batch mode for better efficiency.',
                    confidence=0.9,
                    action='enable_batch_mode',
                    data={'category': most_common[0], 'frequency': most_common[1]}
                ))
        
        # Suggest mode optimization
        if patterns['preferred_modes']:
            preferred = patterns['preferred_modes'].most_common(1)[0]
            if preferred[0] == 'text' and preferred[1] > 20:
                insights.append(AIInsight(
                    type='suggestion',
                    title='Multi-Modal Upgrade',
                    description='You primarily use text processing. Try our multi-modal features for PDFs and images!',
                    confidence=0.6,
                    action='try_multimodal'
                ))
        
        # Error pattern analysis
        if patterns['error_patterns']:
            common_errors = Counter(e.get('error_type', 'unknown') for e in patterns['error_patterns'])
            most_common_error = common_errors.most_common(1)[0]
            
            insights.append(AIInsight(
                type='warning',
                title='Recurring Issue Detected',
                description=f'You\'ve encountered {most_common_error[0]} errors {most_common_error[1]} times. Check our troubleshooting guide.',
                confidence=0.8,
                action='show_troubleshooting',
                data={'error_type': most_common_error[0], 'frequency': most_common_error[1]}
            ))
        
        return insights
    
    def _generate_ai_insights(self, actions: List[Dict[str, Any]]) -> List[AIInsight]:
        """Generate AI-powered insights about user behavior."""
        insights = []
        
        if not actions or len(actions) < 5:
            return insights
        
        try:
            # Create summary of user activity
            activity_summary = self._create_activity_summary(actions)
            
            # Ask AI for insights
            prompt = f"""
            Analyze this user's activity and provide 2-3 actionable insights for improving their workflow:
            
            Activity Summary:
            {activity_summary}
            
            Focus on:
            1. Efficiency improvements
            2. Feature recommendations
            3. Workflow optimizations
            
            Respond in JSON format:
            {{"insights": [{{"title": "...", "description": "...", "confidence": 0.8}}]}}
            """
            
            request = ProcessingRequest(
                text=prompt,
                task_type='analysis',
                max_tokens=300,
                temperature=0.3
            )
            
            response = self.ollama_manager.process_text(request)
            
            # Parse AI response
            try:
                ai_data = json.loads(response.response)
                for insight_data in ai_data.get('insights', []):
                    insights.append(AIInsight(
                        type='suggestion',
                        title=insight_data.get('title', 'AI Suggestion'),
                        description=insight_data.get('description', ''),
                        confidence=insight_data.get('confidence', 0.7),
                        data={'source': 'ai_analysis', 'model': response.model_used}
                    ))
            except json.JSONDecodeError:
                # Fallback to text parsing
                insights.append(AIInsight(
                    type='suggestion',
                    title='AI Workflow Analysis',
                    description=response.response[:200] + '...' if len(response.response) > 200 else response.response,
                    confidence=0.6,
                    data={'source': 'ai_analysis', 'model': response.model_used}
                ))
        
        except Exception as e:
            logger.warning(f"AI insight generation failed: {e}")
        
        return insights
    
    def _create_activity_summary(self, actions: List[Dict[str, Any]]) -> str:
        """Create a summary of user activity for AI analysis."""
        total_actions = len(actions)
        modes_used = Counter(a.get('mode', 'unknown') for a in actions)
        categories = Counter(a.get('category', 'unknown') for a in actions)
        avg_processing_time = sum(a.get('processing_time', 0) for a in actions) / max(total_actions, 1)
        
        success_rate = sum(1 for a in actions if a.get('success', True)) / total_actions * 100
        
        return f"""
        Total Actions: {total_actions}
        Modes Used: {dict(modes_used)}
        Categories: {dict(categories)}
        Average Processing Time: {avg_processing_time:.2f}s
        Success Rate: {success_rate:.1f}%
        Time Period: {datetime.fromtimestamp(actions[0].get('timestamp', time.time())).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(actions[-1].get('timestamp', time.time())).strftime('%Y-%m-%d')}
        """


class EmailFilterEngine:
    """AI-powered email filtering and auto-forwarding system."""
    
    def __init__(self, summail_engine: SumMailEngine, ollama_manager: OllamaManager):
        self.summail_engine = summail_engine
        self.ollama_manager = ollama_manager
        self.learned_rules = []
        self.filter_accuracy = defaultdict(float)
    
    def generate_filter_rules(self, processed_emails: List[Any]) -> List[Dict[str, Any]]:
        """Generate intelligent email filter rules based on processed emails."""
        rules = []
        
        # Sender-based rules
        sender_categories = self._analyze_sender_patterns(processed_emails)
        rules.extend(self._create_sender_rules(sender_categories))
        
        # Subject-based rules
        subject_patterns = self._analyze_subject_patterns(processed_emails)
        rules.extend(self._create_subject_rules(subject_patterns))
        
        # Content-based rules (AI-powered)
        if self.ollama_manager and self.ollama_manager.available_models:
            content_rules = self._generate_ai_content_rules(processed_emails)
            rules.extend(content_rules)
        
        return rules
    
    def _analyze_sender_patterns(self, emails: List[Any]) -> Dict[str, List[EmailCategory]]:
        """Analyze sender patterns to create rules."""
        sender_categories = defaultdict(list)
        
        for email in emails:
            domain = email.metadata.sender_domain
            category = email.metadata.category
            sender_categories[domain].append(category)
        
        # Find consistent patterns
        consistent_senders = {}
        for domain, categories in sender_categories.items():
            category_counts = Counter(categories)
            most_common = category_counts.most_common(1)[0]
            
            # If 80%+ of emails from this sender are same category
            if most_common[1] / len(categories) >= 0.8:
                consistent_senders[domain] = most_common[0]
        
        return consistent_senders
    
    def _create_sender_rules(self, sender_categories: Dict[str, EmailCategory]) -> List[Dict[str, Any]]:
        """Create filter rules based on sender analysis."""
        rules = []
        
        for domain, category in sender_categories.items():
            priority = Priority.LOW if category in [EmailCategory.PROMOTIONAL, EmailCategory.SOCIAL_MEDIA] else Priority.MEDIUM
            
            rules.append({
                'type': 'sender_domain',
                'condition': domain,
                'action': {
                    'category': category.value,
                    'priority': priority.value,
                    'auto_process': True
                },
                'confidence': 0.9,
                'description': f'Auto-categorize emails from {domain} as {category.value}'
            })
        
        return rules
    
    def _analyze_subject_patterns(self, emails: List[Any]) -> List[Dict[str, Any]]:
        """Analyze subject line patterns."""
        patterns = []
        
        # Newsletter patterns
        newsletter_subjects = [e.metadata.subject for e in emails if e.metadata.category == EmailCategory.NEWSLETTER]
        if newsletter_subjects:
            common_words = self._find_common_words(newsletter_subjects)
            for word in common_words:
                patterns.append({
                    'pattern': f'*{word}*',
                    'category': EmailCategory.NEWSLETTER,
                    'confidence': 0.7
                })
        
        # Software update patterns
        update_subjects = [e.metadata.subject for e in emails if e.metadata.category == EmailCategory.SOFTWARE_UPDATE]
        if update_subjects:
            common_words = self._find_common_words(update_subjects)
            for word in common_words:
                patterns.append({
                    'pattern': f'*{word}*',
                    'category': EmailCategory.SOFTWARE_UPDATE,
                    'confidence': 0.8
                })
        
        return patterns
    
    def _find_common_words(self, subjects: List[str], min_frequency: int = 3) -> List[str]:
        """Find commonly occurring words in subjects."""
        all_words = []
        for subject in subjects:
            words = [word.lower() for word in subject.split() if len(word) > 3]
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.items() if count >= min_frequency]
    
    def _create_subject_rules(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create filter rules from subject patterns."""
        rules = []
        
        for pattern in patterns:
            rules.append({
                'type': 'subject_contains',
                'condition': pattern['pattern'],
                'action': {
                    'category': pattern['category'].value,
                    'auto_process': True
                },
                'confidence': pattern['confidence'],
                'description': f'Auto-categorize emails with subject containing "{pattern["pattern"]}"'
            })
        
        return rules
    
    def _generate_ai_content_rules(self, emails: List[Any]) -> List[Dict[str, Any]]:
        """Generate content-based rules using AI analysis."""
        rules = []
        
        try:
            # Sample different categories
            categories_sample = {}
            for category in EmailCategory:
                category_emails = [e for e in emails if e.metadata.category == category]
                if category_emails:
                    categories_sample[category] = category_emails[:3]  # Sample 3 emails
            
            # Ask AI to identify patterns
            for category, sample_emails in categories_sample.items():
                content_samples = [e.summary[:200] for e in sample_emails]
                
                prompt = f"""
                Analyze these {category.value} email summaries and identify 2-3 key patterns or phrases that could help automatically categorize similar emails:
                
                Samples:
                {chr(10).join(f"- {sample}" for sample in content_samples)}
                
                Respond with specific phrases or patterns (not generic descriptions):
                """
                
                request = ProcessingRequest(
                    text=prompt,
                    task_type='analysis',
                    max_tokens=150,
                    temperature=0.2
                )
                
                response = self.ollama_manager.process_text(request)
                
                # Parse response for patterns
                patterns = self._extract_patterns_from_ai_response(response.response)
                
                for pattern in patterns:
                    rules.append({
                        'type': 'content_contains',
                        'condition': pattern,
                        'action': {
                            'category': category.value,
                            'confidence_boost': 0.2
                        },
                        'confidence': 0.6,
                        'description': f'Boost {category.value} confidence for emails containing "{pattern}"',
                        'source': 'ai_analysis'
                    })
        
        except Exception as e:
            logger.warning(f"AI rule generation failed: {e}")
        
        return rules
    
    def _extract_patterns_from_ai_response(self, response: str) -> List[str]:
        """Extract patterns from AI response."""
        patterns = []
        
        # Simple extraction - look for quoted phrases or bullet points
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for quoted phrases
            if '"' in line:
                start = line.find('"')
                end = line.find('"', start + 1)
                if start != -1 and end != -1:
                    pattern = line[start + 1:end].strip()
                    if len(pattern) > 3:
                        patterns.append(pattern)
            
            # Look for bullet points or lists
            elif line.startswith(('-', '*', '•')) and len(line) > 5:
                pattern = line[1:].strip()
                if len(pattern) > 3:
                    patterns.append(pattern)
        
        return patterns[:3]  # Limit to 3 patterns per category
    
    def create_auto_forward_config(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create configuration for email auto-forwarding to SumMail."""
        config = {
            'forwarding_rules': rules,
            'processing_settings': {
                'auto_process': True,
                'generate_daily_digest': True,
                'compress_newsletters': True,
                'extract_action_items': True,
                'track_software_updates': True
            },
            'notification_settings': {
                'critical_emails': 'immediate',
                'high_priority': 'hourly',
                'medium_priority': 'daily_digest',
                'low_priority': 'weekly_digest'
            },
            'privacy_settings': {
                'local_processing_only': True,
                'encrypt_stored_data': True,
                'auto_delete_after_days': 90
            }
        }
        
        return config


class ContextualHelpEngine:
    """Provides contextual AI-powered help and guidance."""
    
    def __init__(self, ollama_manager: OllamaManager):
        self.ollama_manager = ollama_manager
        self.help_cache = {}
    
    def get_contextual_help(self, context: Dict[str, Any]) -> Optional[str]:
        """Get contextual help based on current user context."""
        if not self.ollama_manager or not self.ollama_manager.available_models:
            return self._get_static_help(context)
        
        try:
            context_key = json.dumps(context, sort_keys=True)
            if context_key in self.help_cache:
                return self.help_cache[context_key]
            
            prompt = f"""
            Provide helpful, concise guidance for a user in this context:
            
            Current State: {context.get('current_mode', 'unknown')}
            User Action: {context.get('last_action', 'none')}
            Issues: {context.get('errors', 'none')}
            Experience Level: {context.get('user_level', 'beginner')}
            
            Provide a helpful tip or suggestion in 1-2 sentences. Be specific and actionable.
            """
            
            request = ProcessingRequest(
                text=prompt,
                task_type='assistance',
                max_tokens=100,
                temperature=0.1
            )
            
            response = self.ollama_manager.process_text(request)
            help_text = response.response.strip()
            
            self.help_cache[context_key] = help_text
            return help_text
            
        except Exception as e:
            logger.warning(f"Contextual help generation failed: {e}")
            return self._get_static_help(context)
    
    def _get_static_help(self, context: Dict[str, Any]) -> str:
        """Fallback static help messages."""
        mode = context.get('current_mode', 'text')
        
        static_help = {
            'text': 'Tip: Use the hierarchical engine to get multi-level summaries with key concepts and insights.',
            'multimodal': 'Tip: Upload PDFs, images, or documents for intelligent content extraction and analysis.',
            'email': 'Tip: Connect your email account to start compressing newsletters and extracting action items.',
            'error': 'Try using smaller models or reducing the text size if processing is slow.'
        }
        
        return static_help.get(mode, 'Explore different modes to find the best tool for your content.')


# Enhanced UI Components with Strategic AI Placement
AI_ENHANCED_CSS = """
/* AI Enhancement Styles */
.ai-insight {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 12px;
    margin: 15px 0;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: slideInFromRight 0.5s ease;
}

.ai-insight-icon {
    font-size: 1.5rem;
    opacity: 0.9;
}

.ai-insight-content h4 {
    margin: 0 0 5px 0;
    font-size: 1rem;
    font-weight: 600;
}

.ai-insight-content p {
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.9;
}

.contextual-help {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
    max-width: 300px;
    z-index: 1000;
    transform: translateY(100%);
    transition: transform 0.3s ease;
}

.contextual-help.show {
    transform: translateY(0);
}

.smart-suggestion {
    background: #f0f9ff;
    border: 1px solid #0ea5e9;
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    position: relative;
}

.smart-suggestion::before {
    content: "💡";
    position: absolute;
    top: -8px;
    left: 12px;
    background: #f0f9ff;
    padding: 0 4px;
    font-size: 0.9rem;
}

.processing-optimization {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    padding: 12px;
    border-radius: 6px;
    margin: 10px 0;
}

.pattern-insight {
    background: #fef7cd;
    border-left: 4px solid #eab308;
    padding: 12px;
    border-radius: 6px;
    margin: 10px 0;
}

@keyframes slideInFromRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.ai-status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: rgba(16, 185, 129, 0.1);
    border-radius: 20px;
    font-size: 0.8rem;
    color: #059669;
}

.ai-status-indicator.processing {
    background: rgba(249, 115, 22, 0.1);
    color: #ea580c;
}

.ai-status-indicator::before {
    content: "🤖";
    font-size: 0.9rem;
}
"""

AI_ENHANCED_JS = """
// AI Enhancement JavaScript
class AIEnhancedInterface {
    constructor() {
        this.suggestionEngine = new SmartSuggestionEngine();
        this.contextualHelp = new ContextualHelpEngine();
        this.userActions = [];
        this.currentContext = {};
        
        this.initializeAIFeatures();
    }
    
    initializeAIFeatures() {
        // Track user actions for pattern analysis
        this.trackUserActions();
        
        // Initialize contextual help
        this.initializeContextualHelp();
        
        // Setup smart suggestions
        this.initializeSmartSuggestions();
        
        // Auto-update insights
        setInterval(() => this.updateAIInsights(), 30000); // Every 30 seconds
    }
    
    trackUserActions() {
        // Track form submissions
        document.addEventListener('submit', (e) => {
            this.logAction('form_submit', {
                form_id: e.target.id,
                timestamp: Date.now()
            });
        });
        
        // Track mode switches
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('mode-btn')) {
                this.logAction('mode_switch', {
                    mode: e.target.textContent.trim(),
                    timestamp: Date.now()
                });
            }
        });
    }
    
    logAction(type, data) {
        this.userActions.push({
            type: type,
            ...data
        });
        
        // Keep only last 50 actions
        if (this.userActions.length > 50) {
            this.userActions = this.userActions.slice(-50);
        }
        
        // Update context
        this.updateContext(type, data);
    }
    
    updateContext(actionType, data) {
        this.currentContext = {
            ...this.currentContext,
            last_action: actionType,
            timestamp: Date.now(),
            user_level: this.determineUserLevel()
        };
        
        if (data.mode) {
            this.currentContext.current_mode = data.mode.toLowerCase().split(' ')[0];
        }
    }
    
    determineUserLevel() {
        const actionCount = this.userActions.length;
        const uniqueActions = new Set(this.userActions.map(a => a.type)).size;
        
        if (actionCount < 5) return 'beginner';
        if (actionCount < 20 || uniqueActions < 3) return 'intermediate';
        return 'advanced';
    }
    
    async updateAIInsights() {
        if (this.userActions.length < 3) return;
        
        try {
            // Get AI insights
            const insights = await this.getAIInsights();
            this.displayInsights(insights);
            
            // Update contextual help
            const helpText = await this.getContextualHelp();
            this.updateContextualHelp(helpText);
            
        } catch (error) {
            console.warn('AI insights update failed:', error);
        }
    }
    
    async getAIInsights() {
        const response = await fetch('/api/ai/insights', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                actions: this.userActions.slice(-20), // Last 20 actions
                context: this.currentContext
            })
        });
        
        if (response.ok) {
            return await response.json();
        }
        return [];
    }
    
    displayInsights(insights) {
        const container = document.getElementById('ai-insights') || this.createInsightsContainer();
        container.innerHTML = '';
        
        insights.forEach(insight => {
            const insightElement = this.createInsightElement(insight);
            container.appendChild(insightElement);
        });
    }
    
    createInsightsContainer() {
        const container = document.createElement('div');
        container.id = 'ai-insights';
        container.className = 'ai-insights-container';
        
        // Insert after header
        const header = document.querySelector('.header');
        if (header && header.nextSibling) {
            header.parentNode.insertBefore(container, header.nextSibling);
        }
        
        return container;
    }
    
    createInsightElement(insight) {
        const element = document.createElement('div');
        element.className = `ai-insight ${insight.type}`;
        
        const icon = this.getInsightIcon(insight.type);
        
        element.innerHTML = `
            <div class="ai-insight-icon">${icon}</div>
            <div class="ai-insight-content">
                <h4>${insight.title}</h4>
                <p>${insight.description}</p>
                ${insight.action ? `<button class="btn btn-sm" onclick="performAIAction('${insight.action}')">${this.getActionLabel(insight.action)}</button>` : ''}
            </div>
        `;
        
        return element;
    }
    
    getInsightIcon(type) {
        const icons = {
            'suggestion': '💡',
            'warning': '⚠️',
            'optimization': '⚡',
            'pattern': '📊'
        };
        return icons[type] || '🤖';
    }
    
    getActionLabel(action) {
        const labels = {
            'switch_to_fast_model': 'Use Fast Model',
            'enable_batch_mode': 'Enable Batch Mode',
            'try_multimodal': 'Try Multi-Modal',
            'show_troubleshooting': 'View Help'
        };
        return labels[action] || 'Take Action';
    }
    
    initializeContextualHelp() {
        const helpContainer = document.createElement('div');
        helpContainer.id = 'contextual-help';
        helpContainer.className = 'contextual-help';
        document.body.appendChild(helpContainer);
        
        // Show help when user seems stuck
        let lastActionTime = Date.now();
        setInterval(() => {
            if (Date.now() - lastActionTime > 60000) { // 1 minute of inactivity
                this.showContextualHelp();
            }
        }, 30000);
        
        // Update last action time on any user interaction
        document.addEventListener('click', () => {
            lastActionTime = Date.now();
            this.hideContextualHelp();
        });
    }
    
    async showContextualHelp() {
        const helpText = await this.getContextualHelp();
        this.updateContextualHelp(helpText);
        
        const helpContainer = document.getElementById('contextual-help');
        helpContainer.classList.add('show');
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            this.hideContextualHelp();
        }, 10000);
    }
    
    hideContextualHelp() {
        const helpContainer = document.getElementById('contextual-help');
        helpContainer.classList.remove('show');
    }
    
    async getContextualHelp() {
        try {
            const response = await fetch('/api/ai/help', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({context: this.currentContext})
            });
            
            if (response.ok) {
                const data = await response.json();
                return data.help_text;
            }
        } catch (error) {
            console.warn('Contextual help failed:', error);
        }
        
        return this.getStaticHelp();
    }
    
    updateContextualHelp(helpText) {
        const helpContainer = document.getElementById('contextual-help');
        if (helpContainer && helpText) {
            helpContainer.innerHTML = `
                <div class="help-content">
                    <h4>💡 Tip</h4>
                    <p>${helpText}</p>
                </div>
                <button class="help-close" onclick="aiInterface.hideContextualHelp()">×</button>
            `;
        }
    }
    
    getStaticHelp() {
        const mode = this.currentContext.current_mode || 'text';
        const helpTexts = {
            'text': 'Try using local AI enhancement for better summaries!',
            'multi': 'Upload PDFs or images for intelligent content extraction.',
            'email': 'Connect your email to start compressing newsletters automatically.',
            'default': 'Explore different processing modes to find what works best for you.'
        };
        
        return helpTexts[mode] || helpTexts['default'];
    }
    
    initializeSmartSuggestions() {
        // Add smart suggestions to forms
        this.addSmartSuggestionToTextarea();
        this.addProcessingOptimizations();
    }
    
    addSmartSuggestionToTextarea() {
        const textarea = document.getElementById('textInput');
        if (textarea) {
            textarea.addEventListener('input', debounce((e) => {
                this.analyzeTextInput(e.target.value);
            }, 1000));
        }
    }
    
    analyzeTextInput(text) {
        if (text.length < 100) return;
        
        // Simple heuristics for smart suggestions
        const suggestions = [];
        
        if (text.length > 5000) {
            suggestions.push({
                type: 'optimization',
                message: 'Long text detected. Consider using streaming mode for better performance.',
                action: 'suggest_streaming'
            });
        }
        
        if (text.includes('http') || text.includes('www.')) {
            suggestions.push({
                type: 'suggestion',
                message: 'URLs detected. Try our link extraction feature.',
                action: 'enable_link_extraction'
            });
        }
        
        if (text.match(/\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}/)) {
            suggestions.push({
                type: 'suggestion',
                message: 'Dates detected. Enable timeline analysis for better insights.',
                action: 'enable_timeline'
            });
        }
        
        this.displayTextSuggestions(suggestions);
    }
    
    displayTextSuggestions(suggestions) {
        let container = document.getElementById('text-suggestions');
        if (!container) {
            container = document.createElement('div');
            container.id = 'text-suggestions';
            const textarea = document.getElementById('textInput');
            textarea.parentNode.insertBefore(container, textarea.nextSibling);
        }
        
        container.innerHTML = '';
        suggestions.forEach(suggestion => {
            const div = document.createElement('div');
            div.className = `smart-suggestion ${suggestion.type}`;
            div.innerHTML = `
                <p>${suggestion.message}</p>
                <button class="btn btn-sm" onclick="performTextAction('${suggestion.action}')">${this.getActionLabel(suggestion.action)}</button>
            `;
            container.appendChild(div);
        });
    }
}

// Global AI interface instance
let aiInterface;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    aiInterface = new AIEnhancedInterface();
});

// AI Action handlers
function performAIAction(action) {
    switch(action) {
        case 'switch_to_fast_model':
            // Switch to faster model
            document.getElementById('useLocalAI').checked = true;
            showToast('Switched to fast local model for better performance');
            break;
        case 'enable_batch_mode':
            // Enable batch processing
            showToast('Batch mode suggestion noted. Upload multiple files for batch processing.');
            break;
        case 'try_multimodal':
            // Switch to multimodal mode
            switchMode('file');
            showToast('Switched to multi-modal mode. Try uploading a PDF or image!');
            break;
        default:
            console.log('AI action:', action);
    }
}

function performTextAction(action) {
    // Handle text-specific actions
    console.log('Text action:', action);
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function showToast(message, type = 'info') {
    // Simple toast notification
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #10b981;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        z-index: 1001;
        animation: slideInFromRight 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}
"""