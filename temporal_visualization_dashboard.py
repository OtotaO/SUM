#!/usr/bin/env python3
"""
temporal_visualization_dashboard.py - Beautiful Temporal Intelligence Visualizations

This module creates stunning visualizations of temporal intelligence insights,
making the evolution of thinking visible and beautiful.

Features:
- Concept evolution timelines
- Seasonal pattern heatmaps
- Momentum trajectory graphs
- Knowledge aging curves
- Future projection charts
- Cognitive rhythm visualizations
- Interactive temporal dashboard

Author: ototao
License: Apache License 2.0
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle, Rectangle
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline as pyo
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter

# Import temporal intelligence components
from temporal_intelligence_engine import TemporalIntelligenceEngine, TemporalSUMIntegration

# Configure beautiful plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TemporalVisualization')


class TemporalVisualizationEngine:
    """Creates beautiful visualizations of temporal intelligence data."""
    
    def __init__(self, temporal_engine: TemporalIntelligenceEngine, output_dir: str = "temporal_visualizations"):
        self.temporal_engine = temporal_engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color schemes for beautiful visualizations
        self.concept_colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.momentum_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        self.season_colors = {
            'spring': '#98FB98',
            'summer': '#FFE4B5', 
            'autumn': '#DEB887',
            'winter': '#B0E0E6'
        }
        
        logger.info(f"Temporal Visualization Engine initialized - Output: {output_dir}")
    
    def create_concept_evolution_timeline(self, concept: str = None, save_path: str = None) -> str:
        """Create a beautiful timeline showing concept evolution."""
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"concept_evolution_{timestamp}.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🧠 Concept Evolution Over Time', fontsize=20, y=0.95)
        
        # Get concept evolution data
        if concept and concept in self.temporal_engine.concept_evolutions:
            evolutions = {concept: self.temporal_engine.concept_evolutions[concept]}
        else:
            # Show top 4 most evolved concepts
            evolutions = dict(list(self.temporal_engine.concept_evolutions.items())[:4])
        
        if not evolutions:
            # Create placeholder
            axes[0, 0].text(0.5, 0.5, 'Begin thinking to see\nconcept evolution', 
                           ha='center', va='center', fontsize=16, alpha=0.7)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        
        # Plot each concept evolution
        ax_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for idx, (concept_name, evolution) in enumerate(evolutions.items()):
            if idx >= 4:
                break
                
            ax = axes[ax_positions[idx][0], ax_positions[idx][1]]
            
            if not evolution.depth_progression:
                ax.text(0.5, 0.5, f'No evolution data\nfor {concept_name}', 
                       ha='center', va='center', alpha=0.7)
                continue
            
            # Create time series
            progression_length = len(evolution.depth_progression)
            time_points = np.arange(progression_length)
            
            # Plot depth progression
            ax.plot(time_points, evolution.depth_progression, 
                   'o-', linewidth=3, markersize=8, color=self.concept_colors[idx % len(self.concept_colors)],
                   label='Understanding Depth')
            
            # Plot complexity trajectory if available
            if evolution.complexity_trajectory and len(evolution.complexity_trajectory) == progression_length:
                ax2 = ax.twinx()
                ax2.plot(time_points, evolution.complexity_trajectory, 
                        's--', linewidth=2, markersize=6, color='orange', alpha=0.7,
                        label='Complexity')
                ax2.set_ylabel('Complexity', fontsize=10, color='orange')
                ax2.tick_params(axis='y', labelsize=8, colors='orange')
            
            # Mark breakthrough moments
            for breakthrough_idx, breakthrough_time in enumerate(evolution.breakthrough_moments):
                if breakthrough_idx < progression_length:
                    ax.axvline(x=breakthrough_idx, color='red', linestyle=':', alpha=0.8, linewidth=2)
                    ax.annotate('💡', xy=(breakthrough_idx, evolution.depth_progression[breakthrough_idx]), 
                              xytext=(breakthrough_idx, evolution.depth_progression[breakthrough_idx] + 0.1),
                              fontsize=16, ha='center')
            
            # Formatting
            ax.set_title(f'{concept_name.replace("-", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Progression', fontsize=10)
            ax.set_ylabel('Understanding Depth', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add projection line
            if evolution.projected_next_depth > 0:
                future_x = progression_length
                ax.plot([progression_length-1, future_x], 
                       [evolution.depth_progression[-1], evolution.projected_next_depth],
                       ':', color='gray', alpha=0.8, linewidth=2)
                ax.plot(future_x, evolution.projected_next_depth, 'o', 
                       color='gray', markersize=8, alpha=0.8)
                ax.annotate('Projected', xy=(future_x, evolution.projected_next_depth),
                          xytext=(future_x + 0.5, evolution.projected_next_depth),
                          fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Concept evolution timeline saved: {save_path}")
        return str(save_path)
    
    def create_seasonal_pattern_heatmap(self, save_path: str = None) -> str:
        """Create a beautiful heatmap of seasonal thinking patterns."""
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"seasonal_patterns_{timestamp}.png"
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('⏰ Seasonal Patterns in Your Thinking', fontsize=20, y=0.95)
        
        # Get seasonal pattern data
        patterns = self.temporal_engine.seasonal_patterns
        
        if not patterns:
            axes[1].text(0.5, 0.5, 'Capture thoughts over time\nto reveal seasonal patterns', 
                        ha='center', va='center', fontsize=16, alpha=0.7)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return str(save_path)
        
        # Organize patterns by type
        hourly_patterns = {k: v for k, v in patterns.items() if v.pattern_type == 'hourly'}
        daily_patterns = {k: v for k, v in patterns.items() if v.pattern_type == 'daily'}
        monthly_patterns = {k: v for k, v in patterns.items() if v.pattern_type == 'monthly'}
        
        # Create hourly heatmap
        if hourly_patterns:
            hours = list(range(24))
            concept_hours = defaultdict(lambda: [0] * 24)
            
            for pattern in hourly_patterns.values():
                if pattern.peak_periods:
                    hour = int(pattern.peak_periods[0].split(':')[0])
                    for concept in pattern.associated_concepts[:3]:  # Top 3 concepts
                        concept_hours[concept][hour] = pattern.pattern_strength
            
            if concept_hours:
                # Create heatmap data
                concepts = list(concept_hours.keys())[:10]  # Top 10 concepts
                heatmap_data = np.array([concept_hours[concept] for concept in concepts])
                
                im1 = axes[0].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                axes[0].set_title('Hourly Thinking Patterns', fontsize=16, fontweight='bold')
                axes[0].set_xlabel('Hour of Day')
                axes[0].set_ylabel('Concepts')
                axes[0].set_xticks(range(0, 24, 2))
                axes[0].set_xticklabels([f'{h}:00' for h in range(0, 24, 2)])
                axes[0].set_yticks(range(len(concepts)))
                axes[0].set_yticklabels([c.replace('-', ' ').title() for c in concepts])
                
                # Add colorbar
                cbar1 = plt.colorbar(im1, ax=axes[0])
                cbar1.set_label('Pattern Strength', rotation=270, labelpad=20)
        
        # Create daily heatmap
        if daily_patterns:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            concept_days = defaultdict(lambda: [0] * 7)
            
            for pattern in daily_patterns.values():
                if pattern.peak_periods:
                    day_name = pattern.peak_periods[0]
                    day_idx = days.index(day_name) if day_name in days else -1
                    if day_idx >= 0:
                        for concept in pattern.associated_concepts[:3]:
                            concept_days[concept][day_idx] = pattern.pattern_strength
            
            if concept_days:
                concepts = list(concept_days.keys())[:8]  # Top 8 concepts
                heatmap_data = np.array([concept_days[concept] for concept in concepts])
                
                im2 = axes[1].imshow(heatmap_data, cmap='Blues', aspect='auto')
                axes[1].set_title('Daily Thinking Patterns', fontsize=16, fontweight='bold')
                axes[1].set_xlabel('Day of Week')
                axes[1].set_ylabel('Concepts')
                axes[1].set_xticks(range(7))
                axes[1].set_xticklabels(days)
                axes[1].set_yticks(range(len(concepts)))
                axes[1].set_yticklabels([c.replace('-', ' ').title() for c in concepts])
                
                cbar2 = plt.colorbar(im2, ax=axes[1])
                cbar2.set_label('Pattern Strength', rotation=270, labelpad=20)
        
        # Create monthly heatmap
        if monthly_patterns:
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            concept_months = defaultdict(lambda: [0] * 12)
            
            for pattern in monthly_patterns.values():
                if pattern.peak_periods:
                    month_name = pattern.peak_periods[0]
                    month_idx = months.index(month_name) if month_name in months else -1
                    if month_idx >= 0:
                        for concept in pattern.associated_concepts[:3]:
                            concept_months[concept][month_idx] = pattern.pattern_strength
            
            if concept_months:
                concepts = list(concept_months.keys())[:6]  # Top 6 concepts
                heatmap_data = np.array([concept_months[concept] for concept in concepts])
                
                im3 = axes[2].imshow(heatmap_data, cmap='Greens', aspect='auto')
                axes[2].set_title('Monthly Thinking Patterns', fontsize=16, fontweight='bold')
                axes[2].set_xlabel('Month')
                axes[2].set_ylabel('Concepts')
                axes[2].set_xticks(range(12))
                axes[2].set_xticklabels(months)
                axes[2].set_yticks(range(len(concepts)))
                axes[2].set_yticklabels([c.replace('-', ' ').title() for c in concepts])
                
                cbar3 = plt.colorbar(im3, ax=axes[2])
                cbar3.set_label('Pattern Strength', rotation=270, labelpad=20)
        
        # Handle empty axes
        for i, ax in enumerate(axes):
            if not ax.get_images():  # No heatmap was created
                pattern_types = ['hourly', 'daily', 'monthly']
                ax.text(0.5, 0.5, f'No {pattern_types[i]} patterns detected yet\nContinue capturing thoughts!', 
                       ha='center', va='center', fontsize=14, alpha=0.7)
                ax.set_title(f'{pattern_types[i].title()} Thinking Patterns', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Seasonal pattern heatmap saved: {save_path}")
        return str(save_path)
    
    def create_momentum_trajectory_graph(self, save_path: str = None) -> str:
        """Create a beautiful graph showing intellectual momentum trajectories."""
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"momentum_trajectories_{timestamp}.png"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚀 Intellectual Momentum Trajectories', fontsize=20, y=0.95)
        
        momentum_data = self.temporal_engine.momentum_trackers
        
        if not momentum_data:
            ax2.text(0.5, 0.5, 'Build momentum by exploring\ntopics consistently over time', 
                    ha='center', va='center', fontsize=16, alpha=0.7)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return str(save_path)
        
        # Sort by critical mass
        sorted_momentum = sorted(momentum_data.items(), key=lambda x: x[1].current_critical_mass, reverse=True)
        top_momentum = sorted_momentum[:5]  # Top 5
        
        # 1. Critical Mass Progress
        areas = [momentum.research_area.replace('-', ' ').title() for _, momentum in top_momentum]
        critical_masses = [momentum.current_critical_mass for _, momentum in top_momentum]
        velocities = [momentum.velocity for _, momentum in top_momentum]
        
        bars1 = ax1.barh(areas, critical_masses, color=self.momentum_colors[:len(areas)])
        ax1.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Breakthrough Threshold')
        ax1.set_xlabel('Critical Mass')
        ax1.set_title('Critical Mass by Research Area', fontweight='bold')
        ax1.legend()
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars1, critical_masses)):
            ax1.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.2f}', 
                    va='center', fontsize=10)
        
        # 2. Velocity vs Mass Scatter
        masses = [momentum.mass for _, momentum in top_momentum]
        colors_indexed = [self.momentum_colors[i % len(self.momentum_colors)] for i in range(len(top_momentum))]
        
        scatter = ax2.scatter(velocities, masses, s=[cm*500 for cm in critical_masses], 
                            c=colors_indexed, alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add labels for each point
        for i, (vel, mass, area) in enumerate(zip(velocities, masses, areas)):
            ax2.annotate(area[:15] + '...' if len(area) > 15 else area, 
                        (vel, mass), xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.8)
        
        ax2.set_xlabel('Velocity (thoughts/day)')
        ax2.set_ylabel('Mass (engagement depth)')
        ax2.set_title('Momentum Phase Space', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Flow State Sessions
        flow_sessions = []
        flow_areas = []
        
        for area_id, momentum in top_momentum:
            if momentum.flow_sessions:
                flow_sessions.extend([len(momentum.flow_sessions)])
                flow_areas.extend([momentum.research_area.replace('-', ' ').title()])
        
        if flow_sessions:
            bars3 = ax3.bar(range(len(flow_areas)), flow_sessions, 
                          color=self.momentum_colors[:len(flow_areas)])
            ax3.set_xlabel('Research Areas')
            ax3.set_ylabel('Flow Sessions Count')
            ax3.set_title('Flow State Sessions by Area', fontweight='bold')
            ax3.set_xticks(range(len(flow_areas)))
            ax3.set_xticklabels([area[:10] + '...' if len(area) > 10 else area for area in flow_areas], 
                              rotation=45)
            
            # Add value labels
            for bar, value in zip(bars3, flow_sessions):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(value), 
                        ha='center', va='bottom', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'No flow sessions\ndetected yet', ha='center', va='center', 
                    fontsize=14, alpha=0.7)
            ax3.set_title('Flow State Sessions by Area', fontweight='bold')
        
        # 4. Breakthrough Predictions
        breakthrough_areas = []
        breakthrough_probabilities = []
        breakthrough_dates = []
        
        for area_id, momentum in top_momentum:
            if momentum.estimated_breakthrough_date:
                breakthrough_areas.append(momentum.research_area.replace('-', ' ').title())
                breakthrough_probabilities.append(momentum.current_critical_mass)
                
                # Calculate days until breakthrough
                days_until = (momentum.estimated_breakthrough_date - datetime.now()).days
                breakthrough_dates.append(max(0, days_until))
        
        if breakthrough_areas:
            # Create a bubble chart
            colors_bt = [self.momentum_colors[i % len(self.momentum_colors)] for i in range(len(breakthrough_areas))]
            
            scatter_bt = ax4.scatter(breakthrough_dates, breakthrough_probabilities, 
                                   s=[p*1000 for p in breakthrough_probabilities], 
                                   c=colors_bt, alpha=0.7, edgecolors='black', linewidth=2)
            
            # Add labels
            for i, (days, prob, area) in enumerate(zip(breakthrough_dates, breakthrough_probabilities, breakthrough_areas)):
                ax4.annotate(area[:12] + '...' if len(area) > 12 else area, 
                           (days, prob), xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, alpha=0.8)
            
            ax4.set_xlabel('Days Until Predicted Breakthrough')
            ax4.set_ylabel('Breakthrough Probability')
            ax4.set_title('Breakthrough Predictions', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='High Probability')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No breakthrough\npredictions yet', ha='center', va='center', 
                    fontsize=14, alpha=0.7)
            ax4.set_title('Breakthrough Predictions', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Momentum trajectory graph saved: {save_path}")
        return str(save_path)
    
    def create_knowledge_aging_curves(self, save_path: str = None) -> str:
        """Create beautiful curves showing knowledge aging and resurrection."""
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"knowledge_aging_{timestamp}.png"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('⏳ Knowledge Aging & Resurrection Patterns', fontsize=20, y=0.95)
        
        aging_data = self.temporal_engine.aging_knowledge
        
        if not aging_data:
            ax2.text(0.5, 0.5, 'Knowledge aging patterns\nwill emerge over time', 
                    ha='center', va='center', fontsize=16, alpha=0.7)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return str(save_path)
        
        current_time = datetime.now()
        
        # 1. Relevance Score Distribution
        relevance_scores = [aging.current_relevance_score for aging in aging_data.values()]
        
        ax1.hist(relevance_scores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.axvline(x=np.mean(relevance_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(relevance_scores):.2f}')
        ax1.set_xlabel('Current Relevance Score')
        ax1.set_ylabel('Number of Knowledge Items')
        ax1.set_title('Knowledge Relevance Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Age vs Relevance Scatter
        ages = [(current_time - aging.original_capture_date).days for aging in aging_data.values()]
        
        scatter = ax2.scatter(ages, relevance_scores, alpha=0.6, c=ages, cmap='viridis', s=50)
        ax2.set_xlabel('Age (Days)')
        ax2.set_ylabel('Current Relevance Score')
        ax2.set_title('Age vs Relevance Relationship', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add exponential decay curve
        if ages and relevance_scores:
            max_age = max(ages)
            x_curve = np.linspace(0, max_age, 100)
            # Typical forgetting curve: R(t) = e^(-t/tau)
            tau = max_age / 3  # Decay constant
            y_curve = np.exp(-x_curve / tau)
            ax2.plot(x_curve, y_curve, '--', color='red', alpha=0.8, linewidth=2, 
                    label='Theoretical Forgetting Curve')
            ax2.legend()
        
        plt.colorbar(scatter, ax=ax2, label='Age (Days)')
        
        # 3. Resurrection Events Timeline
        resurrection_events = []
        resurrection_contexts = []
        
        for aging in aging_data.values():
            for event_time in aging.resurrection_events:
                resurrection_events.append(event_time)
                resurrection_contexts.append(aging.knowledge_id)
        
        if resurrection_events:
            # Group by month
            resurrection_months = defaultdict(int)
            for event in resurrection_events:
                month_key = event.strftime('%Y-%m')
                resurrection_months[month_key] += 1
            
            months = sorted(resurrection_months.keys())
            counts = [resurrection_months[month] for month in months]
            
            ax3.plot(range(len(months)), counts, 'o-', linewidth=3, markersize=8, 
                    color='green', alpha=0.8)
            ax3.set_xlabel('Time Period')
            ax3.set_ylabel('Resurrection Events')
            ax3.set_title('Knowledge Resurrection Timeline', fontweight='bold')
            ax3.set_xticks(range(len(months)))
            ax3.set_xticklabels([m[-2:] for m in months], rotation=45)  # Show just month
            ax3.grid(True, alpha=0.3)
            
            # Add trend line
            if len(counts) > 1:
                z = np.polyfit(range(len(counts)), counts, 1)
                p = np.poly1d(z)
                ax3.plot(range(len(counts)), p(range(len(counts))), "--", 
                       color='red', alpha=0.8, linewidth=2, label='Trend')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No knowledge\nresurrections yet', ha='center', va='center', 
                    fontsize=14, alpha=0.7)
            ax3.set_title('Knowledge Resurrection Timeline', fontweight='bold')
        
        # 4. Review Schedule Optimization
        review_needed = []
        review_dates = []
        
        for aging in aging_data.values():
            if aging.next_optimal_review:
                days_until_review = (aging.next_optimal_review - current_time).days
                if days_until_review >= 0:  # Future reviews only
                    review_needed.append(aging.current_relevance_score)
                    review_dates.append(days_until_review)
        
        if review_needed and review_dates:
            # Create review schedule heatmap
            max_days = min(max(review_dates), 30) if review_dates else 30  # Limit to 30 days
            
            # Group by day
            daily_reviews = defaultdict(list)
            for days, relevance in zip(review_dates, review_needed):
                if days <= max_days:
                    daily_reviews[days].append(relevance)
            
            days = sorted(daily_reviews.keys())
            avg_relevance = [np.mean(daily_reviews[day]) for day in days]
            review_counts = [len(daily_reviews[day]) for day in days]
            
            # Create double y-axis plot
            ax4_twin = ax4.twinx()
            
            bars = ax4.bar(days, review_counts, alpha=0.6, color='lightblue', 
                          label='Review Count')
            line = ax4_twin.plot(days, avg_relevance, 'ro-', linewidth=2, 
                               label='Avg Relevance')
            
            ax4.set_xlabel('Days Until Review')
            ax4.set_ylabel('Number of Reviews Needed', color='blue')
            ax4_twin.set_ylabel('Average Relevance Score', color='red')
            ax4.set_title('Optimal Review Schedule', fontweight='bold')
            
            # Combine legends
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines + lines2, labels + labels2, loc='upper right')
            
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No reviews\nscheduled yet', ha='center', va='center', 
                    fontsize=14, alpha=0.7)
            ax4.set_title('Optimal Review Schedule', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Knowledge aging curves saved: {save_path}")
        return str(save_path)
    
    def create_future_projections_chart(self, save_path: str = None) -> str:
        """Create beautiful charts showing future interest projections."""
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"future_projections_{timestamp}.png"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔮 Future Interest Projections', fontsize=20, y=0.95)
        
        projections = self.temporal_engine.future_projections
        
        if not projections:
            ax2.text(0.5, 0.5, 'Future projections will appear\nas thinking patterns develop', 
                    ha='center', va='center', fontsize=16, alpha=0.7)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return str(save_path)
        
        # Get latest projection
        latest_projection = max(projections, key=lambda p: p.generated_date)
        
        # 1. Emerging Interests
        if latest_projection.emerging_interests:
            interests = [interest.replace('-', ' ').title() for interest, _ in latest_projection.emerging_interests[:8]]
            probabilities = [prob for _, prob in latest_projection.emerging_interests[:8]]
            
            bars1 = ax1.barh(interests, probabilities, color='lightgreen', alpha=0.8)
            ax1.set_xlabel('Emergence Probability')
            ax1.set_title('Predicted Emerging Interests', fontweight='bold')
            ax1.set_xlim(0, 1)
            
            # Add probability labels
            for bar, prob in zip(bars1, probabilities):
                ax1.text(prob + 0.01, bar.get_y() + bar.get_height()/2, f'{prob:.2f}', 
                        va='center', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'No emerging\ninterests predicted', ha='center', va='center', 
                    fontsize=14, alpha=0.7)
            ax1.set_title('Predicted Emerging Interests', fontweight='bold')
        
        # 2. Declining Interests
        if latest_projection.declining_interests:
            dec_interests = [interest.replace('-', ' ').title() for interest, _ in latest_projection.declining_interests[:6]]
            dec_probabilities = [prob for _, prob in latest_projection.declining_interests[:6]]
            
            bars2 = ax2.barh(dec_interests, dec_probabilities, color='lightcoral', alpha=0.8)
            ax2.set_xlabel('Decline Probability')
            ax2.set_title('Predicted Declining Interests', fontweight='bold')
            ax2.set_xlim(0, 1)
            
            # Add probability labels
            for bar, prob in zip(bars2, dec_probabilities):
                ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, f'{prob:.2f}', 
                        va='center', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No declining\ninterests predicted', ha='center', va='center', 
                    fontsize=14, alpha=0.7)
            ax2.set_title('Predicted Declining Interests', fontweight='bold')
        
        # 3. Prediction Confidence Over Time
        if len(projections) > 1:
            projection_dates = [p.generated_date for p in projections]
            confidences = [p.prediction_confidence for p in projections]
            
            ax3.plot(projection_dates, confidences, 'o-', linewidth=3, markersize=8, 
                    color='purple', alpha=0.8)
            ax3.set_xlabel('Projection Date')
            ax3.set_ylabel('Prediction Confidence')
            ax3.set_title('Prediction Confidence Evolution', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Format dates
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax3.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(projections)//5)))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            
            # Add trend line
            if len(confidences) > 1:
                z = np.polyfit(mdates.date2num(projection_dates), confidences, 1)
                p = np.poly1d(z)
                trend_line = p(mdates.date2num(projection_dates))
                ax3.plot(projection_dates, trend_line, '--', color='red', alpha=0.8, 
                        linewidth=2, label='Trend')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'More projections needed\nfor confidence tracking', 
                    ha='center', va='center', fontsize=14, alpha=0.7)
            ax3.set_title('Prediction Confidence Evolution', fontweight='bold')
        
        # 4. Interest Trajectory Comparison
        if latest_projection.emerging_interests and latest_projection.declining_interests:
            # Create a comparison chart
            all_interests = []
            all_changes = []
            all_colors = []
            
            # Add emerging (positive change)
            for interest, prob in latest_projection.emerging_interests[:5]:
                all_interests.append(interest.replace('-', ' ').title())
                all_changes.append(prob)
                all_colors.append('green')
            
            # Add declining (negative change) 
            for interest, prob in latest_projection.declining_interests[:5]:
                all_interests.append(interest.replace('-', ' ').title())
                all_changes.append(-prob)  # Make negative
                all_colors.append('red')
            
            bars4 = ax4.barh(all_interests, all_changes, color=all_colors, alpha=0.7)
            ax4.set_xlabel('Interest Change Trajectory')
            ax4.set_title('Interest Evolution Forecast', fontweight='bold')
            ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, change in zip(bars4, all_changes):
                label_x = change + (0.02 if change >= 0 else -0.02)
                ax4.text(label_x, bar.get_y() + bar.get_height()/2, f'{abs(change):.2f}', 
                        va='center', ha='left' if change >= 0 else 'right', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'Interest trajectory\ncomparison coming soon', 
                    ha='center', va='center', fontsize=14, alpha=0.7)
            ax4.set_title('Interest Evolution Forecast', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Future projections chart saved: {save_path}")
        return str(save_path)
    
    def create_cognitive_rhythm_analysis(self, save_path: str = None) -> str:
        """Create beautiful visualizations of cognitive rhythms."""
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"cognitive_rhythms_{timestamp}.png"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🧠 Cognitive Rhythm Analysis', fontsize=20, y=0.95)
        
        # Get rhythm data
        rhythm_data = self.temporal_engine.rhythm_analyzer.get_optimal_thinking_times()
        
        # 1. Hourly Performance Pattern
        hourly_patterns = rhythm_data.get('hourly_patterns', {})
        
        if hourly_patterns:
            hours = sorted(hourly_patterns.keys())
            performance = [hourly_patterns[hour] for hour in hours]
            
            # Create polar plot for 24-hour cycle
            theta = np.array(hours) * 2 * np.pi / 24
            
            ax1 = plt.subplot(221, projection='polar')
            ax1.plot(theta, performance, 'o-', linewidth=3, markersize=8, color='blue', alpha=0.8)
            ax1.fill(theta, performance, alpha=0.3, color='blue')
            ax1.set_title('24-Hour Cognitive Rhythm', pad=20, fontweight='bold')
            ax1.set_theta_zero_location('N')  # 12 o'clock at top
            ax1.set_theta_direction(-1)  # Clockwise
            ax1.set_thetagrids(np.arange(0, 360, 30), 
                             ['12AM', '2AM', '4AM', '6AM', '8AM', '10AM',
                              '12PM', '2PM', '4PM', '6PM', '8PM', '10PM'])
            
            # Mark peak hour
            peak_hour = rhythm_data.get('peak_performance_hour', 10)
            peak_theta = peak_hour * 2 * np.pi / 24
            peak_performance = hourly_patterns.get(peak_hour, 0)
            ax1.plot(peak_theta, peak_performance, 'ro', markersize=15, alpha=0.8, 
                    label=f'Peak: {peak_hour}:00')
            ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        else:
            ax1.text(0.5, 0.5, 'Hourly rhythm patterns\nwill emerge with more data', 
                    ha='center', va='center', fontsize=14, alpha=0.7)
            ax1.set_title('24-Hour Cognitive Rhythm', fontweight='bold')
        
        # Reset to regular subplot for remaining plots
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)
        
        # 2. Weekly Performance Pattern
        daily_patterns = rhythm_data.get('daily_patterns', {})
        
        if daily_patterns:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_performance = [daily_patterns.get(i, 0) for i in range(7)]
            
            bars2 = ax2.bar(days, day_performance, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                                                          '#FFEAA7', '#DDA0DD', '#98D8C8'])
            ax2.set_ylabel('Performance Score')
            ax2.set_title('Weekly Cognitive Pattern', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Mark peak day
            peak_day = rhythm_data.get('peak_performance_day', 1)
            if peak_day < len(day_performance):
                bars2[peak_day].set_color('gold')
                bars2[peak_day].set_edgecolor('black')
                bars2[peak_day].set_linewidth(2)
            
            # Add value labels
            for bar, value in zip(bars2, day_performance):
                if value > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Weekly rhythm patterns\nwill develop over time', 
                    ha='center', va='center', fontsize=14, alpha=0.7)
            ax2.set_title('Weekly Cognitive Pattern', fontweight='bold')
        
        # 3. Flow State Distribution
        flow_hours = rhythm_data.get('flow_state_hours', {})
        
        if flow_hours:
            hours = sorted(flow_hours.keys())
            flow_counts = [flow_hours[hour] for hour in hours]
            
            ax3.bar(hours, flow_counts, color='lightgreen', alpha=0.8, edgecolor='darkgreen')
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Flow Sessions')
            ax3.set_title('Flow State Distribution', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for hour, count in zip(hours, flow_counts):
                if count > 0:
                    ax3.text(hour, count + 0.05, str(count), ha='center', va='bottom', fontsize=10)
            
            # Mark peak flow hour
            if flow_counts:
                max_flow_hour = hours[flow_counts.index(max(flow_counts))]
                max_flow_count = max(flow_counts)
                ax3.bar(max_flow_hour, max_flow_count, color='gold', edgecolor='black', linewidth=2)
        else:
            ax3.text(0.5, 0.5, 'Flow state sessions\nnot detected yet', 
                    ha='center', va='center', fontsize=14, alpha=0.7)
            ax3.set_title('Flow State Distribution', fontweight='bold')
        
        # 4. Monthly Performance Trends
        monthly_patterns = rhythm_data.get('monthly_patterns', {})
        
        if monthly_patterns:
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_performance = [monthly_patterns.get(i+1, 0) for i in range(12)]
            
            ax4.plot(months, month_performance, 'o-', linewidth=3, markersize=8, 
                    color='purple', alpha=0.8)
            ax4.fill_between(months, month_performance, alpha=0.3, color='purple')
            ax4.set_ylabel('Performance Score')
            ax4.set_title('Monthly Cognitive Trends', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            # Mark peak month
            peak_month = rhythm_data.get('peak_performance_month', 3)
            if 1 <= peak_month <= 12 and month_performance[peak_month-1] > 0:
                ax4.plot(months[peak_month-1], month_performance[peak_month-1], 
                        'ro', markersize=15, alpha=0.8, label=f'Peak: {months[peak_month-1]}')
                ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Monthly trends\nwill appear over time', 
                    ha='center', va='center', fontsize=14, alpha=0.7)
            ax4.set_title('Monthly Cognitive Trends', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Cognitive rhythm analysis saved: {save_path}")
        return str(save_path)
    
    def create_comprehensive_dashboard(self, save_path: str = None) -> str:
        """Create a comprehensive temporal intelligence dashboard."""
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"temporal_dashboard_{timestamp}.png"
        
        # Create all individual visualizations
        evolution_path = self.create_concept_evolution_timeline()
        seasonal_path = self.create_seasonal_pattern_heatmap()
        momentum_path = self.create_momentum_trajectory_graph()
        aging_path = self.create_knowledge_aging_curves()
        projections_path = self.create_future_projections_chart()
        rhythm_path = self.create_cognitive_rhythm_analysis()
        
        # Create a summary dashboard
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 0.5], hspace=0.3)
        
        fig.suptitle('🧠⏰ TEMPORAL INTELLIGENCE DASHBOARD\nRevolutionary Time-Aware Understanding', 
                    fontsize=24, y=0.98, fontweight='bold')
        
        # Get temporal insights
        insights = self.temporal_engine.get_temporal_insights()
        
        # Summary statistics
        ax_summary = fig.add_subplot(gs[5, :])
        ax_summary.axis('off')
        
        # Create beautiful summary text
        summary_text = self._create_dashboard_summary(insights)
        ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', 
                       fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Individual visualization previews would go here
        # For now, save the comprehensive view
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Comprehensive temporal dashboard created: {save_path}")
        
        return {
            'dashboard': str(save_path),
            'individual_visualizations': {
                'concept_evolution': evolution_path,
                'seasonal_patterns': seasonal_path,
                'momentum_trajectories': momentum_path,
                'knowledge_aging': aging_path,
                'future_projections': projections_path,
                'cognitive_rhythms': rhythm_path
            }
        }
    
    def _create_dashboard_summary(self, insights: Dict[str, Any]) -> str:
        """Create a beautiful summary for the dashboard."""
        summary_parts = []
        
        # Temporal narrative
        narrative = insights.get('beautiful_narrative', '')
        if narrative:
            summary_parts.append(f"🎭 TEMPORAL NARRATIVE\n{narrative}\n")
        
        # Key statistics
        concept_summary = insights.get('concept_evolution_summary', {})
        momentum_summary = insights.get('intellectual_momentum', {})
        rhythm_summary = insights.get('cognitive_rhythms', {})
        
        stats = []
        if concept_summary.get('total_concepts_tracked'):
            stats.append(f"📈 {concept_summary['total_concepts_tracked']} Concepts Evolving")
        
        if momentum_summary.get('active_research_areas'):
            stats.append(f"🚀 {momentum_summary['active_research_areas']} Research Areas Active")
        
        if rhythm_summary.get('peak_performance_hour'):
            hour = rhythm_summary['peak_performance_hour']
            stats.append(f"⏰ Peak Performance: {hour}:00")
        
        if stats:
            summary_parts.append(f"📊 KEY METRICS: {' • '.join(stats)}")
        
        return '\n\n'.join(summary_parts) if summary_parts else "Your temporal intelligence journey is beginning..."


def create_temporal_dashboard_demo():
    """Create a demonstration of the temporal visualization system."""
    print("🧠⏰ Temporal Visualization Dashboard Demo")
    print("=" * 60)
    
    try:
        # This would integrate with the actual temporal engine
        # For demo purposes, we'll create visualizations with sample data
        
        output_dir = Path("temporal_visualizations_demo")
        output_dir.mkdir(exist_ok=True)
        
        # Create sample visualizations
        viz_engine = TemporalVisualizationEngine(None, str(output_dir))
        
        print("Creating temporal intelligence visualizations...")
        
        # Note: In a real implementation, this would use actual temporal engine data
        # For now, we create the visualization framework
        
        print(f"✨ Temporal visualization system ready!")
        print(f"📁 Output directory: {output_dir}")
        print("\nTo see visualizations, integrate with actual TemporalIntelligenceEngine data")
        
        return str(output_dir)
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        return None


if __name__ == "__main__":
    create_temporal_dashboard_demo()