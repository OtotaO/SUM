#!/usr/bin/env python3
"""
demo_multimodal_complete.py - Complete Multi-Modal Processing Demo for SUM

This demo showcases the full capabilities of SUM's multi-modal processing:
- Image analysis with OCR and visual understanding
- Audio transcription with speaker identification
- Video processing with scene detection
- PDF intelligence with citation tracking
- Code understanding with knowledge graphs
- Cross-modal correlation and insights

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import time
import json
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
import logging

# Import SUM components
from multimodal_engine import MultiModalEngine, ExtendedContentType
from multimodal_integration import MultiModalIntegration, CrossModalInsight
from predictive_intelligence import PredictiveIntelligenceSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiModalDemo:
    """Comprehensive demonstration of multi-modal capabilities."""
    
    def __init__(self):
        """Initialize demo components."""
        self.integration = MultiModalIntegration({
            'multimodal': {
                'whisper_model': 'base',
                'cache_dir': './demo_cache/multimodal'
            },
            'predictive': {
                'model_name': 'all-MiniLM-L6-v2'
            }
        })
        
        # Create demo directories
        self.demo_dir = Path('./demo_multimodal')
        self.demo_dir.mkdir(exist_ok=True)
        
        self.output_dir = self.demo_dir / 'output'
        self.output_dir.mkdir(exist_ok=True)
    
    async def run_complete_demo(self):
        """Run complete multi-modal demonstration."""
        print("\n" + "="*70)
        print("🎭 SUM Multi-Modal Processing Demo - Everything to Insights")
        print("="*70)
        
        # Generate demo content
        demo_files = await self.generate_demo_content()
        
        # Process each type of content
        results = {}
        for content_type, file_path in demo_files.items():
            print(f"\n📄 Processing {content_type}: {file_path.name}")
            print("-" * 50)
            
            # Process file
            request_id = self.integration.process_file(str(file_path), priority=9)
            
            # Track progress
            async for progress in self.integration.process_stream(str(file_path)):
                self._print_progress(progress)
                
                if progress.status == 'completed':
                    results[content_type] = progress.result
                    self._print_result_summary(progress.result)
                elif progress.status == 'error':
                    print(f"❌ Error: {progress.error}")
        
        # Show cross-modal insights
        print("\n" + "="*70)
        print("🔗 Cross-Modal Insights")
        print("="*70)
        
        all_insights = []
        for content_type, file_path in demo_files.items():
            insights = self.integration.get_cross_modal_insights(str(file_path))
            all_insights.extend(insights)
        
        if all_insights:
            for insight in all_insights:
                self._print_insight(insight)
        else:
            print("No cross-modal insights found yet. Process more content to discover connections!")
        
        # Generate comprehensive report
        self._generate_demo_report(results, all_insights)
        
        print("\n✅ Demo complete! Check the output directory for detailed results.")
    
    async def generate_demo_content(self) -> dict:
        """Generate demo content for each supported type."""
        demo_files = {}
        
        # 1. Text document
        text_file = self.demo_dir / "sample_article.txt"
        with open(text_file, 'w') as f:
            f.write("""
The Future of Artificial Intelligence in Healthcare

Artificial Intelligence (AI) is revolutionizing healthcare by enabling faster diagnoses, 
personalized treatments, and improved patient outcomes. Machine learning algorithms can now 
detect diseases earlier than traditional methods, analyze medical images with superhuman 
accuracy, and predict patient risks before symptoms appear.

Key Applications:
1. Medical Imaging: AI systems analyze X-rays, MRIs, and CT scans to detect tumors, 
   fractures, and other abnormalities with 95% accuracy.
2. Drug Discovery: ML models accelerate drug development by predicting molecular behavior 
   and identifying promising compounds.
3. Personalized Medicine: AI analyzes genetic data to customize treatments for individual 
   patients based on their unique biology.

The integration of AI in healthcare promises to reduce costs, improve access to care, 
and save millions of lives worldwide.
            """)
        demo_files['text'] = text_file
        
        # 2. Python code
        code_file = self.demo_dir / "ai_health_analyzer.py"
        with open(code_file, 'w') as f:
            f.write('''
"""AI-powered health data analyzer for patient risk prediction."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Tuple

class HealthRiskAnalyzer:
    """Analyzes patient data to predict health risks using ML."""
    
    def __init__(self, model_path: str = None):
        """Initialize the health risk analyzer."""
        self.model = RandomForestClassifier(n_estimators=100)
        self.risk_categories = ['low', 'medium', 'high']
        self.trained = False
        
    def train(self, patient_data: np.ndarray, risk_labels: np.ndarray):
        """Train the model on historical patient data."""
        self.model.fit(patient_data, risk_labels)
        self.trained = True
        return self.model.score(patient_data, risk_labels)
    
    def predict_risk(self, patient_features: Dict[str, float]) -> Tuple[str, float]:
        """Predict health risk for a patient."""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Convert features to array
        feature_array = np.array([list(patient_features.values())])
        
        # Predict risk
        risk_class = self.model.predict(feature_array)[0]
        risk_probability = self.model.predict_proba(feature_array)[0].max()
        
        return self.risk_categories[risk_class], risk_probability
    
    def analyze_population(self, population_data: List[Dict]) -> Dict[str, float]:
        """Analyze health risks across a population."""
        risk_distribution = {cat: 0 for cat in self.risk_categories}
        
        for patient in population_data:
            risk, _ = self.predict_risk(patient)
            risk_distribution[risk] += 1
        
        # Convert to percentages
        total = len(population_data)
        return {k: (v/total)*100 for k, v in risk_distribution.items()}

# Example usage
if __name__ == "__main__":
    analyzer = HealthRiskAnalyzer()
    print("Health Risk Analyzer initialized")
            ''')
        demo_files['code'] = code_file
        
        # 3. Markdown document
        md_file = self.demo_dir / "ai_healthcare_notes.md"
        with open(md_file, 'w') as f:
            f.write("""
# AI in Healthcare Research Notes

## Meeting: Medical AI Strategy Session
**Date:** 2024-01-15
**Attendees:** Dr. Smith (Chief Medical Officer), AI Team

### Key Decisions:
1. **Pilot Program**: Launch AI diagnostic assistant in radiology department
2. **Timeline**: 6-month implementation phase starting February
3. **Budget**: $2.5M allocated for infrastructure and training

### Technical Requirements:
- GPU cluster for model training
- HIPAA-compliant data storage
- Integration with existing EMR system

### Next Steps:
- [ ] Vendor evaluation (by Jan 30)
- [ ] Staff training plan (by Feb 15)
- [ ] Regulatory compliance review
- [ ] Patient consent protocols

### References:
- Smith et al. (2023). "AI in Clinical Practice: A Systematic Review"
- Johnson & Lee (2024). "Machine Learning for Medical Imaging"
            """)
        demo_files['markdown'] = md_file
        
        # 4. HTML report
        html_file = self.demo_dir / "ai_impact_report.html"
        with open(html_file, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>AI Healthcare Impact Report 2024</title>
</head>
<body>
    <h1>AI Healthcare Impact Report</h1>
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>AI implementation in healthcare has shown remarkable results:</p>
        <ul>
            <li>35% reduction in diagnostic errors</li>
            <li>50% faster image analysis</li>
            <li>$120M in cost savings annually</li>
        </ul>
    </div>
    <div class="data">
        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Department</th><th>AI Adoption</th><th>Efficiency Gain</th></tr>
            <tr><td>Radiology</td><td>85%</td><td>+47%</td></tr>
            <tr><td>Pathology</td><td>72%</td><td>+38%</td></tr>
            <tr><td>Emergency</td><td>45%</td><td>+22%</td></tr>
        </table>
    </div>
</body>
</html>
            """)
        demo_files['html'] = html_file
        
        # 5. Simple image with text (simulate whiteboard/diagram)
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw AI healthcare workflow
            draw.rectangle([50, 50, 750, 550], outline='black', width=2)
            draw.text((400, 30), "AI Healthcare Workflow", anchor="mm", fill='black')
            
            # Components
            components = [
                (200, 150, "Patient Data"),
                (400, 150, "AI Analysis"),
                (600, 150, "Diagnosis"),
                (200, 350, "EMR System"),
                (400, 350, "ML Models"),
                (600, 350, "Physician Review")
            ]
            
            for x, y, text in components:
                draw.rectangle([x-80, y-30, x+80, y+30], outline='blue', width=2)
                draw.text((x, y), text, anchor="mm", fill='black')
            
            # Arrows
            arrow_points = [
                (280, 150, 320, 150),
                (480, 150, 520, 150),
                (200, 180, 200, 320),
                (400, 180, 400, 320),
                (600, 180, 600, 320)
            ]
            
            for x1, y1, x2, y2 in arrow_points:
                draw.line([(x1, y1), (x2, y2)], fill='green', width=2)
            
            img_file = self.demo_dir / "ai_workflow_diagram.png"
            img.save(img_file)
            demo_files['image'] = img_file
            
        except ImportError:
            logger.warning("PIL not available, skipping image generation")
        
        # 6. Generate sample audio (text-to-speech simulation)
        audio_file = self.demo_dir / "ai_healthcare_podcast.txt"
        with open(audio_file, 'w') as f:
            f.write("""
[Podcast Transcript - AI in Healthcare]

Host: Welcome to the Medical AI Podcast. Today we're discussing how artificial 
intelligence is transforming healthcare delivery.

Guest: Thanks for having me. The impact has been tremendous. We're seeing AI 
assist doctors in making faster, more accurate diagnoses, especially in 
radiology and pathology.

Host: Can you give us a specific example?

Guest: Absolutely. In our hospital, we implemented an AI system for analyzing 
chest X-rays. It can detect pneumonia, tuberculosis, and even early-stage 
lung cancer with 95% accuracy. What used to take a radiologist 30 minutes 
now takes 2 minutes with AI assistance.

Host: That's incredible. What about patient privacy concerns?

Guest: Great question. We use federated learning, which means the AI models 
are trained without patient data leaving the hospital. All processing happens 
on-premise with strict HIPAA compliance.

Host: What's next for AI in healthcare?

Guest: I'm excited about personalized medicine. AI can analyze a patient's 
genetic profile, medical history, and lifestyle to create truly personalized 
treatment plans. We're also seeing AI help predict patient deterioration 
hours before traditional monitoring would catch it.

Host: Thank you for sharing these insights. It's clear that AI is not 
replacing doctors but empowering them to provide better care.

[End of transcript]
            """)
        demo_files['audio_transcript'] = audio_file
        
        return demo_files
    
    def _print_progress(self, progress):
        """Print progress update."""
        bar_length = 40
        filled = int(bar_length * progress.progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r[{bar}] {progress.progress*100:.0f}% - {progress.current_step}", end='', flush=True)
        
        if progress.status in ['completed', 'error']:
            print()  # New line after completion
    
    def _print_result_summary(self, result):
        """Print summary of processing result."""
        print(f"\n✓ Content Type: {result.content_type.value}")
        print(f"✓ Confidence Score: {result.confidence_score:.2%}")
        print(f"✓ Processing Time: {result.processing_time:.2f}s")
        
        if result.extracted_text:
            preview = result.extracted_text[:200].replace('\n', ' ')
            print(f"✓ Text Preview: {preview}...")
        
        if result.concepts:
            print(f"✓ Concepts: {', '.join(result.concepts[:5])}")
        
        if result.metadata.get('predictions'):
            predictions = result.metadata['predictions']
            if predictions.get('suggested_connections'):
                print(f"✓ Suggested Connections: {len(predictions['suggested_connections'])}")
    
    def _print_insight(self, insight: CrossModalInsight):
        """Print cross-modal insight."""
        print(f"\n🔗 {insight.insight_type.replace('_', ' ').title()}")
        print(f"   Sources: {', '.join(Path(p).name for p in insight.content_sources)}")
        print(f"   {insight.description}")
        print(f"   Confidence: {insight.confidence:.2%}")
    
    def _generate_demo_report(self, results: dict, insights: list):
        """Generate comprehensive demo report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'processing_results': {},
            'cross_modal_insights': [],
            'statistics': {
                'total_files': len(results),
                'total_insights': len(insights),
                'content_types': list(results.keys()),
                'average_confidence': sum(r.confidence_score for r in results.values()) / len(results) if results else 0
            }
        }
        
        # Add results
        for content_type, result in results.items():
            report['processing_results'][content_type] = {
                'confidence': result.confidence_score,
                'processing_time': result.processing_time,
                'text_length': len(result.extracted_text) if result.extracted_text else 0,
                'concepts': result.concepts[:10] if result.concepts else [],
                'has_predictions': 'predictions' in result.metadata
            }
        
        # Add insights
        for insight in insights:
            report['cross_modal_insights'].append({
                'type': insight.insight_type,
                'description': insight.description,
                'confidence': insight.confidence,
                'sources': [Path(p).name for p in insight.content_sources]
            })
        
        # Save report
        report_file = self.output_dir / 'multimodal_demo_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Detailed report saved to: {report_file}")


async def main():
    """Run the multi-modal demo."""
    demo = MultiModalDemo()
    
    try:
        await demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())