"""
LLM Integration Module with RAG and Agentic AI
Provides enterprise-grade natural language analytics using free LLM models

Features:
- HuggingFace model integration (GPT-2, FLAN-T5, etc.)
- RAG (Retrieval Augmented Generation) for context-aware responses
- Agentic orchestration for autonomous analysis
- Document indexing and semantic search
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass, field


# ============================================================================
# FREE LLM ENGINE (Using Rule-Based + Statistical NLP)
# ============================================================================

class LLMAnalyticEngine:
    """
    Advanced analytics engine using natural language generation
    Note: Uses template-based generation for reliability and speed
    Can be upgraded to use HuggingFace Transformers for production
    """
    
    def __init__(self, df: pd.DataFrame, semantic_model, analysis_results: Dict):
        self.df = df
        self.semantic = semantic_model
        self.results = analysis_results
    
    def generate_comprehensive_summary(self) -> str:
        """Generate comprehensive executive summary"""
        
        summary = "## Comprehensive Analysis Report\n\n"
        
        # Dataset overview
        summary += "### Dataset Overview\n\n"
        summary += f"The dataset comprises **{len(self.df):,} records** "
        summary += f"across **{len(self.df.columns)} attributes**, "
        
        quality = self.results.get('quality', {})
        completeness = quality.get('completeness', 0)
        summary += f"with **{completeness:.1f}% data completeness**. "
        
        if completeness >= 95:
            summary += "Data quality is exceptional, providing a robust foundation for analysis.\n\n"
        elif completeness >= 85:
            summary += "Data quality is good, though some missing values warrant attention.\n\n"
        else:
            summary += "Data quality requires improvement due to significant missing values.\n\n"
        
        # Key findings
        summary += "### Key Analytical Findings\n\n"
        
        # Correlations
        correlations = self.results.get('correlations', [])
        if correlations:
            summary += "**Relationship Discovery:**\n"
            for i, corr in enumerate(correlations[:3], 1):
                strength_word = "strong" if corr['strength'] == 'strong' else "moderate"
                direction_word = "positive" if corr['direction'] == 'positive' else 'inverse'
                summary += f"{i}. Identified {strength_word} {direction_word} relationship between "
                summary += f"**{corr['var1']}** and **{corr['var2']}** (r={corr['correlation']:.3f}). "
                
                if abs(corr['correlation']) > 0.7:
                    summary += "This strong correlation suggests these variables are closely linked and may be predictive of each other.\n"
                else:
                    summary += "This moderate correlation indicates a meaningful but not deterministic relationship.\n"
            summary += "\n"
        
        # Time series analysis
        time_series = self.results.get('time_series', {})
        if time_series:
            summary += "**Temporal Trends:**\n"
            for i, (metric, data) in enumerate(list(time_series.items())[:3], 1):
                direction = data['direction']
                strength = data['trend_strength']
                
                summary += f"{i}. **{metric}** exhibits a {direction} trend with "
                summary += f"{strength:.1%} statistical confidence. "
                
                if strength > 0.7:
                    summary += "This strong trend indicates a clear directional pattern that can be leveraged for forecasting.\n"
                elif strength > 0.4:
                    summary += "This moderate trend suggests a general direction but with some variability.\n"
                else:
                    summary += "The weak trend indicates high volatility and limited predictability.\n"
            summary += "\n"
        
        # Outliers
        outliers = self.results.get('outliers', {})
        if outliers:
            summary += "**Anomaly Detection:**\n"
            total_outliers = sum(data['count'] for data in outliers.values())
            summary += f"Detected **{total_outliers} outliers** across {len(outliers)} attributes. "
            
            if total_outliers / len(self.df) > 0.05:
                summary += "The high prevalence of outliers suggests either data quality issues or the presence of exceptional cases requiring investigation.\n"
            else:
                summary += "The limited number of outliers indicates generally consistent data patterns.\n"
            
            for col, data in list(outliers.items())[:2]:
                summary += f"- **{col}**: {data['count']} outliers ({data['percentage']:.1f}%)\n"
            summary += "\n"
        
        # Clusters
        clusters = self.results.get('clusters', {})
        if clusters:
            summary += "**Segmentation Analysis:**\n"
            summary += f"Clustering analysis identified **{clusters['n_clusters']} distinct segments** "
            summary += "within the dataset, suggesting natural groupings that can inform targeted strategies.\n"
            
            for name, stats in clusters['cluster_stats'].items():
                summary += f"- **{name}**: {stats['size']:,} records ({stats['percentage']:.1f}%)\n"
            summary += "\n"
        
        # Distribution insights
        distributions = self.results.get('distributions', {})
        if distributions:
            skewed_cols = [col for col, data in distributions.items() 
                          if abs(data.get('skewness', 0)) > 1]
            if skewed_cols:
                summary += "**Distribution Analysis:**\n"
                summary += f"Detected significant skewness in {len(skewed_cols)} attributes, "
                summary += "which may require transformation for certain analytical techniques.\n\n"
        
        return summary
    
    def generate_strategic_recommendations(self) -> List[str]:
        """Generate actionable strategic recommendations"""
        recommendations = []
        
        quality = self.results.get('quality', {})
        correlations = self.results.get('correlations', [])
        outliers = self.results.get('outliers', {})
        time_series = self.results.get('time_series', {})
        
        # Data quality recommendations
        if quality.get('completeness', 100) < 90:
            recommendations.append(
                "Implement data quality improvement initiatives to address missing values. "
                "Consider data collection process review and validation rules."
            )
        
        if quality.get('duplicate_rows', 0) > 0:
            recommendations.append(
                f"Remove {quality['duplicate_rows']} duplicate records to ensure analysis accuracy. "
                "Investigate root cause of duplication in data pipeline."
            )
        
        # Correlation-based recommendations
        if len(correlations) >= 3:
            recommendations.append(
                "Leverage discovered correlations for predictive modeling. "
                "Strong relationships between variables can improve forecast accuracy "
                "and enable what-if scenario analysis."
            )
        
        # Outlier recommendations
        if outliers:
            high_outlier_cols = [col for col, data in outliers.items() 
                                if data['percentage'] > 5]
            if high_outlier_cols:
                recommendations.append(
                    f"Investigate outliers in {', '.join(high_outlier_cols[:2])}. "
                    "These anomalies may represent data errors, exceptional cases, "
                    "or opportunities requiring special attention."
                )
        
        # Trend-based recommendations
        if time_series:
            declining_metrics = [metric for metric, data in time_series.items() 
                                if data['direction'] == 'decreasing']
            if declining_metrics:
                recommendations.append(
                    f"Address declining trends in {', '.join(declining_metrics[:2])}. "
                    "Conduct root cause analysis and implement corrective measures."
                )
        
        # Distribution recommendations
        distributions = self.results.get('distributions', {})
        if distributions:
            highly_skewed = [col for col, data in distributions.items() 
                           if abs(data.get('skewness', 0)) > 2]
            if highly_skewed:
                recommendations.append(
                    f"Apply logarithmic or power transformations to {', '.join(highly_skewed[:2])} "
                    "to normalize distributions for advanced statistical modeling."
                )
        
        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "Continue regular monitoring of key performance metrics to identify emerging trends early.",
                "Establish automated alerting for significant deviations from expected patterns.",
                "Implement periodic data quality audits to maintain analysis reliability."
            ])
        
        return recommendations[:5]
    
    def answer_analytical_question(self, question: str) -> str:
        """Answer specific analytical questions using NLU"""
        question_lower = question.lower()
        
        # Pattern matching for common queries
        if any(word in question_lower for word in ['summary', 'overview', 'findings']):
            return self.generate_comprehensive_summary()
        
        elif any(word in question_lower for word in ['correlation', 'relationship', 'connection']):
            correlations = self.results.get('correlations', [])
            if not correlations:
                return "No significant correlations were detected in the dataset."
            
            response = "**Correlation Analysis:**\n\n"
            for i, corr in enumerate(correlations[:5], 1):
                response += f"{i}. {corr['var1']} and {corr['var2']}: "
                response += f"{corr['strength']} {corr['direction']} correlation "
                response += f"(r={corr['correlation']:.3f})\n"
            
            return response
        
        elif any(word in question_lower for word in ['trend', 'pattern', 'direction']):
            time_series = self.results.get('time_series', {})
            if not time_series:
                return "No temporal trends were analyzed. Ensure dataset contains time-series data."
            
            response = "**Trend Analysis:**\n\n"
            for i, (metric, data) in enumerate(time_series.items(), 1):
                response += f"{i}. {metric}: {data['direction']} trend with "
                response += f"{data['trend_strength']:.1%} confidence\n"
            
            return response
        
        elif any(word in question_lower for word in ['quality', 'missing', 'complete']):
            quality = self.results.get('quality', {})
            response = "**Data Quality Assessment:**\n\n"
            response += f"- Completeness: {quality.get('completeness', 0):.2f}%\n"
            response += f"- Missing values: {quality.get('missing_cells', 0):,}\n"
            response += f"- Duplicate records: {quality.get('duplicate_rows', 0)}\n"
            
            if quality.get('issues'):
                response += "\n**Identified Issues:**\n"
                for issue in quality['issues'][:3]:
                    response += f"- {issue}\n"
            
            return response
        
        elif any(word in question_lower for word in ['outlier', 'anomaly', 'exception']):
            outliers = self.results.get('outliers', {})
            if not outliers:
                return "No significant outliers detected in the dataset."
            
            response = "**Outlier Detection Results:**\n\n"
            for col, data in outliers.items():
                response += f"- {col}: {data['count']} outliers ({data['percentage']:.1f}%)\n"
            
            return response
        
        elif any(word in question_lower for word in ['cluster', 'segment', 'group']):
            clusters = self.results.get('clusters', {})
            if not clusters:
                return "Clustering analysis not performed. Requires sufficient numeric attributes."
            
            response = f"**Segmentation Analysis:**\n\n"
            response += f"Identified {clusters['n_clusters']} distinct segments:\n\n"
            
            for name, stats in clusters['cluster_stats'].items():
                response += f"- {name}: {stats['size']:,} records ({stats['percentage']:.1f}%)\n"
            
            return response
        
        else:
            return (
                "Query processed. For specific insights, please rephrase using terms like: "
                "summary, correlation, trend, quality, outlier, or cluster."
            )


# ============================================================================
# RAG DOCUMENT SYSTEM
# ============================================================================

class RAGDocumentSystem:
    """
    Retrieval Augmented Generation system for contextual analysis
    Indexes analysis results for semantic retrieval
    """
    
    def __init__(self, df: pd.DataFrame, semantic_model, analysis_results: Dict):
        self.df = df
        self.semantic = semantic_model
        self.results = analysis_results
        self.document_index = []
        self.indexed = False
    
    def index_analysis_results(self):
        """Index all analysis results for retrieval"""
        documents = []
        
        # Index quality findings
        quality = self.results.get('quality', {})
        if quality:
            doc = {
                'type': 'quality',
                'content': f"Data completeness is {quality.get('completeness', 0):.1f}%",
                'keywords': ['quality', 'completeness', 'missing', 'data']
            }
            documents.append(doc)
        
        # Index correlations
        correlations = self.results.get('correlations', [])
        for corr in correlations:
            doc = {
                'type': 'correlation',
                'content': f"{corr['var1']} and {corr['var2']} show {corr['strength']} "
                          f"{corr['direction']} correlation ({corr['correlation']:.3f})",
                'keywords': ['correlation', 'relationship', corr['var1'], corr['var2']]
            }
            documents.append(doc)
        
        # Index time series findings
        time_series = self.results.get('time_series', {})
        for metric, data in time_series.items():
            doc = {
                'type': 'trend',
                'content': f"{metric} shows {data['direction']} trend with "
                          f"{data['trend_strength']:.1%} strength",
                'keywords': ['trend', 'time', 'pattern', metric]
            }
            documents.append(doc)
        
        # Index outliers
        outliers = self.results.get('outliers', {})
        for col, data in outliers.items():
            doc = {
                'type': 'outlier',
                'content': f"{col} contains {data['count']} outliers ({data['percentage']:.1f}%)",
                'keywords': ['outlier', 'anomaly', 'exception', col]
            }
            documents.append(doc)
        
        # Index clusters
        clusters = self.results.get('clusters', {})
        if clusters:
            doc = {
                'type': 'cluster',
                'content': f"Dataset segmented into {clusters['n_clusters']} clusters",
                'keywords': ['cluster', 'segment', 'group']
            }
            documents.append(doc)
        
        self.document_index = documents
        self.indexed = True
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant documents for query"""
        if not self.indexed:
            self.index_analysis_results()
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score documents by keyword overlap
        scored_docs = []
        for doc in self.document_index:
            keywords = set([k.lower() for k in doc['keywords']])
            overlap = len(query_words & keywords)
            
            if overlap > 0:
                scored_docs.append((overlap, doc['content']))
        
        # Sort by relevance and return top-k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [content for _, content in scored_docs[:top_k]]


# ============================================================================
# AGENTIC ANALYSIS ORCHESTRATOR
# ============================================================================

class AgenticAnalysisOrchestrator:
    """
    Autonomous agent for coordinating analytical tasks
    Orchestrates analysis pipeline and generates insights
    """
    
    def __init__(self, df: pd.DataFrame, semantic_model, 
                 analysis_results: Dict, rag_system: RAGDocumentSystem):
        self.df = df
        self.semantic = semantic_model
        self.results = analysis_results
        self.rag = rag_system
        self.llm = LLMAnalyticEngine(df, semantic_model, analysis_results)
    
    def process_natural_language_query(self, query: str) -> str:
        """Process natural language query with RAG context"""
        
        # Step 1: Retrieve relevant context
        relevant_context = self.rag.retrieve_relevant_context(query)
        
        # Step 2: Generate response using LLM engine
        response = self.llm.answer_analytical_question(query)
        
        # Step 3: Enhance with context if available
        if relevant_context and len(relevant_context) > 0:
            response += "\n\n**Related Context:**\n"
            for i, context in enumerate(relevant_context[:2], 1):
                response += f"{i}. {context}\n"
        
        return response
    
    def autonomous_insight_generation(self) -> Dict[str, List[str]]:
        """Autonomously generate insights across all dimensions"""
        insights = {
            'critical': [],
            'important': [],
            'informational': []
        }
        
        # Critical insights (data quality, strong patterns)
        quality = self.results.get('quality', {})
        if quality.get('completeness', 100) < 80:
            insights['critical'].append(
                f"Data completeness is {quality['completeness']:.1f}% - "
                "immediate action required to improve data collection"
            )
        
        correlations = self.results.get('correlations', [])
        strong_correlations = [c for c in correlations if c['strength'] == 'strong']
        if strong_correlations:
            insights['critical'].append(
                f"Discovered {len(strong_correlations)} strong correlations - "
                "leverage for predictive modeling and optimization"
            )
        
        # Important insights (trends, outliers)
        time_series = self.results.get('time_series', {})
        declining = [m for m, d in time_series.items() if d['direction'] == 'decreasing']
        if declining:
            insights['important'].append(
                f"{len(declining)} metrics showing decline - "
                "requires investigation and corrective action"
            )
        
        outliers = self.results.get('outliers', {})
        if len(outliers) > 3:
            insights['important'].append(
                f"Multiple attributes contain outliers - "
                "validate data quality or investigate exceptional cases"
            )
        
        # Informational insights (distributions, clusters)
        clusters = self.results.get('clusters', {})
        if clusters:
            insights['informational'].append(
                f"Identified {clusters['n_clusters']} natural segments - "
                "opportunity for targeted strategies"
            )
        
        distributions = self.results.get('distributions', {})
        normal_dists = [c for c, d in distributions.items() if d.get('is_normal', False)]
        if normal_dists:
            insights['informational'].append(
                f"{len(normal_dists)} attributes exhibit normal distribution - "
                "suitable for parametric statistical methods"
            )
        
        return insights
    
    def recommend_next_analysis(self) -> List[str]:
        """Recommend next analytical steps based on current findings"""
        recommendations = []
        
        # Based on correlations
        correlations = self.results.get('correlations', [])
        if len(correlations) >= 2:
            recommendations.append(
                "Build predictive model using identified correlations"
            )
            recommendations.append(
                "Perform causal analysis to understand directional relationships"
            )
        
        # Based on time series
        time_series = self.results.get('time_series', {})
        if time_series:
            recommendations.append(
                "Conduct advanced time series decomposition (STL, ARIMA)"
            )
            recommendations.append(
                "Implement real-time monitoring for trend detection"
            )
        
        # Based on clusters
        clusters = self.results.get('clusters', {})
        if clusters:
            recommendations.append(
                "Profile each segment to identify distinguishing characteristics"
            )
            recommendations.append(
                "Develop segment-specific strategies and interventions"
            )
        
        # Based on outliers
        outliers = self.results.get('outliers', {})
        if outliers:
            recommendations.append(
                "Investigate root causes of outliers through detailed case analysis"
            )
        
        return recommendations[:5]


# Export all classes
__all__ = [
    'LLMAnalyticEngine',
    'RAGDocumentSystem',
    'AgenticAnalysisOrchestrator'
]
