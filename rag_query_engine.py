"""
RAG Query Engine for Intelligent Data Analysis
Retrieves relevant context and generates answers using LLM reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json


class AnalysisKnowledgeBase:
    """
    Stores and indexes all analysis results for retrieval
    """
    
    def __init__(self):
        self.documents = []
        self.index = {}
        
    def add_document(self, doc_type: str, content: str, metadata: Dict[str, Any]):
        """Add a document to the knowledge base"""
        doc_id = len(self.documents)
        
        document = {
            'id': doc_id,
            'type': doc_type,
            'content': content,
            'metadata': metadata,
            'keywords': self._extract_keywords(content)
        }
        
        self.documents.append(document)
        
        # Index by keywords
        for keyword in document['keywords']:
            if keyword not in self.index:
                self.index[keyword] = []
            self.index[keyword].append(doc_id)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        keywords = set()
        
        text_lower = text.lower()
        
        # Statistical terms
        stats_terms = [
            'mean', 'median', 'std', 'variance', 'correlation', 'distribution',
            'normal', 'skew', 'kurtosis', 'outlier', 'missing', 'null',
            'trend', 'pattern', 'cluster', 'anomaly', 'significant'
        ]
        
        for term in stats_terms:
            if term in text_lower:
                keywords.add(term)
        
        # Extract column names (assuming they're in the metadata)
        words = text.split()
        for word in words:
            if len(word) > 3 and word[0].isupper():
                keywords.add(word.lower())
        
        return list(keywords)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score documents by keyword overlap
        scores = {}
        
        for doc in self.documents:
            score = 0
            
            # Keyword matching
            keyword_overlap = len(set(doc['keywords']) & query_words)
            score += keyword_overlap * 2
            
            # Content matching
            content_lower = doc['content'].lower()
            for word in query_words:
                if word in content_lower:
                    score += 1
            
            if score > 0:
                scores[doc['id']] = score
        
        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = self.documents[doc_id]
            results.append({
                'document': doc,
                'relevance_score': score
            })
        
        return results


class QueryUnderstanding:
    """
    Understands user queries and determines intent
    """
    
    @staticmethod
    def analyze_query(query: str) -> Dict[str, Any]:
        """Analyze query to determine intent and entities"""
        query_lower = query.lower()
        
        intent = QueryUnderstanding._determine_intent(query_lower)
        entities = QueryUnderstanding._extract_entities(query_lower)
        query_type = QueryUnderstanding._classify_query_type(query_lower)
        
        return {
            'intent': intent,
            'entities': entities,
            'query_type': query_type,
            'original_query': query
        }
    
    @staticmethod
    def _determine_intent(query: str) -> str:
        """Determine the intent of the query"""
        
        # Summary intent
        if any(word in query for word in ['summary', 'overview', 'describe', 'explain']):
            return 'summary'
        
        # Statistical intent
        elif any(word in query for word in ['mean', 'median', 'average', 'statistics', 'stats']):
            return 'statistics'
        
        # Correlation intent
        elif any(word in query for word in ['correlation', 'relationship', 'related', 'connection']):
            return 'correlation'
        
        # Trend intent
        elif any(word in query for word in ['trend', 'pattern', 'change', 'over time']):
            return 'trend'
        
        # Quality intent
        elif any(word in query for word in ['quality', 'missing', 'null', 'incomplete']):
            return 'quality'
        
        # Outlier intent
        elif any(word in query for word in ['outlier', 'anomaly', 'unusual', 'extreme']):
            return 'outlier'
        
        # Recommendation intent
        elif any(word in query for word in ['recommend', 'suggest', 'should', 'advice']):
            return 'recommendation'
        
        # Comparison intent
        elif any(word in query for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        
        # Prediction intent
        elif any(word in query for word in ['predict', 'forecast', 'future', 'will']):
            return 'prediction'
        
        else:
            return 'general'
    
    @staticmethod
    def _extract_entities(query: str) -> List[str]:
        """Extract entities (column names, values) from query"""
        entities = []
        
        # Simple entity extraction
        words = query.split()
        for word in words:
            # Capitalized words might be column names
            if word[0].isupper() and len(word) > 2:
                entities.append(word)
        
        return entities
    
    @staticmethod
    def _classify_query_type(query: str) -> str:
        """Classify the type of query"""
        
        if '?' in query:
            return 'question'
        elif any(word in query for word in ['show', 'display', 'plot', 'visualize']):
            return 'visualization_request'
        elif any(word in query for word in ['how', 'why', 'what', 'when', 'where']):
            return 'information_seeking'
        else:
            return 'statement'


class IntelligentQueryEngine:
    """
    Main RAG query engine that answers user questions
    """
    
    def __init__(self, df: pd.DataFrame, analysis_results: Dict[str, Any]):
        self.df = df
        self.analysis_results = analysis_results
        self.knowledge_base = AnalysisKnowledgeBase()
        self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Build knowledge base from analysis results"""
        
        # Index data understanding
        if 'data_understanding' in self.analysis_results:
            understanding = self.analysis_results['data_understanding']
            content = f"Dataset has {understanding['shape']['rows']} rows and {understanding['shape']['columns']} columns."
            self.knowledge_base.add_document('data_overview', content, understanding)
        
        # Index quality assessment
        if 'quality_assessment' in self.analysis_results:
            quality = self.analysis_results['quality_assessment']
            content = f"Data completeness is {quality['completeness']['score']:.2f}% with {quality['completeness']['missing_cells']} missing values."
            self.knowledge_base.add_document('quality', content, quality)
        
        # Index feature analysis
        if 'feature_analysis' in self.analysis_results:
            features = self.analysis_results['feature_analysis']
            
            # Index correlations
            for corr in features.get('correlations', []):
                content = f"{corr['feature1']} and {corr['feature2']} have {corr['strength']} {corr['direction']} correlation of {corr['correlation']:.3f}"
                self.knowledge_base.add_document('correlation', content, corr)
        
        # Index preprocessing recommendations
        if 'preprocessing_recommendations' in self.analysis_results:
            for rec in self.analysis_results['preprocessing_recommendations']:
                content = f"Recommend {rec['step']} for {rec.get('column', 'dataset')} because {rec['reason']}"
                self.knowledge_base.add_document('recommendation', content, rec)
    
    def answer_query(self, query: str) -> str:
        """
        Answer user query using RAG approach
        """
        
        # Step 1: Understand query
        query_analysis = QueryUnderstanding.analyze_query(query)
        
        # Step 2: Retrieve relevant context
        relevant_docs = self.knowledge_base.search(query, top_k=3)
        
        # Step 3: Generate answer based on intent
        answer = self._generate_answer(query_analysis, relevant_docs)
        
        return answer
    
    def _generate_answer(self, query_analysis: Dict[str, Any], relevant_docs: List[Dict]) -> str:
        """
        Generate answer based on query intent and retrieved context
        """
        intent = query_analysis['intent']
        query = query_analysis['original_query']
        
        # Start with context
        answer = ""
        
        if intent == 'summary':
            answer = self._generate_summary()
        
        elif intent == 'statistics':
            answer = self._generate_statistics_answer(query)
        
        elif intent == 'correlation':
            answer = self._generate_correlation_answer(relevant_docs)
        
        elif intent == 'trend':
            answer = self._generate_trend_answer()
        
        elif intent == 'quality':
            answer = self._generate_quality_answer()
        
        elif intent == 'outlier':
            answer = self._generate_outlier_answer()
        
        elif intent == 'recommendation':
            answer = self._generate_recommendation_answer()
        
        elif intent == 'comparison':
            answer = self._generate_comparison_answer(query_analysis['entities'])
        
        else:
            # General answer using relevant documents
            if relevant_docs:
                answer = "Based on the analysis:\n\n"
                for i, doc_result in enumerate(relevant_docs[:3], 1):
                    doc = doc_result['document']
                    answer += f"{i}. {doc['content']}\n"
            else:
                answer = "I don't have enough information in the current analysis to answer that question. Please try rephrasing or ask about available insights."
        
        return answer
    
    def _generate_summary(self) -> str:
        """Generate comprehensive summary"""
        understanding = self.analysis_results.get('data_understanding', {})
        quality = self.analysis_results.get('quality_assessment', {})
        features = self.analysis_results.get('feature_analysis', {})
        
        summary = "## Dataset Summary\n\n"
        
        # Basic info
        if understanding:
            summary += f"The dataset contains **{understanding['shape']['rows']:,} records** "
            summary += f"and **{understanding['shape']['columns']} features**.\n\n"
        
        # Quality
        if quality:
            summary += f"Data quality is **{quality['completeness']['score']:.1f}%** complete "
            summary += f"with **{quality['completeness']['missing_cells']}** missing values.\n\n"
        
        # Key findings
        if features and features.get('correlations'):
            summary += f"Found **{len(features['correlations'])}** significant correlations between features.\n\n"
        
        return summary
    
    def _generate_statistics_answer(self, query: str) -> str:
        """Generate statistics answer"""
        understanding = self.analysis_results.get('data_understanding', {})
        
        answer = "## Statistical Overview\n\n"
        
        for col, info in understanding.get('columns', {}).items():
            if 'statistics' in info:
                stats = info['statistics']
                answer += f"**{col}**:\n"
                answer += f"- Mean: {stats['mean']:.2f}\n"
                answer += f"- Median: {stats['median']:.2f}\n"
                answer += f"- Std Dev: {stats['std']:.2f}\n"
                answer += f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n\n"
        
        return answer
    
    def _generate_correlation_answer(self, relevant_docs: List[Dict]) -> str:
        """Generate correlation answer"""
        features = self.analysis_results.get('feature_analysis', {})
        
        if not features.get('correlations'):
            return "No significant correlations were found in the dataset."
        
        answer = "## Correlation Analysis\n\n"
        answer += "The following significant correlations were discovered:\n\n"
        
        for i, corr in enumerate(features['correlations'][:5], 1):
            answer += f"{i}. **{corr['feature1']}** and **{corr['feature2']}**: "
            answer += f"{corr['strength']} {corr['direction']} correlation ({corr['correlation']:.3f})\n"
        
        return answer
    
    def _generate_trend_answer(self) -> str:
        """Generate trend answer"""
        patterns = self.analysis_results.get('pattern_discovery', {})
        
        if not patterns or not patterns.get('trends'):
            return "No significant trends were detected in the dataset."
        
        answer = "## Trend Analysis\n\n"
        
        for trend in patterns['trends']:
            answer += f"- **{trend['feature']}** shows a {trend['direction']} trend "
            answer += f"with {trend['strength']:.2f} strength"
            if trend.get('is_significant'):
                answer += " (statistically significant)"
            answer += "\n"
        
        return answer
    
    def _generate_quality_answer(self) -> str:
        """Generate data quality answer"""
        quality = self.analysis_results.get('quality_assessment', {})
        
        if not quality:
            return "Data quality assessment not available."
        
        answer = "## Data Quality Report\n\n"
        answer += f"Overall completeness: **{quality['completeness']['score']:.2f}%**\n\n"
        
        if quality.get('issues'):
            answer += "### Identified Issues:\n\n"
            for issue in quality['issues'][:5]:
                answer += f"- **{issue.get('severity', '').upper()}**: {issue['issue']}\n"
        else:
            answer += "No significant quality issues detected.\n"
        
        return answer
    
    def _generate_outlier_answer(self) -> str:
        """Generate outlier answer"""
        quality = self.analysis_results.get('quality_assessment', {})
        
        if not quality or not quality.get('accuracy', {}).get('outlier_counts'):
            return "No significant outliers detected."
        
        answer = "## Outlier Analysis\n\n"
        
        for col, count in quality['accuracy']['outlier_counts'].items():
            percentage = (count / len(self.df)) * 100
            answer += f"- **{col}**: {count} outliers ({percentage:.1f}% of data)\n"
        
        return answer
    
    def _generate_recommendation_answer(self) -> str:
        """Generate preprocessing recommendations"""
        recommendations = self.analysis_results.get('preprocessing_recommendations', [])
        
        if not recommendations:
            return "No preprocessing recommendations available."
        
        answer = "## Recommended Actions\n\n"
        
        high_priority = [r for r in recommendations if r['priority'] == 'high']
        
        if high_priority:
            answer += "### High Priority:\n\n"
            for i, rec in enumerate(high_priority[:5], 1):
                answer += f"{i}. **{rec['step'].replace('_', ' ').title()}**: "
                answer += f"{rec.get('column', 'Multiple columns')}\n"
                answer += f"   - Reason: {rec['reason']}\n"
                answer += f"   - Method: {rec.get('method', 'N/A')}\n\n"
        
        return answer
    
    def _generate_comparison_answer(self, entities: List[str]) -> str:
        """Generate comparison answer"""
        if len(entities) < 2:
            return "Please specify which features you'd like to compare."
        
        # Try to find these entities in the dataset
        available_cols = []
        for entity in entities:
            for col in self.df.columns:
                if entity.lower() in col.lower():
                    available_cols.append(col)
                    break
        
        if len(available_cols) < 2:
            return f"Could not find features matching: {', '.join(entities)}"
        
        answer = f"## Comparison: {' vs '.join(available_cols[:2])}\n\n"
        
        for col in available_cols[:2]:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                answer += f"**{col}**:\n"
                answer += f"- Mean: {self.df[col].mean():.2f}\n"
                answer += f"- Std: {self.df[col].std():.2f}\n"
                answer += f"- Range: [{self.df[col].min():.2f}, {self.df[col].max():.2f}]\n\n"
        
        return answer


# Export
__all__ = ['IntelligentQueryEngine', 'AnalysisKnowledgeBase', 'QueryUnderstanding']
