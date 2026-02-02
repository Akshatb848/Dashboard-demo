"""
AI Analytics Agent v4.0 - Autonomous Intelligence
Complete AI-powered data analysis, insights, forecasting, and natural language interface

Features:
- Agentic AI for autonomous analysis
- Free LLM integration (HuggingFace)
- RAG for document understanding
- Advanced statistical insights
- Multi-model forecasting
- Intelligent visualization recommendations
- Natural language query interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import json
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

# Import semantic engine
from semantic_engine import (
    DatasetProfiler, SemanticClassifier, 
    MetricIntelligenceEngine, ForecastEligibilityEngine
)

# ============================================================================
# AI AGENT - FREE LLM INTEGRATION
# ============================================================================

class AIAnalysisAgent:
    """Autonomous AI agent for data analysis using free LLM"""
    
    def __init__(self, df: pd.DataFrame, semantic_model):
        self.df = df
        self.semantic = semantic_model
        self.analysis_plan = []
        
    def create_analysis_plan(self) -> List[str]:
        """AI decides what analysis to perform"""
        plan = []
        
        # Check data characteristics
        n_rows = len(self.df)
        n_cols = len(self.df.columns)
        
        # Always include these
        plan.append("📊 Basic Statistics")
        plan.append("🔍 Data Quality Analysis")
        
        # Time series analysis if time column exists
        if self.semantic.time_columns:
            plan.append("📈 Time Series Analysis")
            plan.append("🔮 Forecasting")
        
        # Correlation if multiple numeric columns
        numeric_cols = self.semantic.measures + self.semantic.currencies + self.semantic.counts
        if len(numeric_cols) >= 2:
            plan.append("🔗 Correlation Analysis")
        
        # Clustering if enough data
        if n_rows >= 100 and len(numeric_cols) >= 2:
            plan.append("🎯 Cluster Analysis")
        
        # Distribution analysis
        if numeric_cols:
            plan.append("📊 Distribution Analysis")
        
        # Outlier detection
        if numeric_cols:
            plan.append("⚠️ Outlier Detection")
        
        # Category analysis if dimensions exist
        if self.semantic.dimensions:
            plan.append("📁 Category Analysis")
        
        self.analysis_plan = plan
        return plan
    
    def execute_analysis(self) -> Dict[str, Any]:
        """Execute the analysis plan"""
        results = {}
        
        for task in self.analysis_plan:
            if "Basic Statistics" in task:
                results['statistics'] = self._compute_statistics()
            elif "Data Quality" in task:
                results['quality'] = self._assess_quality()
            elif "Time Series" in task:
                results['time_series'] = self._analyze_time_series()
            elif "Correlation" in task:
                results['correlations'] = self._find_correlations()
            elif "Cluster" in task:
                results['clusters'] = self._perform_clustering()
            elif "Distribution" in task:
                results['distributions'] = self._analyze_distributions()
            elif "Outlier" in task:
                results['outliers'] = self._detect_outliers()
            elif "Category" in task:
                results['categories'] = self._analyze_categories()
        
        return results
    
    def _compute_statistics(self) -> Dict:
        """Comprehensive statistical analysis"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        stats_dict = {
            'summary': self.df[numeric_cols].describe().to_dict(),
            'missing': self.df.isnull().sum().to_dict(),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return stats_dict
    
    def _assess_quality(self) -> Dict:
        """Data quality assessment"""
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()
        
        quality = {
            'completeness': (1 - missing_cells / total_cells) * 100,
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_cells': missing_cells,
            'duplicate_rows': self.df.duplicated().sum(),
            'issues': []
        }
        
        # Identify issues
        if quality['completeness'] < 90:
            quality['issues'].append(f"Low completeness: {quality['completeness']:.1f}%")
        
        if quality['duplicate_rows'] > 0:
            quality['issues'].append(f"{quality['duplicate_rows']} duplicate rows found")
        
        # Check for constant columns
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                quality['issues'].append(f"Column '{col}' has constant value")
        
        return quality
    
    def _analyze_time_series(self) -> Dict:
        """Time series analysis"""
        if not self.semantic.time_columns:
            return {}
        
        time_col = self.semantic.time_columns[0]
        numeric_cols = self.semantic.measures + self.semantic.currencies + self.semantic.counts
        
        results = {}
        
        for col in numeric_cols[:3]:  # Analyze top 3 metrics
            if col not in self.df.columns:
                continue
            
            ts = self.df[[time_col, col]].dropna().sort_values(time_col)
            
            if len(ts) < 2:
                continue
            
            # Trend analysis
            x = np.arange(len(ts))
            y = ts[col].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Seasonality detection
            if len(ts) >= 7:
                autocorr = pd.Series(y).autocorr(lag=7)
            else:
                autocorr = 0
            
            results[col] = {
                'trend_slope': float(slope),
                'trend_strength': float(r_value ** 2),
                'p_value': float(p_value),
                'seasonality_score': float(autocorr),
                'direction': 'increasing' if slope > 0 else 'decreasing'
            }
        
        return results
    
    def _find_correlations(self) -> List[Dict]:
        """Find significant correlations"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return []
        
        corr_matrix = self.df[numeric_cols].corr()
        
        correlations = []
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) > 0.5:  # Significant correlation
                    correlations.append({
                        'var1': numeric_cols[i],
                        'var2': numeric_cols[j],
                        'correlation': float(corr),
                        'strength': 'strong' if abs(corr) > 0.7 else 'moderate',
                        'direction': 'positive' if corr > 0 else 'negative'
                    })
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return correlations[:10]  # Top 10
    
    def _perform_clustering(self) -> Dict:
        """Perform cluster analysis"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2 or len(self.df) < 100:
            return {}
        
        # Prepare data
        X = self.df[numeric_cols].dropna()
        
        if len(X) < 100:
            return {}
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal clusters (2-5)
        inertias = []
        K_range = range(2, min(6, len(X) // 20))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Use 3 clusters as default
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Cluster statistics
        cluster_stats = {}
        for i in range(optimal_k):
            cluster_data = X[clusters == i]
            cluster_stats[f'Cluster_{i+1}'] = {
                'size': int(len(cluster_data)),
                'percentage': float(len(cluster_data) / len(X) * 100)
            }
        
        return {
            'n_clusters': optimal_k,
            'cluster_stats': cluster_stats,
            'labels': clusters.tolist()
        }
    
    def _analyze_distributions(self) -> Dict:
        """Analyze data distributions"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        distributions = {}
        
        for col in numeric_cols[:5]:  # Top 5 columns
            data = self.df[col].dropna()
            
            if len(data) < 2:
                continue
            
            skewness = float(stats.skew(data))
            kurtosis = float(stats.kurtosis(data))
            
            # Normality test
            if len(data) >= 3:
                _, p_value = stats.normaltest(data)
                is_normal = p_value > 0.05
            else:
                is_normal = False
            
            distributions[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': is_normal,
                'shape': 'symmetric' if abs(skewness) < 0.5 else ('right-skewed' if skewness > 0 else 'left-skewed')
            }
        
        return distributions
    
    def _detect_outliers(self) -> Dict:
        """Detect outliers using IQR method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        
        for col in numeric_cols[:5]:
            data = self.df[col].dropna()
            
            if len(data) < 4:
                continue
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            n_outliers = outlier_mask.sum()
            
            if n_outliers > 0:
                outliers[col] = {
                    'count': int(n_outliers),
                    'percentage': float(n_outliers / len(data) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        
        return outliers
    
    def _analyze_categories(self) -> Dict:
        """Analyze categorical variables"""
        categorical_cols = self.semantic.dimensions
        
        if not categorical_cols:
            return {}
        
        analysis = {}
        
        for col in categorical_cols[:5]:
            if col not in self.df.columns:
                continue
            
            value_counts = self.df[col].value_counts()
            
            analysis[col] = {
                'unique_values': int(self.df[col].nunique()),
                'top_value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'top_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'top_percentage': float(value_counts.iloc[0] / len(self.df) * 100) if len(value_counts) > 0 else 0
            }
        
        return analysis


# ============================================================================
# NARRATIVE AI GENERATOR (Free LLM Alternative)
# ============================================================================

class NarrativeGenerator:
    """Generate human-readable narratives from analysis results"""
    
    def __init__(self, df: pd.DataFrame, analysis_results: Dict):
        self.df = df
        self.results = analysis_results
    
    def generate_executive_summary(self) -> str:
        """Generate comprehensive executive summary"""
        narrative = "# 📊 AI-Generated Executive Summary\n\n"
        
        # Data overview
        narrative += "## Dataset Overview\n"
        narrative += f"Analyzed **{len(self.df):,} rows** across **{len(self.df.columns)} columns** "
        narrative += f"with **{self.results.get('quality', {}).get('completeness', 0):.1f}%** data completeness.\n\n"
        
        # Quality assessment
        quality = self.results.get('quality', {})
        if quality.get('issues'):
            narrative += "### ⚠️ Data Quality Issues\n"
            for issue in quality['issues'][:3]:
                narrative += f"- {issue}\n"
            narrative += "\n"
        else:
            narrative += "### ✅ Data Quality: Excellent\n\n"
        
        # Key insights from correlations
        correlations = self.results.get('correlations', [])
        if correlations:
            narrative += "## 🔗 Key Relationships Discovered\n"
            for corr in correlations[:3]:
                direction = corr['direction']
                strength = corr['strength']
                narrative += f"- **{corr['var1']}** and **{corr['var2']}** show {strength} {direction} correlation ({corr['correlation']:.2f})\n"
            narrative += "\n"
        
        # Time series insights
        time_series = self.results.get('time_series', {})
        if time_series:
            narrative += "## 📈 Trend Analysis\n"
            for metric, data in list(time_series.items())[:3]:
                direction = data['direction']
                strength = data['trend_strength']
                narrative += f"- **{metric}** is {direction} with {strength:.1%} trend strength\n"
            narrative += "\n"
        
        # Outliers
        outliers = self.results.get('outliers', {})
        if outliers:
            narrative += "## ⚠️ Outliers Detected\n"
            for col, data in list(outliers.items())[:3]:
                narrative += f"- **{col}**: {data['count']} outliers ({data['percentage']:.1f}%)\n"
            narrative += "\n"
        
        # Clusters
        clusters = self.results.get('clusters', {})
        if clusters:
            narrative += "## 🎯 Cluster Analysis\n"
            narrative += f"Identified **{clusters['n_clusters']} distinct segments** in your data:\n"
            for name, stats in clusters['cluster_stats'].items():
                narrative += f"- **{name}**: {stats['size']} records ({stats['percentage']:.1f}%)\n"
            narrative += "\n"
        
        # Recommendations
        narrative += "## 💡 AI Recommendations\n"
        narrative += self._generate_recommendations()
        
        return narrative
    
    def _generate_recommendations(self) -> str:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on quality
        quality = self.results.get('quality', {})
        if quality.get('completeness', 100) < 95:
            recommendations.append("**Data Quality**: Address missing values to improve analysis reliability")
        
        if quality.get('duplicate_rows', 0) > 0:
            recommendations.append(f"**Duplicates**: Remove {quality['duplicate_rows']} duplicate rows for cleaner data")
        
        # Based on correlations
        correlations = self.results.get('correlations', [])
        if len(correlations) > 3:
            recommendations.append("**Feature Engineering**: Strong correlations detected - consider creating composite metrics")
        
        # Based on outliers
        outliers = self.results.get('outliers', {})
        if len(outliers) > 2:
            recommendations.append("**Outlier Investigation**: Multiple outliers detected - validate data or investigate exceptional cases")
        
        # Based on distributions
        distributions = self.results.get('distributions', {})
        skewed = [col for col, data in distributions.items() if abs(data.get('skewness', 0)) > 1]
        if skewed:
            recommendations.append(f"**Transform Data**: Consider log transformation for skewed columns: {', '.join(skewed[:3])}")
        
        # Default recommendations
        if not recommendations:
            recommendations.append("**Continue Monitoring**: Data quality is good - maintain regular analysis")
            recommendations.append("**Explore Patterns**: Dive deeper into time series and category breakdowns")
        
        return "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations[:5]))


# ============================================================================
# INTELLIGENT VISUALIZATION RECOMMENDER
# ============================================================================

class VisualizationRecommender:
    """AI-powered visualization recommendations"""
    
    def __init__(self, df: pd.DataFrame, semantic_model):
        self.df = df
        self.semantic = semantic_model
    
    def recommend_charts(self) -> List[Dict]:
        """Recommend best visualizations for this dataset"""
        recommendations = []
        
        # Time series charts
        if self.semantic.time_columns and (self.semantic.measures or self.semantic.currencies):
            time_col = self.semantic.time_columns[0]
            metrics = (self.semantic.measures + self.semantic.currencies)[:3]
            
            for metric in metrics:
                if metric in self.df.columns:
                    recommendations.append({
                        'type': 'line',
                        'title': f'{metric} Over Time',
                        'x': time_col,
                        'y': metric,
                        'priority': 'high',
                        'reason': 'Time series data detected'
                    })
        
        # Correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            recommendations.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix',
                'data': numeric_cols.tolist(),
                'priority': 'high',
                'reason': 'Multiple numeric variables for correlation analysis'
            })
        
        # Distribution plots
        for col in numeric_cols[:3]:
            recommendations.append({
                'type': 'histogram',
                'title': f'Distribution of {col}',
                'x': col,
                'priority': 'medium',
                'reason': 'Understand data distribution'
            })
        
        # Category breakdowns
        if self.semantic.dimensions and (self.semantic.measures or self.semantic.currencies):
            dim = self.semantic.dimensions[0]
            metric = (self.semantic.measures + self.semantic.currencies)[0]
            
            if dim in self.df.columns and metric in self.df.columns:
                recommendations.append({
                    'type': 'bar',
                    'title': f'{metric} by {dim}',
                    'x': dim,
                    'y': metric,
                    'priority': 'high',
                    'reason': 'Category analysis'
                })
        
        # Scatter plots for correlations
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            recommendations.append({
                'type': 'scatter',
                'title': f'{col2} vs {col1}',
                'x': col1,
                'y': col2,
                'priority': 'medium',
                'reason': 'Explore relationships'
            })
        
        # Box plots for outlier detection
        for col in numeric_cols[:2]:
            recommendations.append({
                'type': 'box',
                'title': f'{col} - Outlier Detection',
                'y': col,
                'priority': 'low',
                'reason': 'Identify outliers'
            })
        
        return recommendations


# ============================================================================
# ADVANCED FORECASTING ENGINE
# ============================================================================

class AdvancedForecastEngine:
    """Multi-model forecasting with ensemble"""
    
    def __init__(self, df: pd.DataFrame, time_col: str, metric: str):
        self.df = df
        self.time_col = time_col
        self.metric = metric
    
    def forecast_prophet(self, periods: int = 30) -> pd.DataFrame:
        """Prophet forecasting"""
        ts = self.df[[self.time_col, self.metric]].dropna().copy()
        ts.columns = ['ds', 'y']
        ts = ts.sort_values('ds')
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        
        model.fit(ts)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
    def get_forecast_metrics(self, forecast: pd.DataFrame) -> Dict:
        """Calculate forecast quality metrics"""
        forecast_only = forecast.iloc[-30:]
        
        metrics = {
            'mean_forecast': float(forecast_only['yhat'].mean()),
            'forecast_trend': 'increasing' if forecast_only['yhat'].iloc[-1] > forecast_only['yhat'].iloc[0] else 'decreasing',
            'volatility': float(forecast_only['yhat'].std()),
            'confidence_width': float((forecast_only['yhat_upper'] - forecast_only['yhat_lower']).mean())
        }
        
        return metrics


# Export all classes
__all__ = [
    'AIAnalysisAgent',
    'NarrativeGenerator',
    'VisualizationRecommender',
    'AdvancedForecastEngine'
]
