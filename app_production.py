"""
AI Analytics Dashboard v3.0 - Production Edition
Industry-Grade Analytics Platform with Semantic Intelligence

Architecture Flow:
User Upload → Semantic Profiler → Business Model Generator → 
Metric Intelligence → Prediction Eligibility → Multi-model Forecast →
Insight Engine → Narrative AI → Visualization

Author: Akshat Banga
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
import json
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

# Import semantic engine
from semantic_engine import (
    DatasetProfiler, SemanticClassifier, 
    MetricIntelligenceEngine, ForecastEligibilityEngine,
    KPICandidate, ForecastEligibility
)

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AI Analytics Dashboard v3.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'semantic_model' not in st.session_state:
    st.session_state.semantic_model = None
if 'kpi_candidates' not in st.session_state:
    st.session_state.kpi_candidates = []
if 'insights' not in st.session_state:
    st.session_state.insights = {}

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #10b981;
        --accent: #f59e0b;
        --danger: #ef4444;
        --background: #0f172a;
        --surface: #1e293b;
        --text: #f1f5f9;
        --text-muted: #94a3b8;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(16, 185, 129, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #10b981, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(51, 65, 85, 0.7));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(16, 185, 129, 0.05));
        border-left: 4px solid #6366f1;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .insight-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .narrative-section {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(30, 41, 59, 0.9));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# BUSINESS MODEL GENERATOR
# ============================================================================

class BusinessModelGenerator:
    """Generates business intelligence model from semantic classification"""
    
    def __init__(self, df: pd.DataFrame, semantic_model):
        self.df = df
        self.semantic = semantic_model
    
    def generate_model(self) -> Dict[str, Any]:
        """Generate complete business model"""
        return {
            'entity_type': self._infer_entity_type(),
            'grain': self._determine_grain(),
            'primary_time_dimension': self.semantic.time_columns[0] if self.semantic.time_columns else None,
            'fact_tables': self._identify_facts(),
            'dimension_hierarchy': self._build_hierarchy(),
            'aggregation_rules': self._define_aggregations()
        }
    
    def _infer_entity_type(self) -> str:
        """Infer what the dataset represents"""
        cols_lower = [c.lower() for c in self.df.columns]
        
        if any('customer' in c or 'client' in c for c in cols_lower):
            return "Customer Transactions"
        elif any('product' in c or 'item' in c for c in cols_lower):
            return "Product Sales"
        elif any('employee' in c or 'staff' in c for c in cols_lower):
            return "HR/Workforce"
        elif any('order' in c for c in cols_lower):
            return "Order Management"
        else:
            return "General Business Data"
    
    def _determine_grain(self) -> str:
        """Determine granularity of data"""
        if len(self.df) > 100000:
            return "Transactional"
        elif len(self.df) > 1000:
            return "Operational"
        else:
            return "Summarized"
    
    def _identify_facts(self) -> List[str]:
        """Identify fact table candidates"""
        return self.semantic.measures + self.semantic.currencies + self.semantic.counts
    
    def _build_hierarchy(self) -> Dict[str, List[str]]:
        """Build dimension hierarchies"""
        hierarchy = {}
        
        if self.semantic.time_columns:
            hierarchy['Time'] = self.semantic.time_columns
        
        if self.semantic.dimensions:
            hierarchy['Descriptive'] = self.semantic.dimensions
        
        return hierarchy
    
    def _define_aggregations(self) -> Dict[str, str]:
        """Define how metrics should aggregate"""
        rules = {}
        
        for col in self.semantic.currencies:
            rules[col] = 'SUM'
        
        for col in self.semantic.counts:
            rules[col] = 'SUM'
        
        for col in self.semantic.rates:
            rules[col] = 'AVG'
        
        for col in self.semantic.measures:
            if col not in rules:
                rules[col] = 'SUM'
        
        return rules


# ============================================================================
# INSIGHT ENGINE
# ============================================================================

@dataclass
class Insight:
    """Represents a business insight"""
    type: str
    priority: str
    title: str
    description: str
    metric: Optional[str] = None
    value: Optional[float] = None
    change: Optional[float] = None
    
    def to_dict(self):
        return {
            'type': self.type,
            'priority': self.priority,
            'title': self.title,
            'description': self.description,
            'metric': self.metric,
            'value': self.value,
            'change': self.change
        }


class InsightEngine:
    """Generates automated insights from data"""
    
    def __init__(self, df: pd.DataFrame, semantic_model, kpis: List[KPICandidate]):
        self.df = df
        self.semantic = semantic_model
        self.kpis = kpis
    
    def generate_insights(self) -> List[Insight]:
        """Generate all insights"""
        insights = []
        
        # KPI insights
        insights.extend(self._analyze_kpis())
        
        # Trend insights
        if self.semantic.time_columns:
            insights.extend(self._analyze_trends())
        
        # Distribution insights
        insights.extend(self._analyze_distributions())
        
        # Correlation insights
        insights.extend(self._analyze_correlations())
        
        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        insights.sort(key=lambda x: priority_order.get(x.priority, 999))
        
        return insights
    
    def _analyze_kpis(self) -> List[Insight]:
        """Analyze KPI metrics"""
        insights = []
        
        for kpi in self.kpis[:5]:  # Top 5 KPIs
            if kpi.column not in self.df.columns:
                continue
            
            series = self.df[kpi.column].dropna()
            
            if len(series) == 0:
                continue
            
            # Summary statistics
            mean_val = series.mean()
            std_val = series.std()
            cv = std_val / mean_val if mean_val != 0 else 0
            
            # Volatility insight
            if cv > 0.5:
                insights.append(Insight(
                    type='KPI',
                    priority='High',
                    title=f'{kpi.column} shows high volatility',
                    description=f'Coefficient of variation is {cv:.2f}, indicating significant fluctuation',
                    metric=kpi.column,
                    value=mean_val,
                    change=cv
                ))
            
            # Outlier insight
            outliers = self._detect_outliers(series)
            if len(outliers) > len(series) * 0.05:
                insights.append(Insight(
                    type='Anomaly',
                    priority='Medium',
                    title=f'{kpi.column} contains {len(outliers)} outliers',
                    description=f'{len(outliers)/len(series):.1%} of values are statistical outliers',
                    metric=kpi.column
                ))
        
        return insights
    
    def _analyze_trends(self) -> List[Insight]:
        """Analyze time series trends"""
        insights = []
        
        time_col = self.semantic.time_columns[0]
        
        for measure in (self.semantic.measures + self.semantic.currencies)[:3]:
            if measure not in self.df.columns:
                continue
            
            ts = self.df[[time_col, measure]].dropna().sort_values(time_col)
            
            if len(ts) < 2:
                continue
            
            # Calculate trend
            values = ts[measure].values
            trend = self._calculate_trend(values)
            
            if abs(trend) > 0.1:  # Significant trend
                direction = "increasing" if trend > 0 else "decreasing"
                insights.append(Insight(
                    type='Trend',
                    priority='High' if abs(trend) > 0.3 else 'Medium',
                    title=f'{measure} is {direction}',
                    description=f'Detected {direction} trend with {abs(trend):.1%} rate of change',
                    metric=measure,
                    change=trend
                ))
        
        return insights
    
    def _analyze_distributions(self) -> List[Insight]:
        """Analyze value distributions"""
        insights = []
        
        for col in self.semantic.measures[:3]:
            if col not in self.df.columns:
                continue
            
            series = self.df[col].dropna()
            
            if len(series) < 10:
                continue
            
            # Skewness
            from scipy.stats import skew
            skewness = skew(series)
            
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                insights.append(Insight(
                    type='Distribution',
                    priority='Low',
                    title=f'{col} is {direction}-skewed',
                    description=f'Distribution shows {direction} skew (skewness={skewness:.2f})',
                    metric=col
                ))
        
        return insights
    
    def _analyze_correlations(self) -> List[Insight]:
        """Find significant correlations"""
        insights = []
        
        numeric_cols = self.semantic.measures + self.semantic.currencies + self.semantic.counts
        numeric_cols = [c for c in numeric_cols if c in self.df.columns]
        
        if len(numeric_cols) < 2:
            return insights
        
        # Correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find strong correlations
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) > 0.7:  # Strong correlation
                    relationship = "positive" if corr > 0 else "negative"
                    insights.append(Insight(
                        type='Correlation',
                        priority='Medium',
                        title=f'Strong {relationship} correlation found',
                        description=f'{numeric_cols[i]} and {numeric_cols[j]} are {relationship}ly correlated (r={corr:.2f})',
                        metric=f'{numeric_cols[i]} vs {numeric_cols[j]}',
                        value=corr
                    ))
        
        return insights
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return series[(series < lower_bound) | (series > upper_bound)]
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend as percentage change"""
        if len(values) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Convert to percentage
        mean_val = np.mean(values)
        return (slope * len(values)) / mean_val if mean_val != 0 else 0.0


# ============================================================================
# NARRATIVE AI ENGINE (FREE LLM)
# ============================================================================

class NarrativeAIEngine:
    """Generates natural language narratives from insights"""
    
    def __init__(self, insights: List[Insight], semantic_model):
        self.insights = insights
        self.semantic = semantic_model
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary narrative"""
        
        if not self.insights:
            return "No significant insights detected in the dataset."
        
        # Count insight types
        critical = sum(1 for i in self.insights if i.priority == 'Critical')
        high = sum(1 for i in self.insights if i.priority == 'High')
        
        narrative = f"## Executive Summary\n\n"
        narrative += f"Analysis identified **{len(self.insights)} insights** "
        narrative += f"({critical} critical, {high} high priority).\n\n"
        
        # Top insights
        narrative += "### Key Findings\n\n"
        for insight in self.insights[:5]:
            narrative += f"**{insight.priority}:** {insight.title}\n"
            narrative += f"- {insight.description}\n\n"
        
        # Recommendations
        narrative += "### Recommendations\n\n"
        narrative += self._generate_recommendations()
        
        return narrative
    
    def generate_insight_narrative(self, insight: Insight) -> str:
        """Generate detailed narrative for single insight"""
        narrative = f"### {insight.title}\n\n"
        narrative += f"**Type:** {insight.type} | **Priority:** {insight.priority}\n\n"
        narrative += insight.description + "\n\n"
        
        # Add context-specific details
        if insight.type == 'Trend':
            narrative += self._trend_context(insight)
        elif insight.type == 'Anomaly':
            narrative += self._anomaly_context(insight)
        elif insight.type == 'Correlation':
            narrative += self._correlation_context(insight)
        
        return narrative
    
    def _generate_recommendations(self) -> str:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Volatility recommendations
        volatility_insights = [i for i in self.insights if 'volatility' in i.title.lower()]
        if volatility_insights:
            recommendations.append(
                "- **Monitor volatile metrics** closely and establish alert thresholds"
            )
        
        # Trend recommendations
        trend_insights = [i for i in self.insights if i.type == 'Trend']
        if trend_insights:
            decreasing = sum(1 for i in trend_insights if i.change and i.change < 0)
            if decreasing > 0:
                recommendations.append(
                    f"- **Investigate declining trends** in {decreasing} metric(s)"
                )
        
        # Outlier recommendations
        outlier_insights = [i for i in self.insights if 'outlier' in i.title.lower()]
        if outlier_insights:
            recommendations.append(
                "- **Review outliers** for data quality or exceptional events"
            )
        
        if not recommendations:
            recommendations.append("- Continue monitoring key metrics")
            recommendations.append("- Schedule regular data quality reviews")
        
        return "\n".join(recommendations)
    
    def _trend_context(self, insight: Insight) -> str:
        """Add context for trend insights"""
        if insight.change and insight.change > 0:
            return "**Implication:** Growth opportunity or potential demand increase.\n"
        else:
            return "**Implication:** Potential concern requiring investigation.\n"
    
    def _anomaly_context(self, insight: Insight) -> str:
        """Add context for anomaly insights"""
        return "**Action:** Verify data quality and investigate root causes.\n"
    
    def _correlation_context(self, insight: Insight) -> str:
        """Add context for correlation insights"""
        return "**Opportunity:** Leverage relationship for predictive modeling.\n"


# ============================================================================
# MULTI-MODEL FORECAST ENGINE
# ============================================================================

class MultiModelForecastEngine:
    """Forecasting with multiple models and ensemble"""
    
    def __init__(self, df: pd.DataFrame, time_col: str, metric: str):
        self.df = df
        self.time_col = time_col
        self.metric = metric
    
    def forecast_prophet(self, periods: int = 30) -> pd.DataFrame:
        """Forecast using Prophet model"""
        # Prepare data
        ts = self.df[[self.time_col, self.metric]].dropna().copy()
        ts.columns = ['ds', 'y']
        ts = ts.sort_values('ds')
        
        # Train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(ts)
        
        # Generate forecast
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def create_forecast_visualization(self, forecast_df: pd.DataFrame) -> go.Figure:
        """Create interactive forecast visualization"""
        
        fig = go.Figure()
        
        # Historical data
        historical = self.df[[self.time_col, self.metric]].dropna()
        
        fig.add_trace(go.Scatter(
            x=historical[self.time_col],
            y=historical[self.metric],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#6366f1', width=2),
            marker=dict(size=6)
        ))
        
        # Forecast
        forecast_only = forecast_df.iloc[-30:]  # Last 30 periods
        
        fig.add_trace(go.Scatter(
            x=forecast_only['ds'],
            y=forecast_only['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#10b981', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_only['ds'],
            y=forecast_only['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_only['ds'],
            y=forecast_only['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(16, 185, 129, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'{self.metric} Forecast',
            xaxis_title='Date',
            yaxis_title=self.metric,
            template='plotly_dark',
            hovermode='x unified',
            height=500
        )
        
        return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application flow"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Analytics Dashboard v3.0</h1>
        <p>Industry-Grade Analytics with Semantic Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your data file to begin analysis"
        )
        
        if uploaded_file:
            # Load data
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
        
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### ⚙️ Settings")
            
            show_debug = st.checkbox("Show Debug Info", value=False)
    
    # Main content
    if st.session_state.df is None:
        st.info("👆 Upload a CSV or Excel file to begin analysis")
        
        # Demo section
        st.markdown("### 🎯 Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Semantic Profiling**
            - Auto-detect column types
            - Business model generation
            - KPI discovery
            """)
        
        with col2:
            st.markdown("""
            **🔮 Predictive Analytics**
            - Forecast eligibility check
            - Prophet-based forecasting
            - Confidence intervals
            """)
        
        with col3:
            st.markdown("""
            **💡 AI Insights**
            - Automated insight generation
            - Narrative AI summaries
            - Actionable recommendations
            """)
        
        return
    
    df = st.session_state.df
    
    # ========================================================================
    # STEP 1: SEMANTIC PROFILER ENGINE
    # ========================================================================
    
    with st.spinner("🔍 Running semantic profiler..."):
        profiler = DatasetProfiler(df)
        profiles = profiler.profile()
        
        classifier = SemanticClassifier(profiles)
        semantic_model = classifier.classify()
        
        st.session_state.semantic_model = semantic_model
    
    # ========================================================================
    # STEP 2: BUSINESS MODEL GENERATOR
    # ========================================================================
    
    with st.spinner("🏢 Generating business model..."):
        bm_generator = BusinessModelGenerator(df, semantic_model)
        business_model = bm_generator.generate_model()
    
    # ========================================================================
    # STEP 3: METRIC INTELLIGENCE ENGINE
    # ========================================================================
    
    with st.spinner("📊 Discovering KPIs..."):
        metric_engine = MetricIntelligenceEngine(df, semantic_model)
        kpi_candidates = metric_engine.discover_kpis()
        
        st.session_state.kpi_candidates = kpi_candidates
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    tabs = st.tabs([
        "📊 Overview",
        "🎯 KPIs & Metrics",
        "🔮 Forecasting",
        "💡 Insights",
        "📈 Visualizations"
    ])
    
    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    
    with tabs[0]:
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Rows</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df.columns)}</div>
                <div class="metric-label">Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{completeness:.1f}%</div>
                <div class="metric-label">Complete</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{memory_mb:.1f}MB</div>
                <div class="metric-label">Memory</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🏢 Business Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Entity Type:** {business_model['entity_type']}  
            **Grain:** {business_model['grain']}  
            **Time Dimension:** {business_model['primary_time_dimension'] or 'None'}
            """)
        
        with col2:
            st.markdown("**Semantic Classification:**")
            st.markdown(f"- ⏰ Time Columns: {len(semantic_model.time_columns)}")
            st.markdown(f"- 📊 Measures: {len(semantic_model.measures)}")
            st.markdown(f"- 📁 Dimensions: {len(semantic_model.dimensions)}")
            st.markdown(f"- 🔑 Identifiers: {len(semantic_model.identifiers)}")
        
        st.markdown("---")
        
        st.markdown("### 📄 Data Preview")
        st.dataframe(df.head(100), use_container_width=True, height=400)
    
    # ========================================================================
    # TAB 2: KPIs & METRICS
    # ========================================================================
    
    with tabs[1]:
        st.markdown("### 🎯 Discovered KPIs")
        
        if not kpi_candidates:
            st.info("No KPIs discovered. Upload data with numeric measures.")
        else:
            st.markdown(f"Found **{len(kpi_candidates)} KPI candidates** ranked by business value")
            
            for i, kpi in enumerate(kpi_candidates[:10], 1):
                with st.expander(f"#{i} {kpi.column} - Score: {kpi.score:.0f}/100"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Category:** {kpi.category}")
                        st.markdown("**Rationale:**")
                        for reason in kpi.rationale:
                            st.markdown(f"- {reason}")
                    
                    with col2:
                        # Show statistics
                        if kpi.column in df.columns:
                            series = df[kpi.column].dropna()
                            
                            st.metric("Mean", f"{series.mean():.2f}")
                            st.metric("Std Dev", f"{series.std():.2f}")
                            st.metric("Range", f"{series.min():.2f} - {series.max():.2f}")
                    
                    # Visualization
                    if kpi.column in df.columns:
                        fig = px.histogram(
                            df,
                            x=kpi.column,
                            title=f"Distribution of {kpi.column}",
                            template='plotly_dark',
                            color_discrete_sequence=['#6366f1']
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 3: FORECASTING
    # ========================================================================
    
    with tabs[2]:
        st.markdown("### 🔮 Predictive Forecasting")
        
        if not semantic_model.time_columns:
            st.warning("⚠️ No time column detected. Forecasting requires temporal data.")
        else:
            time_col = semantic_model.time_columns[0]
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                except:
                    st.error(f"Cannot convert {time_col} to datetime")
                    return
            
            st.markdown(f"**Time Column:** {time_col}")
            
            # Select metric to forecast
            forecast_candidates = (semantic_model.measures + 
                                 semantic_model.currencies + 
                                 semantic_model.counts)
            
            if not forecast_candidates:
                st.info("No numeric measures available for forecasting")
            else:
                selected_metric = st.selectbox(
                    "Select metric to forecast",
                    forecast_candidates
                )
                
                # Eligibility check
                st.markdown("#### 📋 Forecast Eligibility Check")
                
                eligibility_engine = ForecastEligibilityEngine(
                    df, time_col, selected_metric
                )
                eligibility = eligibility_engine.check()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    status = "success" if eligibility.is_eligible else "error"
                    st.markdown(f"""
                    <div class="status-badge status-{status}">
                        {'✅ ELIGIBLE' if eligibility.is_eligible else '❌ NOT ELIGIBLE'}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Eligibility Score", f"{eligibility.score:.0f}/100")
                
                with col3:
                    st.metric("Warnings", len(eligibility.warnings))
                
                st.markdown(f"**Reason:** {eligibility.reason}")
                
                if eligibility.warnings:
                    with st.expander("⚠️ View Warnings"):
                        for warning in eligibility.warnings:
                            st.warning(warning)
                
                # Forecasting
                if eligibility.is_eligible:
                    st.markdown("---")
                    st.markdown("#### 📈 Generate Forecast")
                    
                    forecast_periods = st.slider(
                        "Forecast periods",
                        min_value=7,
                        max_value=90,
                        value=30,
                        step=7
                    )
                    
                    if st.button("🚀 Generate Forecast", type="primary"):
                        with st.spinner("Running Prophet forecast model..."):
                            try:
                                forecast_engine = MultiModelForecastEngine(
                                    df, time_col, selected_metric
                                )
                                
                                forecast_df = forecast_engine.forecast_prophet(
                                    periods=forecast_periods
                                )
                                
                                # Visualization
                                fig = forecast_engine.create_forecast_visualization(forecast_df)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast summary
                                st.markdown("#### 📊 Forecast Summary")
                                
                                forecast_only = forecast_df.iloc[-forecast_periods:]
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    avg_forecast = forecast_only['yhat'].mean()
                                    st.metric("Avg Forecast", f"{avg_forecast:.2f}")
                                
                                with col2:
                                    trend = ((forecast_only['yhat'].iloc[-1] - 
                                             forecast_only['yhat'].iloc[0]) / 
                                            forecast_only['yhat'].iloc[0] * 100)
                                    st.metric("Trend", f"{trend:+.1f}%")
                                
                                with col3:
                                    uncertainty = (forecast_only['yhat_upper'] - 
                                                 forecast_only['yhat_lower']).mean()
                                    st.metric("Avg Uncertainty", f"±{uncertainty:.2f}")
                                
                                # Download
                                csv = forecast_only.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Forecast",
                                    csv,
                                    "forecast.csv",
                                    "text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"Forecasting error: {e}")
                else:
                    st.info("💡 Improve eligibility by:\n- Adding more historical data\n- Ensuring regular time intervals\n- Reducing volatility")
    
    # ========================================================================
    # TAB 4: INSIGHTS
    # ========================================================================
    
    with tabs[3]:
        st.markdown("### 💡 AI-Generated Insights")
        
        with st.spinner("🤖 Generating insights..."):
            insight_engine = InsightEngine(df, semantic_model, kpi_candidates)
            insights = insight_engine.generate_insights()
            
            st.session_state.insights = insights
        
        if not insights:
            st.info("No significant insights detected")
        else:
            # Generate narrative
            narrative_engine = NarrativeAIEngine(insights, semantic_model)
            executive_summary = narrative_engine.generate_executive_summary()
            
            st.markdown("""
            <div class="narrative-section">
            """, unsafe_allow_html=True)
            
            st.markdown(executive_summary)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Individual insights
            st.markdown("### 📋 Detailed Insights")
            
            priority_filter = st.multiselect(
                "Filter by priority",
                ["Critical", "High", "Medium", "Low"],
                default=["Critical", "High", "Medium"]
            )
            
            filtered_insights = [i for i in insights if i.priority in priority_filter]
            
            for insight in filtered_insights:
                priority_class = {
                    'Critical': 'error',
                    'High': 'warning',
                    'Medium': 'success',
                    'Low': 'success'
                }.get(insight.priority, 'success')
                
                st.markdown(f"""
                <div class="insight-card">
                    <span class="status-badge status-{priority_class}">{insight.priority}</span>
                    <h4>{insight.title}</h4>
                    <p>{insight.description}</p>
                    {'<p><strong>Metric:</strong> ' + insight.metric + '</p>' if insight.metric else ''}
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 5: VISUALIZATIONS
    # ========================================================================
    
    with tabs[4]:
        st.markdown("### 📈 Interactive Visualizations")
        
        # Chart builder
        col1, col2 = st.columns([2, 1])
        
        with col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Box Plot", "Heatmap"]
            )
        
        with col2:
            if chart_type == "Heatmap":
                st.info("Heatmap of numeric correlations")
            else:
                x_col = st.selectbox("X-axis", df.columns)
                y_col = st.selectbox("Y-axis", [c for c in df.columns if c != x_col])
        
        # Generate chart
        if st.button("Generate Chart"):
            if chart_type == "Heatmap":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    corr = df[numeric_cols].corr()
                    
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        title="Correlation Heatmap",
                        template='plotly_dark',
                        color_continuous_scale='RdBu'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for heatmap")
            
            elif chart_type == "Line Chart":
                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over {x_col}",
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Bar Chart":
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} by {x_col}",
                    template='plotly_dark',
                    color_discrete_sequence=['#6366f1']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Scatter Plot":
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}",
                    template='plotly_dark',
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Box Plot":
                fig = px.box(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} distribution by {x_col}",
                    template='plotly_dark',
                    color_discrete_sequence=['#6366f1']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 2rem;">
        <p><strong>AI Analytics Dashboard v3.0</strong> | Production Edition</p>
        <p>Built with Semantic Intelligence • Prophet Forecasting • Automated Insights</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
