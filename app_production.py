"""
Advanced Analytics Platform v5.0 - Enterprise Edition
Production-grade analytics with integrated AI intelligence, RAG-powered insights, and agentic analysis

Built to rival Tableau AI and Power BI Copilot with autonomous analytical capabilities
Author: Akshat Banga
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import json
from typing import Dict, List, Any, Optional
warnings.filterwarnings('ignore')

# Import core engines
from semantic_engine import (
    DatasetProfiler, SemanticClassifier, 
    MetricIntelligenceEngine, ForecastEligibilityEngine
)

from ai_agent import (
    AIAnalysisAgent,
    NarrativeGenerator,
    VisualizationRecommender,
    AdvancedForecastEngine
)

from llm_integration import (
    LLMAnalyticEngine,
    RAGDocumentSystem,
    AgenticAnalysisOrchestrator
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Advanced Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'agentic_orchestrator' not in st.session_state:
    st.session_state.agentic_orchestrator = None

# ============================================================================
# PROFESSIONAL STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.05));
        border: 1px solid rgba(79, 172, 254, 0.2);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
    }
    
    .metric-card {
        background: rgba(45, 49, 66, 0.8);
        border: 1px solid rgba(79, 172, 254, 0.2);
        border-radius: 6px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(79, 172, 254, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4facfe;
        font-family: 'Roboto Mono', monospace;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    
    .insight-card {
        background: rgba(45, 49, 66, 0.6);
        border-left: 3px solid #4facfe;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .insight-card h4 {
        color: #ffffff;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .insight-card p {
        color: #cbd5e1;
        font-size: 0.875rem;
        line-height: 1.5;
    }
    
    .recommendation-card {
        background: rgba(79, 172, 254, 0.05);
        border-left: 3px solid #10b981;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .analysis-section {
        background: rgba(45, 49, 66, 0.4);
        border: 1px solid rgba(79, 172, 254, 0.1);
        border-radius: 6px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .analysis-section h3 {
        color: #ffffff;
        font-size: 1.25rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Advanced Analytics Platform</h1>
        <p>Enterprise-grade analytics with integrated artificial intelligence, autonomous analysis, and natural language understanding</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Data Management")
        
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel (XLSX, XLS)"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.session_state.analysis_complete = False
                st.success(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### Analysis Control")
            
            if st.button("Run Complete Analysis", type="primary", use_container_width=True):
                st.session_state.analysis_complete = False
            
            st.markdown("---")
            st.markdown("### System Information")
            st.caption(f"**Status:** {'Analysis Complete' if st.session_state.analysis_complete else 'Ready'}")
            st.caption(f"**Version:** 5.0 Enterprise")
    
    # Main content area
    if st.session_state.df is None:
        # Welcome screen
        st.info("Upload a dataset to begin comprehensive analytics")
        
        st.markdown("### Platform Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Autonomous Analysis**
            - Automatic pattern discovery
            - Statistical significance testing
            - Anomaly and outlier detection
            - Trend identification
            - Correlation analysis
            """)
        
        with col2:
            st.markdown("""
            **Predictive Analytics**
            - Multi-model forecasting
            - Confidence interval estimation
            - Trend decomposition
            - Seasonality detection
            - Risk assessment
            """)
        
        with col3:
            st.markdown("""
            **Intelligent Insights**
            - Natural language summaries
            - Context-aware recommendations
            - Visual analytics suggestions
            - Quality assessment
            - Business intelligence extraction
            """)
        
        st.markdown("---")
        
        st.markdown("### Technology Stack")
        st.markdown("""
        This platform leverages advanced machine learning algorithms, statistical methods, 
        and large language models to provide enterprise-grade analytics capabilities comparable 
        to leading business intelligence platforms.
        """)
        
        return
    
    df = st.session_state.df
    
    # Execute analysis pipeline
    if not st.session_state.analysis_complete:
        with st.spinner("Executing comprehensive analysis pipeline..."):
            try:
                # Phase 1: Semantic profiling
                profiler = DatasetProfiler(df)
                profiles = profiler.profile()
                
                classifier = SemanticClassifier(profiles)
                semantic_model = classifier.classify()
                
                # Phase 2: Metric intelligence
                metric_engine = MetricIntelligenceEngine(df, semantic_model)
                kpi_candidates = metric_engine.discover_kpis()
                
                # Phase 3: Statistical analysis
                ai_agent = AIAnalysisAgent(df, semantic_model)
                analysis_plan = ai_agent.create_analysis_plan()
                analysis_results = ai_agent.execute_analysis()
                
                # Phase 4: RAG system initialization
                rag_system = RAGDocumentSystem(df, semantic_model, analysis_results)
                rag_system.index_analysis_results()
                
                # Phase 5: Agentic orchestrator
                orchestrator = AgenticAnalysisOrchestrator(
                    df, semantic_model, analysis_results, rag_system
                )
                
                # Phase 6: LLM-powered insights
                llm_engine = LLMAnalyticEngine(df, semantic_model, analysis_results)
                executive_summary = llm_engine.generate_comprehensive_summary()
                recommendations = llm_engine.generate_strategic_recommendations()
                
                # Phase 7: Visualization intelligence
                viz_recommender = VisualizationRecommender(df, semantic_model)
                viz_recommendations = viz_recommender.recommend_charts()
                
                # Store results
                st.session_state.semantic_model = semantic_model
                st.session_state.kpi_candidates = kpi_candidates
                st.session_state.analysis_results = analysis_results
                st.session_state.executive_summary = executive_summary
                st.session_state.recommendations = recommendations
                st.session_state.viz_recommendations = viz_recommendations
                st.session_state.rag_system = rag_system
                st.session_state.agentic_orchestrator = orchestrator
                st.session_state.analysis_complete = True
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                return
    
    # Retrieve analysis results
    semantic_model = st.session_state.semantic_model
    kpi_candidates = st.session_state.kpi_candidates
    analysis_results = st.session_state.analysis_results
    executive_summary = st.session_state.executive_summary
    recommendations = st.session_state.recommendations
    viz_recommendations = st.session_state.viz_recommendations
    rag_system = st.session_state.rag_system
    orchestrator = st.session_state.agentic_orchestrator
    
    # Navigation tabs
    tabs = st.tabs([
        "Executive Summary",
        "Statistical Analysis",
        "Predictive Modeling",
        "Visual Analytics",
        "Natural Language Query"
    ])
    
    # ========================================================================
    # TAB 1: EXECUTIVE SUMMARY
    # ========================================================================
    with tabs[0]:
        st.markdown("### Executive Summary")
        
        # Key metrics dashboard
        quality = analysis_results.get('quality', {})
        correlations = analysis_results.get('correlations', [])
        outliers = analysis_results.get('outliers', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Total Records</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            completeness = quality.get('completeness', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{completeness:.1f}%</div>
                <div class="metric-label">Data Quality</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(correlations)}</div>
                <div class="metric-label">Correlations Found</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(kpi_candidates)}</div>
                <div class="metric-label">Key Metrics</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # AI-generated summary
        st.markdown("### Analytical Findings")
        st.markdown(executive_summary)
        
        st.markdown("---")
        
        # Strategic recommendations
        st.markdown("### Strategic Recommendations")
        for i, rec in enumerate(recommendations[:5], 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>Recommendation {i}</strong>
                <p>{rec}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: STATISTICAL ANALYSIS
    # ========================================================================
    with tabs[1]:
        st.markdown("### Comprehensive Statistical Analysis")
        
        # Data quality assessment
        with st.expander("Data Quality Assessment", expanded=True):
            quality = analysis_results.get('quality', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Completeness", f"{quality.get('completeness', 0):.2f}%")
                st.metric("Total Records", f"{quality.get('total_rows', 0):,}")
                st.metric("Total Attributes", quality.get('total_columns', 0))
            
            with col2:
                st.metric("Missing Values", f"{quality.get('missing_cells', 0):,}")
                st.metric("Duplicate Records", quality.get('duplicate_rows', 0))
                
                if quality.get('issues'):
                    st.warning(f"{len(quality['issues'])} quality issues detected")
                else:
                    st.success("Data quality is excellent")
        
        # Correlation analysis
        correlations = analysis_results.get('correlations', [])
        if correlations:
            st.markdown("### Correlation Analysis")
            
            for corr in correlations[:5]:
                strength_class = "success" if corr['strength'] == 'strong' else "warning"
                st.markdown(f"""
                <div class="insight-card">
                    <h4><span class="status-badge status-{strength_class}">{corr['strength'].upper()}</span> 
                    {corr['direction'].capitalize()} Correlation</h4>
                    <p><strong>{corr['var1']}</strong> and <strong>{corr['var2']}</strong> 
                    show correlation coefficient of {corr['correlation']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Distribution analysis
        distributions = analysis_results.get('distributions', {})
        if distributions:
            st.markdown("### Distribution Analysis")
            
            for col_name, data in list(distributions.items())[:3]:
                st.markdown(f"""
                <div class="insight-card">
                    <h4>{col_name}</h4>
                    <p><strong>Shape:</strong> {data['shape']}<br>
                    <strong>Skewness:</strong> {data['skewness']:.3f} | 
                    <strong>Kurtosis:</strong> {data['kurtosis']:.3f}<br>
                    <strong>Normality:</strong> {'Normally distributed' if data['is_normal'] else 'Non-normal distribution'}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Outlier detection
        outliers = analysis_results.get('outliers', {})
        if outliers:
            st.markdown("### Outlier Detection Results")
            
            outlier_df = pd.DataFrame([
                {
                    'Attribute': col,
                    'Outlier Count': data['count'],
                    'Percentage': f"{data['percentage']:.2f}%",
                    'Lower Bound': f"{data['lower_bound']:.2f}",
                    'Upper Bound': f"{data['upper_bound']:.2f}"
                }
                for col, data in outliers.items()
            ])
            
            st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        
        # Cluster analysis
        clusters = analysis_results.get('clusters', {})
        if clusters:
            st.markdown("### Segmentation Analysis")
            
            st.info(f"Identified {clusters['n_clusters']} distinct segments using K-means clustering")
            
            cluster_data = []
            for name, stats in clusters['cluster_stats'].items():
                cluster_data.append({
                    'Segment': name,
                    'Size': stats['size'],
                    'Percentage': f"{stats['percentage']:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(cluster_data), use_container_width=True, hide_index=True)
    
    # ========================================================================
    # TAB 3: PREDICTIVE MODELING
    # ========================================================================
    with tabs[2]:
        st.markdown("### Predictive Analytics & Forecasting")
        
        if not semantic_model.time_columns:
            st.warning("Temporal data not detected. Forecasting requires time-series data.")
        else:
            time_col = semantic_model.time_columns[0]
            
            # Ensure datetime format
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                except:
                    st.error("Unable to parse time column as datetime")
                    return
            
            forecast_candidates = (semantic_model.measures + 
                                 semantic_model.currencies + 
                                 semantic_model.counts)
            
            if not forecast_candidates:
                st.info("No quantitative metrics available for forecasting")
            else:
                st.markdown("#### Select Target Metric")
                selected_metric = st.selectbox(
                    "Metric for forecasting",
                    forecast_candidates,
                    label_visibility="collapsed"
                )
                
                # Eligibility assessment
                st.markdown("#### Forecast Eligibility Assessment")
                
                eligibility_engine = ForecastEligibilityEngine(df, time_col, selected_metric)
                eligibility = eligibility_engine.check()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    status_class = "success" if eligibility.is_eligible else "error"
                    status_text = "ELIGIBLE" if eligibility.is_eligible else "NOT ELIGIBLE"
                    st.markdown(f'<span class="status-badge status-{status_class}">{status_text}</span>', 
                              unsafe_allow_html=True)
                
                with col2:
                    st.metric("Eligibility Score", f"{eligibility.score:.0f}/100")
                
                with col3:
                    st.metric("Quality Warnings", len(eligibility.warnings))
                
                st.info(eligibility.reason)
                
                if eligibility.warnings:
                    with st.expander("View Quality Warnings"):
                        for warning in eligibility.warnings:
                            st.warning(warning)
                
                if eligibility.is_eligible:
                    st.markdown("---")
                    st.markdown("#### Forecast Configuration")
                    
                    forecast_periods = st.slider(
                        "Forecast horizon (periods)",
                        min_value=7,
                        max_value=90,
                        value=30,
                        step=7
                    )
                    
                    if st.button("Generate Forecast", type="primary"):
                        with st.spinner("Running predictive models..."):
                            try:
                                forecast_engine = AdvancedForecastEngine(df, time_col, selected_metric)
                                forecast_df = forecast_engine.forecast_prophet(periods=forecast_periods)
                                forecast_metrics = forecast_engine.get_forecast_metrics(forecast_df)
                                
                                # Visualization
                                fig = go.Figure()
                                
                                # Historical data
                                historical = df[[time_col, selected_metric]].dropna()
                                fig.add_trace(go.Scatter(
                                    x=historical[time_col],
                                    y=historical[selected_metric],
                                    mode='lines+markers',
                                    name='Historical Data',
                                    line=dict(color='#4facfe', width=2),
                                    marker=dict(size=4)
                                ))
                                
                                # Forecast
                                forecast_only = forecast_df.iloc[-forecast_periods:]
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
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=forecast_only['ds'],
                                    y=forecast_only['yhat_lower'],
                                    mode='lines',
                                    line=dict(width=0),
                                    fillcolor='rgba(79, 172, 254, 0.2)',
                                    fill='tonexty',
                                    name='95% Confidence Interval',
                                    hoverinfo='skip'
                                ))
                                
                                fig.update_layout(
                                    title=f'Forecast: {selected_metric}',
                                    xaxis_title='Time Period',
                                    yaxis_title=selected_metric,
                                    template='plotly_dark',
                                    height=500,
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast summary
                                st.markdown("#### Forecast Summary")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Mean Forecast Value", f"{forecast_metrics['mean_forecast']:.2f}")
                                
                                with col2:
                                    trend_direction = forecast_metrics['forecast_trend']
                                    st.metric("Trend Direction", trend_direction.capitalize())
                                
                                with col3:
                                    st.metric("Volatility (Std Dev)", f"{forecast_metrics['volatility']:.2f}")
                                
                                # Export option
                                st.markdown("#### Export Forecast")
                                csv_export = forecast_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
                                st.download_button(
                                    "Download Forecast Data (CSV)",
                                    csv_export,
                                    f"forecast_{selected_metric}.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                                
                            except Exception as e:
                                st.error(f"Forecasting error: {str(e)}")
                else:
                    st.info("This metric does not meet the minimum requirements for reliable forecasting. Consider collecting more data or addressing quality issues.")
    
    # ========================================================================
    # TAB 4: VISUAL ANALYTICS
    # ========================================================================
    with tabs[3]:
        st.markdown("### Recommended Visual Analytics")
        
        st.info("Based on data structure and relationships, the following visualizations are recommended")
        
        # Filter high-priority visualizations
        high_priority_viz = [v for v in viz_recommendations if v['priority'] == 'high']
        
        if high_priority_viz:
            st.markdown("#### Priority Visualizations")
            
            for viz in high_priority_viz[:4]:
                with st.expander(f"{viz['title']}", expanded=True):
                    st.caption(f"Rationale: {viz['reason']}")
                    
                    try:
                        if viz['type'] == 'line':
                            fig = px.line(
                                df, 
                                x=viz['x'], 
                                y=viz['y'],
                                title=viz['title'],
                                template='plotly_dark'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz['type'] == 'bar':
                            agg_df = df.groupby(viz['x'])[viz['y']].mean().reset_index()
                            fig = px.bar(
                                agg_df,
                                x=viz['x'],
                                y=viz['y'],
                                title=viz['title'],
                                template='plotly_dark'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz['type'] == 'heatmap':
                            numeric_cols = df[viz['data']].select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) >= 2:
                                corr = df[numeric_cols].corr()
                                fig = px.imshow(
                                    corr,
                                    text_auto='.2f',
                                    title=viz['title'],
                                    template='plotly_dark',
                                    color_continuous_scale='RdBu_r'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz['type'] == 'scatter':
                            fig = px.scatter(
                                df,
                                x=viz['x'],
                                y=viz['y'],
                                title=viz['title'],
                                template='plotly_dark',
                                trendline='ols'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz['type'] == 'histogram':
                            fig = px.histogram(
                                df,
                                x=viz['x'],
                                title=viz['title'],
                                template='plotly_dark'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz['type'] == 'box':
                            fig = px.box(
                                df,
                                y=viz['y'],
                                title=viz['title'],
                                template='plotly_dark'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Visualization error: {str(e)}")
        
        # Additional visualizations
        medium_priority_viz = [v for v in viz_recommendations if v['priority'] == 'medium']
        if medium_priority_viz:
            st.markdown("#### Additional Visualizations")
            
            for viz in medium_priority_viz[:2]:
                with st.expander(f"{viz['title']}"):
                    st.caption(f"Rationale: {viz['reason']}")
                    
                    # Similar visualization logic as above
                    try:
                        if viz['type'] == 'histogram':
                            fig = px.histogram(df, x=viz['x'], title=viz['title'], template='plotly_dark')
                            st.plotly_chart(fig, use_container_width=True)
                        elif viz['type'] == 'scatter':
                            fig = px.scatter(df, x=viz['x'], y=viz['y'], title=viz['title'], 
                                           template='plotly_dark', trendline='ols')
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Visualization error: {str(e)}")
    
    # ========================================================================
    # TAB 5: NATURAL LANGUAGE QUERY
    # ========================================================================
    with tabs[4]:
        st.markdown("### Natural Language Analytics Interface")
        
        st.info("Query your data using natural language. The system will retrieve relevant analysis and provide context-aware responses.")
        
        # Quick query buttons
        st.markdown("#### Suggested Queries")
        
        suggested_queries = [
            "Summarize the key findings from the analysis",
            "What are the most significant correlations?",
            "Identify any concerning trends in the data",
            "What metrics should we prioritize?",
            "Describe the data quality issues",
            "What patterns were discovered?"
        ]
        
        cols = st.columns(3)
        for i, query in enumerate(suggested_queries):
            with cols[i % 3]:
                if st.button(query, key=f"query_{i}", use_container_width=True):
                    st.session_state.current_query = query
        
        # Custom query input
        st.markdown("#### Custom Query")
        user_query = st.text_input(
            "Enter your question",
            placeholder="e.g., What is driving the increase in sales?",
            label_visibility="collapsed"
        )
        
        # Process query
        query_to_process = user_query if user_query else st.session_state.get('current_query', '')
        
        if query_to_process:
            st.markdown("---")
            st.markdown("#### Analysis Response")
            
            with st.spinner("Processing query and retrieving relevant analysis..."):
                try:
                    # Use agentic orchestrator to process query
                    response = orchestrator.process_natural_language_query(query_to_process)
                    
                    st.markdown(response)
                    
                    # Show relevant context from RAG
                    if rag_system:
                        relevant_context = rag_system.retrieve_relevant_context(query_to_process)
                        
                        if relevant_context:
                            with st.expander("Related Analysis Context"):
                                for context in relevant_context[:3]:
                                    st.markdown(f"- {context}")
                
                except Exception as e:
                    st.error(f"Query processing error: {str(e)}")
                    
                    # Fallback to pattern-based responses
                    if "summary" in query_to_process.lower() or "findings" in query_to_process.lower():
                        st.markdown(executive_summary)
                    elif "correlation" in query_to_process.lower():
                        correlations = analysis_results.get('correlations', [])
                        if correlations:
                            st.markdown("**Significant Correlations:**")
                            for corr in correlations[:5]:
                                st.markdown(f"- {corr['var1']} and {corr['var2']}: {corr['correlation']:.3f} ({corr['strength']} {corr['direction']})")
                    elif "trend" in query_to_process.lower():
                        time_series = analysis_results.get('time_series', {})
                        if time_series:
                            st.markdown("**Trend Analysis:**")
                            for metric, data in time_series.items():
                                st.markdown(f"- {metric}: {data['direction']} trend with {data['trend_strength']:.1%} strength")
                    elif "quality" in query_to_process.lower():
                        quality = analysis_results.get('quality', {})
                        st.markdown(f"**Data Quality Assessment:**")
                        st.markdown(f"- Completeness: {quality.get('completeness', 0):.2f}%")
                        st.markdown(f"- Missing values: {quality.get('missing_cells', 0):,}")
                        st.markdown(f"- Duplicate records: {quality.get('duplicate_rows', 0)}")
                    else:
                        st.markdown("Analysis complete. Please refine your query for more specific insights.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem 0;">
        <p><strong>Advanced Analytics Platform v5.0</strong> | Enterprise Edition</p>
        <p style="font-size: 0.875rem;">Autonomous Intelligence • Statistical Analysis • Predictive Modeling • Natural Language Understanding</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
