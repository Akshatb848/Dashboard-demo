"""
Advanced Analytics Platform v5.0 - Production Ready
BULLETPROOF VERSION with comprehensive error handling and debugging
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import traceback
import warnings
warnings.filterwarnings('ignore')

# Import with error handling
try:
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
    IMPORTS_SUCCESS = True
except Exception as e:
    IMPORTS_SUCCESS = False
    IMPORT_ERROR = str(e)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Advanced Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with defaults
def init_session_state():
    defaults = {
        'df': None,
        'analysis_complete': False,
        'run_analysis': False,
        'semantic_model': None,
        'kpi_candidates': [],
        'analysis_results': {},
        'executive_summary': '',
        'recommendations': [],
        'viz_recommendations': [],
        'rag_system': None,
        'agentic_orchestrator': None,
        'error_log': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.05));
        border: 1px solid rgba(79, 172, 254, 0.2);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(45, 49, 66, 0.8);
        border: 1px solid rgba(79, 172, 254, 0.2);
        border-radius: 6px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4facfe;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    .insight-card {
        background: rgba(45, 49, 66, 0.6);
        border-left: 3px solid #4facfe;
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
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Check imports
    if not IMPORTS_SUCCESS:
        st.error(f"Import Error: {IMPORT_ERROR}")
        st.stop()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Advanced Analytics Platform</h1>
        <p>Enterprise-grade analytics with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Data Management")
        
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=['csv', 'xlsx', 'xls'],
            help="CSV or Excel files"
        )
        
        if uploaded_file:
            try:
                with st.spinner("Loading file..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.df = df
                    st.session_state.analysis_complete = False
                    st.success(f"✓ Loaded: {len(df):,} rows × {len(df.columns)} columns")
                    
            except Exception as e:
                st.error(f"File load error: {str(e)}")
                st.code(traceback.format_exc())
        
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### Analysis Control")
            
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                st.session_state.run_analysis = True
                st.session_state.analysis_complete = False
                st.rerun()
            
            st.markdown("---")
            st.markdown("### Status")
            
            if st.session_state.analysis_complete:
                st.success("Analysis Complete")
            elif st.session_state.run_analysis:
                st.warning("Running...")
            else:
                st.info("Ready to analyze")
            
            st.caption(f"Rows: {len(st.session_state.df):,}")
            st.caption(f"Columns: {len(st.session_state.df.columns)}")
    
    # Main content
    if st.session_state.df is None:
        st.info("👆 Upload a dataset to begin")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Analysis**\n- Pattern discovery\n- Statistical testing\n- Outlier detection")
        with col2:
            st.markdown("**Forecasting**\n- Multi-model prediction\n- Confidence intervals\n- Trend analysis")
        with col3:
            st.markdown("**Insights**\n- Natural language summaries\n- Recommendations\n- Smart visualizations")
        
        return
    
    df = st.session_state.df
    
    # Run analysis when triggered
    if st.session_state.run_analysis and not st.session_state.analysis_complete:
        
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.text("Starting analysis...")
            progress.progress(5)
            
            # Phase 1: Profiling
            status.text("Phase 1/7: Data profiling...")
            profiler = DatasetProfiler(df)
            profiles = profiler.profile()
            classifier = SemanticClassifier(profiles)
            semantic_model = classifier.classify()
            progress.progress(15)
            
            # Phase 2: Metrics
            status.text("Phase 2/7: Metric discovery...")
            metric_engine = MetricIntelligenceEngine(df, semantic_model)
            kpi_candidates = metric_engine.discover_kpis()
            progress.progress(30)
            
            # Phase 3: Statistical
            status.text("Phase 3/7: Statistical analysis...")
            ai_agent = AIAnalysisAgent(df, semantic_model)
            analysis_plan = ai_agent.create_analysis_plan()
            analysis_results = ai_agent.execute_analysis()
            progress.progress(50)
            
            # Phase 4: RAG
            status.text("Phase 4/7: Knowledge indexing...")
            rag_system = RAGDocumentSystem(df, semantic_model, analysis_results)
            rag_system.index_analysis_results()
            progress.progress(65)
            
            # Phase 5: Orchestrator
            status.text("Phase 5/7: AI orchestration...")
            orchestrator = AgenticAnalysisOrchestrator(df, semantic_model, analysis_results, rag_system)
            progress.progress(75)
            
            # Phase 6: LLM
            status.text("Phase 6/7: Generating insights...")
            llm_engine = LLMAnalyticEngine(df, semantic_model, analysis_results)
            executive_summary = llm_engine.generate_comprehensive_summary()
            recommendations = llm_engine.generate_strategic_recommendations()
            progress.progress(90)
            
            # Phase 7: Viz
            status.text("Phase 7/7: Visualizations...")
            viz_recommender = VisualizationRecommender(df, semantic_model)
            viz_recommendations = viz_recommender.recommend_charts()
            progress.progress(100)
            
            # Save to session state
            st.session_state.semantic_model = semantic_model
            st.session_state.kpi_candidates = kpi_candidates
            st.session_state.analysis_results = analysis_results
            st.session_state.executive_summary = executive_summary
            st.session_state.recommendations = recommendations
            st.session_state.viz_recommendations = viz_recommendations
            st.session_state.rag_system = rag_system
            st.session_state.agentic_orchestrator = orchestrator
            st.session_state.analysis_complete = True
            st.session_state.run_analysis = False
            
            status.text("✓ Complete!")
            progress.empty()
            status.empty()
            
            st.success("Analysis completed successfully!")
            st.balloons()
            st.rerun()
            
        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f"Analysis failed: {str(e)}")
            st.code(traceback.format_exc())
            st.session_state.run_analysis = False
            st.session_state.error_log.append(str(e))
            return
    
    # Show results
    if not st.session_state.analysis_complete:
        st.info("👈 Click 'Run Analysis' to start")
        
        # Show preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### Quick Stats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        
        return
    
    # Display analysis results
    st.success("✓ Analysis Complete - Explore tabs below")
    
    tabs = st.tabs(["Executive Summary", "Statistical Analysis", "Forecasting", "Visualizations", "Query"])
    
    # Get results
    analysis_results = st.session_state.analysis_results
    executive_summary = st.session_state.executive_summary
    recommendations = st.session_state.recommendations
    viz_recommendations = st.session_state.viz_recommendations
    semantic_model = st.session_state.semantic_model
    
    # TAB 1: Executive Summary
    with tabs[0]:
        st.markdown("### Executive Summary")
        
        # Metrics
        quality = analysis_results.get('quality', {})
        correlations = analysis_results.get('correlations', [])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Data Quality", f"{quality.get('completeness', 0):.1f}%")
        with col3:
            st.metric("Correlations", len(correlations))
        with col4:
            st.metric("Outlier Columns", len(analysis_results.get('outliers', {})))
        
        st.markdown("---")
        st.markdown(executive_summary)
        
        st.markdown("### Recommendations")
        for i, rec in enumerate(recommendations[:5], 1):
            st.markdown(f"{i}. {rec}")
    
    # TAB 2: Statistical Analysis
    with tabs[1]:
        st.markdown("### Statistical Analysis")
        
        # Quality
        quality = analysis_results.get('quality', {})
        st.markdown("#### Data Quality")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Completeness", f"{quality.get('completeness', 0):.1f}%")
        with col2:
            st.metric("Missing Cells", f"{quality.get('missing_cells', 0):,}")
        with col3:
            st.metric("Duplicates", quality.get('duplicate_rows', 0))
        
        # Correlations
        correlations = analysis_results.get('correlations', [])
        if correlations:
            st.markdown("#### Correlation Analysis")
            for corr in correlations[:5]:
                st.markdown(f"- **{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']:.3f} ({corr['strength']} {corr['direction']})")
        
        # Outliers
        outliers = analysis_results.get('outliers', {})
        if outliers:
            st.markdown("#### Outlier Detection")
            for col, data in outliers.items():
                st.markdown(f"- **{col}**: {data['count']} outliers ({data['percentage']:.1f}%)")
    
    # TAB 3: Forecasting
    with tabs[2]:
        st.markdown("### Predictive Modeling")
        
        if not semantic_model.time_columns:
            st.warning("No time column detected for forecasting")
        else:
            time_col = semantic_model.time_columns[0]
            metrics = semantic_model.measures + semantic_model.currencies + semantic_model.counts
            
            if not metrics:
                st.info("No numeric metrics for forecasting")
            else:
                selected_metric = st.selectbox("Select metric", metrics)
                
                # Eligibility
                from semantic_engine import ForecastEligibilityEngine
                eligibility_engine = ForecastEligibilityEngine(df, time_col, selected_metric)
                eligibility = eligibility_engine.check()
                
                st.metric("Eligibility Score", f"{eligibility.score}/100")
                st.info(eligibility.reason)
                
                if eligibility.is_eligible and st.button("Generate Forecast"):
                    try:
                        # Ensure datetime
                        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                            df[time_col] = pd.to_datetime(df[time_col])
                        
                        forecast_engine = AdvancedForecastEngine(df, time_col, selected_metric)
                        forecast_df = forecast_engine.forecast_prophet(periods=30)
                        
                        # Plot
                        fig = go.Figure()
                        historical = df[[time_col, selected_metric]].dropna()
                        fig.add_trace(go.Scatter(x=historical[time_col], y=historical[selected_metric],
                                                mode='lines', name='Historical'))
                        
                        forecast_only = forecast_df.iloc[-30:]
                        fig.add_trace(go.Scatter(x=forecast_only['ds'], y=forecast_only['yhat'],
                                                mode='lines', name='Forecast', line=dict(dash='dash')))
                        
                        fig.update_layout(title=f'Forecast: {selected_metric}', template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Forecasting error: {str(e)}")
    
    # TAB 4: Visualizations
    with tabs[3]:
        st.markdown("### Recommended Visualizations")
        
        high_priority = [v for v in viz_recommendations if v['priority'] == 'high']
        
        for viz in high_priority[:3]:
            with st.expander(viz['title'], expanded=True):
                st.caption(f"Rationale: {viz['reason']}")
                
                try:
                    if viz['type'] == 'line' and viz['x'] in df.columns and viz['y'] in df.columns:
                        fig = px.line(df, x=viz['x'], y=viz['y'], template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz['type'] == 'bar' and viz['x'] in df.columns and viz['y'] in df.columns:
                        agg = df.groupby(viz['x'])[viz['y']].mean().reset_index()
                        fig = px.bar(agg, x=viz['x'], y=viz['y'], template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz['type'] == 'heatmap':
                        numeric = df[viz['data']].select_dtypes(include=[np.number]).columns
                        if len(numeric) >= 2:
                            corr = df[numeric].corr()
                            fig = px.imshow(corr, text_auto='.2f', template='plotly_dark')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz['type'] == 'scatter' and viz['x'] in df.columns and viz['y'] in df.columns:
                        fig = px.scatter(df, x=viz['x'], y=viz['y'], template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.warning(f"Could not generate chart: {str(e)}")
    
    # TAB 5: Query
    with tabs[4]:
        st.markdown("### Natural Language Query")
        
        queries = [
            "What are the key insights?",
            "Show significant correlations",
            "Identify concerning trends",
            "What should I focus on?",
            "Describe data quality",
            "What patterns were found?"
        ]
        
        cols = st.columns(3)
        for i, q in enumerate(queries):
            with cols[i % 3]:
                if st.button(q, key=f"q_{i}"):
                    st.session_state.current_query = q
        
        user_query = st.text_input("Or ask your own:", placeholder="Your question here")
        
        query = user_query if user_query else st.session_state.get('current_query', '')
        
        if query:
            try:
                orchestrator = st.session_state.agentic_orchestrator
                response = orchestrator.process_natural_language_query(query)
                st.markdown("#### Response")
                st.markdown(response)
            except Exception as e:
                st.error(f"Query error: {str(e)}")
                st.markdown("Try rephrasing your question.")

if __name__ == "__main__":
    main()
