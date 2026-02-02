"""
Intelligent Data Analytics Platform
Production-grade analytics with Agentic AI, RAG, and LLM integration
No icons, fully professional, enterprise-ready

Author: Akshat Banga
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import traceback
from datetime import datetime

# Import intelligent modules
from intelligent_analyst import IntelligentDataAnalyst
from rag_query_engine import IntelligentQueryEngine

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Intelligent Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# Session state
def init_state():
    defaults = {
        'df': None,
        'analyst': None,
        'query_engine': None,
        'analysis_results': {},
        'analysis_complete': False,
        'run_analysis': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(59, 130, 246, 0.05));
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    h1 {
        color: #ffffff;
        font-weight: 700;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(37, 99, 235, 0.3);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
    }
    
    .insight-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Intelligent Data Analytics Platform</h1>
        <p style="color: #94a3b8;">Powered by Agentic AI, RAG, and Advanced Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Data Upload")
        
        uploaded_file = st.file_uploader(
            "Select CSV file",
            type=['csv'],
            help="Upload your dataset for intelligent analysis"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.analysis_complete = False
                st.success(f"Loaded: {len(df):,} rows × {len(df.columns)} cols")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### Analysis Control")
            
            if st.button("Run Intelligent Analysis", type="primary", use_container_width=True):
                st.session_state.run_analysis = True
                st.rerun()
            
            st.markdown("---")
            st.markdown("### Status")
            if st.session_state.analysis_complete:
                st.success("Analysis Complete")
            else:
                st.info("Ready to Analyze")
    
    # Main area
    if st.session_state.df is None:
        st.info("Upload a CSV file to begin intelligent analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Autonomous Analysis**
            - Intelligent EDA
            - Quality assessment
            - Pattern discovery
            - Statistical testing
            """)
        with col2:
            st.markdown("""
            **Smart Recommendations**
            - Preprocessing steps
            - Feature engineering
            - Visualization strategy
            - Action items
            """)
        with col3:
            st.markdown("""
            **Natural Language**
            - Ask questions
            - Get insights
            - RAG-powered answers
            - Context-aware responses
            """)
        return
    
    df = st.session_state.df
    
    # Run analysis
    if st.session_state.run_analysis and not st.session_state.analysis_complete:
        progress = st.progress(0)
        status = st.empty()
        
        try:
            # Create analyst
            status.text("Initializing intelligent analyst...")
            progress.progress(5)
            
            analyst = IntelligentDataAnalyst(df)
            
            # Create analysis plan
            status.text("Phase 1/7: Creating analysis plan...")
            analyst.create_analysis_plan()
            progress.progress(15)
            
            # Execute analyses
            results = {}
            
            status.text("Phase 2/7: Data understanding...")
            results['data_understanding'] = analyst.execute_data_understanding()
            progress.progress(30)
            
            status.text("Phase 3/7: Quality assessment...")
            results['quality_assessment'] = analyst.execute_quality_assessment()
            progress.progress(45)
            
            status.text("Phase 4/7: Feature analysis...")
            results['feature_analysis'] = analyst.execute_feature_analysis()
            progress.progress(60)
            
            status.text("Phase 5/7: Statistical testing...")
            results['statistical_testing'] = analyst.execute_statistical_testing()
            progress.progress(75)
            
            status.text("Phase 6/7: Pattern discovery...")
            results['pattern_discovery'] = analyst.execute_pattern_discovery()
            progress.progress(85)
            
            status.text("Phase 7/7: Generating recommendations...")
            results['preprocessing_recommendations'] = analyst.execute_preprocessing_recommendations()
            results['visualization_strategy'] = analyst.execute_visualization_strategy()
            progress.progress(95)
            
            # Build RAG system
            status.text("Building knowledge base...")
            query_engine = IntelligentQueryEngine(df, results)
            progress.progress(100)
            
            # Store results
            st.session_state.analyst = analyst
            st.session_state.query_engine = query_engine
            st.session_state.analysis_results = results
            st.session_state.analysis_complete = True
            st.session_state.run_analysis = False
            
            progress.empty()
            status.empty()
            st.success("Analysis completed successfully!")
            st.rerun()
            
        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f"Analysis failed: {str(e)}")
            st.code(traceback.format_exc())
            st.session_state.run_analysis = False
            return
    
    # Display results
    if not st.session_state.analysis_complete:
        st.info("Click 'Run Intelligent Analysis' to begin")
        
        st.markdown("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            numeric = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric", numeric)
        
        return
    
    # Tabs
    tabs = st.tabs([
        "Executive Summary",
        "Detailed Analysis",
        "Data Quality",
        "Preprocessing Guide",
        "Visualizations",
        "Ask the Analyst"
    ])
    
    results = st.session_state.analysis_results
    query_engine = st.session_state.query_engine
    
    # TAB 1: Executive Summary
    with tabs[0]:
        st.markdown("### Executive Summary")
        
        understanding = results['data_understanding']
        quality = results['quality_assessment']
        features = results['feature_analysis']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", f"{understanding['shape']['rows']:,}")
        with col2:
            st.metric("Features", understanding['shape']['columns'])
        with col3:
            st.metric("Quality", f"{quality['completeness']['score']:.1f}%")
        with col4:
            st.metric("Correlations", len(features.get('correlations', [])))
        
        st.markdown("---")
        
        # Generate and display comprehensive report
        report = st.session_state.analyst.generate_comprehensive_report()
        st.markdown(report)
    
    # TAB 2: Detailed Analysis
    with tabs[1]:
        st.markdown("### Feature Analysis")
        
        features = results['feature_analysis']
        
        # Numeric features
        if features['numeric_features']:
            st.markdown("#### Numeric Feature Distributions")
            for col, info in list(features['numeric_features'].items())[:5]:
                st.markdown(f"""
                <div class="insight-box">
                    <h4>{col}</h4>
                    <p>Distribution: {info['distribution_type']}<br>
                    Skewness: {info['skewness']:.2f} | Kurtosis: {info['kurtosis']:.2f}<br>
                    Normally distributed: {'Yes' if info['is_normal'] else 'No'}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Correlations
        if features.get('correlations'):
            st.markdown("#### Significant Correlations")
            for corr in features['correlations'][:5]:
                st.markdown(f"""
                - **{corr['feature1']}** ↔ **{corr['feature2']}**: 
                {corr['strength']} {corr['direction']} ({corr['correlation']:.3f})
                """)
    
    # TAB 3: Data Quality
    with tabs[2]:
        st.markdown("### Data Quality Report")
        
        quality = results['quality_assessment']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Completeness", f"{quality['completeness']['score']:.1f}%")
        with col2:
            st.metric("Missing Cells", f"{quality['completeness']['missing_cells']:,}")
        with col3:
            st.metric("Duplicates", quality['consistency']['duplicate_rows'])
        
        if quality['issues']:
            st.markdown("#### Identified Issues")
            for issue in quality['issues']:
                severity_color = {
                    'high': '#ef4444',
                    'medium': '#f59e0b',
                    'low': '#3b82f6'
                }.get(issue['severity'], '#6b7280')
                
                st.markdown(f"""
                <div class="insight-box" style="border-left-color: {severity_color};">
                    <strong>{issue['severity'].upper()}</strong>: {issue['issue']}<br>
                    <em>Recommendation: {issue['recommendation']}</em>
                </div>
                """, unsafe_allow_html=True)
    
    # TAB 4: Preprocessing Guide
    with tabs[3]:
        st.markdown("### Recommended Preprocessing Steps")
        
        recommendations = results['preprocessing_recommendations']
        
        high = [r for r in recommendations if r['priority'] == 'high']
        medium = [r for r in recommendations if r['priority'] == 'medium']
        
        if high:
            st.markdown("#### High Priority")
            for i, rec in enumerate(high, 1):
                st.markdown(f"""
                {i}. **{rec['step'].replace('_', ' ').title()}**
                - Column: {rec.get('column', 'Multiple')}
                - Method: {rec.get('method', 'N/A')}
                - Reason: {rec['reason']}
                """)
        
        if medium:
            with st.expander("Medium Priority Actions"):
                for i, rec in enumerate(medium, 1):
                    st.markdown(f"""
                    {i}. **{rec['step'].replace('_', ' ').title()}**
                    - {rec.get('column', 'Multiple')}: {rec['reason']}
                    """)
    
    # TAB 5: Visualizations
    with tabs[4]:
        st.markdown("### Intelligent Visualizations")
        
        viz_strategy = results['visualization_strategy']
        high_priority = [v for v in viz_strategy if v['priority'] == 'high']
        
        for viz in high_priority[:4]:
            with st.expander(viz['title'], expanded=True):
                st.caption(f"Purpose: {viz['purpose']}")
                
                try:
                    if viz['type'] == 'histogram':
                        fig = px.histogram(df, x=viz['x_axis'], template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz['type'] == 'boxplot':
                        fig = px.box(df, y=viz['y_axis'], template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz['type'] == 'heatmap':
                        numeric_cols = df[viz['data']].select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            corr = df[numeric_cols].corr()
                            fig = px.imshow(corr, text_auto='.2f', template='plotly_dark')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz['type'] == 'scatter':
                        fig = px.scatter(df, x=viz['x_axis'], y=viz['y_axis'], 
                                       template='plotly_dark', trendline='ols')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz['type'] == 'bar':
                        fig = px.bar(df[viz['x_axis']].value_counts().head(10), 
                                   template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.warning(f"Could not generate: {str(e)}")
    
    # TAB 6: Ask the Analyst
    with tabs[5]:
        st.markdown("### Ask the Intelligent Analyst")
        
        st.info("Ask questions about your data in natural language")
        
        # Quick questions
        st.markdown("#### Quick Questions")
        quick_q = [
            "Summarize the dataset",
            "What correlations exist?",
            "Are there quality issues?",
            "What should I do to preprocess?",
            "Show me the trends",
            "Are there outliers?"
        ]
        
        cols = st.columns(3)
        for i, q in enumerate(quick_q):
            with cols[i % 3]:
                if st.button(q, key=f"q{i}", use_container_width=True):
                    st.session_state.current_query = q
        
        # Custom query
        user_query = st.text_input(
            "Or ask your own question:",
            placeholder="E.g., What preprocessing steps do you recommend?"
        )
        
        query = user_query if user_query else st.session_state.get('current_query', '')
        
        if query:
            st.markdown("---")
            st.markdown("#### Answer")
            
            try:
                answer = query_engine.answer_query(query)
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p>Intelligent Data Analytics Platform | Powered by Agentic AI & RAG</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
