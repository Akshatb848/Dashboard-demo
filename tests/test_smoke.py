"""Smoke tests: the analysis pipeline and the Streamlit app run end to end
on the bundled synthetic datasets (generate_test_data.py)."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import generate_test_data as gen
from intelligent_analyst import IntelligentDataAnalyst
from rag_query_engine import IntelligentQueryEngine

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "sales": gen.generate_sales_data,
    "financial": gen.generate_financial_data,
    "hr": gen.generate_hr_data,
    "ecommerce": gen.generate_ecommerce_data,
    "dirty": gen.generate_dirty_data,
}


def run_pipeline(df: pd.DataFrame) -> tuple[IntelligentDataAnalyst, dict]:
    analyst = IntelligentDataAnalyst(df)
    analyst.create_analysis_plan()
    results = {
        "data_understanding": analyst.execute_data_understanding(),
        "quality_assessment": analyst.execute_quality_assessment(),
        "feature_analysis": analyst.execute_feature_analysis(),
        "statistical_testing": analyst.execute_statistical_testing(),
        "pattern_discovery": analyst.execute_pattern_discovery(),
        "preprocessing_recommendations": analyst.execute_preprocessing_recommendations(),
        "visualization_strategy": analyst.execute_visualization_strategy(),
    }
    return analyst, results


@pytest.mark.parametrize("name", DATASETS)
def test_pipeline_runs_on_synthetic_data(name):
    df = DATASETS[name]()
    analyst, results = run_pipeline(df)
    assert results["data_understanding"]["shape"]["rows"] == len(df)
    assert 0 <= results["quality_assessment"]["completeness"]["score"] <= 100
    report = analyst.generate_comprehensive_report()
    assert "Dataset Overview" in report


def test_dirty_data_reports_missing_values():
    _, results = run_pipeline(gen.generate_dirty_data())
    assert results["quality_assessment"]["completeness"]["score"] < 100


@pytest.mark.parametrize(
    "question",
    ["Summarize the dataset", "What correlations exist?", "Are there quality issues?",
     "What should I do to preprocess?", "Show me the trends", "Are there outliers?"],
)
def test_query_engine_answers_quick_questions(question):
    df = gen.generate_sales_data()
    _, results = run_pipeline(df)
    answer = IntelligentQueryEngine(df, results).answer_query(question)
    assert isinstance(answer, str) and answer.strip()


def test_app_renders_all_tabs_after_analysis():
    at = AppTest.from_file(str(ROOT / "app_production.py"), default_timeout=120)
    at.run()
    assert not at.exception
    at.session_state.df = gen.generate_sales_data()
    at.session_state.run_analysis = True
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state.analysis_complete
    assert len(at.tabs) == 6
