"""
Validation Script for AI Analytics Dashboard
Tests all features and ensures production readiness
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import components
from semantic_engine import (
    DatasetProfiler, SemanticClassifier, 
    MetricIntelligenceEngine, ForecastEligibilityEngine
)

class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def test_semantic_profiler(df):
    """Test semantic profiler engine"""
    print_header("TEST 1: Semantic Profiler Engine")
    
    try:
        profiler = DatasetProfiler(df)
        profiles = profiler.profile()
        
        assert len(profiles) == len(df.columns), "Profile count mismatch"
        print_success(f"Profiled {len(profiles)} columns")
        
        # Check profile completeness
        for col, profile in profiles.items():
            assert profile.name == col
            assert profile.dtype is not None
            assert 0 <= profile.null_ratio <= 1
            assert 0 <= profile.unique_ratio <= 1
        
        print_success("All column profiles valid")
        return True, profiles
        
    except Exception as e:
        print_error(f"Profiler failed: {e}")
        return False, None

def test_semantic_classifier(profiles):
    """Test semantic classifier"""
    print_header("TEST 2: Semantic Classifier")
    
    try:
        classifier = SemanticClassifier(profiles)
        semantic_model = classifier.classify()
        
        # Verify model structure
        assert hasattr(semantic_model, 'time_columns')
        assert hasattr(semantic_model, 'measures')
        assert hasattr(semantic_model, 'dimensions')
        
        total_classified = (len(semantic_model.time_columns) +
                          len(semantic_model.measures) +
                          len(semantic_model.dimensions) +
                          len(semantic_model.identifiers) +
                          len(semantic_model.rates) +
                          len(semantic_model.currencies) +
                          len(semantic_model.counts))
        
        print_success(f"Classified {total_classified} columns:")
        print(f"  - Time: {len(semantic_model.time_columns)}")
        print(f"  - Measures: {len(semantic_model.measures)}")
        print(f"  - Dimensions: {len(semantic_model.dimensions)}")
        print(f"  - Identifiers: {len(semantic_model.identifiers)}")
        print(f"  - Rates: {len(semantic_model.rates)}")
        print(f"  - Currencies: {len(semantic_model.currencies)}")
        print(f"  - Counts: {len(semantic_model.counts)}")
        
        return True, semantic_model
        
    except Exception as e:
        print_error(f"Classifier failed: {e}")
        return False, None

def test_metric_intelligence(df, semantic_model):
    """Test metric intelligence engine"""
    print_header("TEST 3: Metric Intelligence Engine")
    
    try:
        metric_engine = MetricIntelligenceEngine(df, semantic_model)
        kpi_candidates = metric_engine.discover_kpis()
        
        print_success(f"Discovered {len(kpi_candidates)} KPI candidates")
        
        if kpi_candidates:
            # Verify top KPI
            top_kpi = kpi_candidates[0]
            assert 0 <= top_kpi.score <= 100
            assert top_kpi.column in df.columns
            assert top_kpi.category is not None
            
            print(f"\n  Top KPI: {top_kpi.column}")
            print(f"  Score: {top_kpi.score:.1f}/100")
            print(f"  Category: {top_kpi.category}")
            print(f"  Rationale: {', '.join(top_kpi.rationale[:2])}")
        
        return True, kpi_candidates
        
    except Exception as e:
        print_error(f"Metric intelligence failed: {e}")
        return False, []

def test_forecast_eligibility(df, semantic_model):
    """Test forecast eligibility engine"""
    print_header("TEST 4: Forecast Eligibility Engine")
    
    if not semantic_model.time_columns:
        print_warning("No time column detected - skipping forecast tests")
        return True, None
    
    time_col = semantic_model.time_columns[0]
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            print_warning(f"Cannot convert {time_col} to datetime - skipping")
            return True, None
    
    # Test with first numeric measure
    test_metrics = (semantic_model.measures + 
                   semantic_model.currencies + 
                   semantic_model.counts)
    
    if not test_metrics:
        print_warning("No numeric measures found - skipping forecast tests")
        return True, None
    
    try:
        test_metric = test_metrics[0]
        eligibility_engine = ForecastEligibilityEngine(df, time_col, test_metric)
        eligibility = eligibility_engine.check()
        
        print_success(f"Eligibility check completed for '{test_metric}'")
        print(f"  Eligible: {'Yes' if eligibility.is_eligible else 'No'}")
        print(f"  Score: {eligibility.score:.0f}/100")
        print(f"  Reason: {eligibility.reason}")
        
        if eligibility.warnings:
            print(f"  Warnings: {len(eligibility.warnings)}")
        
        return True, eligibility
        
    except Exception as e:
        print_error(f"Forecast eligibility failed: {e}")
        return False, None

def test_data_quality_checks(df):
    """Test data quality validations"""
    print_header("TEST 5: Data Quality Checks")
    
    try:
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            print_warning(f"Found {len(empty_cols)} empty columns: {empty_cols}")
        else:
            print_success("No completely empty columns")
        
        # Check for columns with zero variance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zero_var_cols = []
        for col in numeric_cols:
            if df[col].dropna().var() == 0:
                zero_var_cols.append(col)
        
        if zero_var_cols:
            print_warning(f"Found {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
        else:
            print_success("No zero-variance numeric columns")
        
        # Check overall completeness
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        print_success(f"Overall data completeness: {completeness:.1f}%")
        
        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            print_warning(f"Found {n_duplicates} duplicate rows ({n_duplicates/len(df)*100:.1f}%)")
        else:
            print_success("No duplicate rows")
        
        return True
        
    except Exception as e:
        print_error(f"Data quality checks failed: {e}")
        return False

def test_edge_cases():
    """Test edge case handling"""
    print_header("TEST 6: Edge Case Handling")
    
    passed = 0
    total = 0
    
    # Test 1: Empty DataFrame
    total += 1
    try:
        df_empty = pd.DataFrame()
        profiler = DatasetProfiler(df_empty)
        profiles = profiler.profile()
        assert len(profiles) == 0
        print_success("Handled empty DataFrame")
        passed += 1
    except Exception as e:
        print_error(f"Empty DataFrame test failed: {e}")
    
    # Test 2: Single column
    total += 1
    try:
        df_single = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        profiler = DatasetProfiler(df_single)
        profiles = profiler.profile()
        assert len(profiles) == 1
        print_success("Handled single column DataFrame")
        passed += 1
    except Exception as e:
        print_error(f"Single column test failed: {e}")
    
    # Test 3: All null column
    total += 1
    try:
        df_null = pd.DataFrame({'null_col': [None, None, None]})
        profiler = DatasetProfiler(df_null)
        profiles = profiler.profile()
        assert profiles['null_col'].null_ratio == 1.0
        print_success("Handled all-null column")
        passed += 1
    except Exception as e:
        print_error(f"All-null column test failed: {e}")
    
    # Test 4: Mixed types
    total += 1
    try:
        df_mixed = pd.DataFrame({
            'mixed': [1, 'two', 3.0, None, '5']
        })
        profiler = DatasetProfiler(df_mixed)
        profiles = profiler.profile()
        print_success("Handled mixed-type column")
        passed += 1
    except Exception as e:
        print_error(f"Mixed types test failed: {e}")
    
    print(f"\n📊 Edge cases passed: {passed}/{total}")
    return passed == total

def run_comprehensive_tests(df, dataset_name):
    """Run all tests on a dataset"""
    print("\n" + "="*80)
    print(f"{Colors.BOLD}TESTING DATASET: {dataset_name}{Colors.END}")
    print(f"Dimensions: {len(df)} rows × {len(df.columns)} columns")
    print("="*80)
    
    results = {}
    
    # Test 1: Profiler
    success, profiles = test_semantic_profiler(df)
    results['profiler'] = success
    if not success:
        return results
    
    # Test 2: Classifier
    success, semantic_model = test_semantic_classifier(profiles)
    results['classifier'] = success
    if not success:
        return results
    
    # Test 3: Metric Intelligence
    success, kpis = test_metric_intelligence(df, semantic_model)
    results['metric_intelligence'] = success
    
    # Test 4: Forecast Eligibility
    success, eligibility = test_forecast_eligibility(df, semantic_model)
    results['forecast_eligibility'] = success
    
    # Test 5: Data Quality
    success = test_data_quality_checks(df)
    results['data_quality'] = success
    
    return results

def main():
    """Main validation function"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "AI ANALYTICS DASHBOARD" + " "*36 + "║")
    print("║" + " "*20 + "VALIDATION TEST SUITE" + " "*37 + "║")
    print("╚" + "="*78 + "╝")
    print(Colors.END)
    
    # Generate test data
    print_header("Generating Test Datasets")
    
    from generate_test_data import (
        generate_sales_data, generate_financial_data,
        generate_hr_data, generate_dirty_data
    )
    
    test_datasets = {
        'Sales Data': generate_sales_data(500),
        'Financial Data': generate_financial_data(200),
        'HR Data': generate_hr_data(150),
        'Dirty Data': generate_dirty_data(100)
    }
    
    print_success(f"Generated {len(test_datasets)} test datasets")
    
    # Run tests on each dataset
    all_results = {}
    
    for name, df in test_datasets.items():
        results = run_comprehensive_tests(df, name)
        all_results[name] = results
    
    # Test edge cases
    edge_case_success = test_edge_cases()
    
    # Final summary
    print_header("VALIDATION SUMMARY")
    
    total_tests = 0
    passed_tests = 0
    
    for dataset, results in all_results.items():
        print(f"\n{Colors.BOLD}{dataset}:{Colors.END}")
        for test_name, passed in results.items():
            total_tests += 1
            if passed:
                passed_tests += 1
                print_success(test_name)
            else:
                print_error(test_name)
    
    print(f"\n{Colors.BOLD}Edge Cases:{Colors.END}")
    if edge_case_success:
        print_success("All edge cases handled")
        total_tests += 1
        passed_tests += 1
    else:
        print_error("Some edge cases failed")
        total_tests += 1
    
    # Final score
    score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "="*80)
    print(f"{Colors.BOLD}FINAL SCORE: {passed_tests}/{total_tests} tests passed ({score:.1f}%){Colors.END}")
    
    if score == 100:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL TESTS PASSED - PRODUCTION READY!{Colors.END}")
    elif score >= 80:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  MOSTLY PASSING - REVIEW FAILURES{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ MULTIPLE FAILURES - FIX BEFORE DEPLOYMENT{Colors.END}")
    
    print("="*80 + "\n")
    
    return score == 100

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
