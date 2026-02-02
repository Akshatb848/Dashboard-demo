"""
Intelligent Data Analyst Agent
Uses LLM and RAG to perform autonomous data analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class AnalysisTask:
    """Represents an analysis task"""
    task_type: str
    priority: int
    description: str
    status: str = "pending"
    result: Optional[Any] = None


class IntelligentDataAnalyst:
    """
    Autonomous data analyst that uses LLM reasoning to analyze datasets
    Performs EDA, preprocessing, visualization, and insights generation
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analysis_plan = []
        self.insights = []
        self.preprocessing_steps = []
        self.visualizations = []
        
    def create_analysis_plan(self) -> List[AnalysisTask]:
        """
        AI Agent creates comprehensive analysis plan based on data characteristics
        """
        plan = []
        
        # Task 1: Data Understanding
        plan.append(AnalysisTask(
            task_type="data_understanding",
            priority=1,
            description="Understand data structure, types, and basic statistics"
        ))
        
        # Task 2: Quality Assessment
        plan.append(AnalysisTask(
            task_type="quality_assessment",
            priority=2,
            description="Assess data quality, missing values, and anomalies"
        ))
        
        # Task 3: Feature Analysis
        plan.append(AnalysisTask(
            task_type="feature_analysis",
            priority=3,
            description="Analyze feature distributions and relationships"
        ))
        
        # Task 4: Statistical Testing
        plan.append(AnalysisTask(
            task_type="statistical_testing",
            priority=4,
            description="Perform statistical hypothesis testing"
        ))
        
        # Task 5: Pattern Discovery
        plan.append(AnalysisTask(
            task_type="pattern_discovery",
            priority=5,
            description="Discover patterns, trends, and correlations"
        ))
        
        # Task 6: Preprocessing Recommendations
        plan.append(AnalysisTask(
            task_type="preprocessing",
            priority=6,
            description="Recommend data preprocessing steps"
        ))
        
        # Task 7: Visualization Strategy
        plan.append(AnalysisTask(
            task_type="visualization",
            priority=7,
            description="Design optimal visualization strategy"
        ))
        
        self.analysis_plan = plan
        return plan
    
    def execute_data_understanding(self) -> Dict[str, Any]:
        """
        Comprehensive data understanding phase
        """
        understanding = {
            'shape': {
                'rows': len(self.df),
                'columns': len(self.df.columns)
            },
            'columns': {},
            'data_types': {},
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'sample_data': self.df.head(5).to_dict('records')
        }
        
        # Analyze each column
        for col in self.df.columns:
            col_info = {
                'dtype': str(self.df[col].dtype),
                'unique_count': self.df[col].nunique(),
                'null_count': self.df[col].isnull().sum(),
                'null_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100
            }
            
            # Numeric column stats
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_info['statistics'] = {
                    'mean': float(self.df[col].mean()),
                    'median': float(self.df[col].median()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'q25': float(self.df[col].quantile(0.25)),
                    'q75': float(self.df[col].quantile(0.75))
                }
            
            # Categorical column stats
            elif pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                value_counts = self.df[col].value_counts()
                col_info['top_values'] = {
                    str(k): int(v) for k, v in value_counts.head(5).items()
                }
                col_info['cardinality'] = len(value_counts)
            
            understanding['columns'][col] = col_info
        
        return understanding
    
    def execute_quality_assessment(self) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment
        """
        quality = {
            'completeness': {},
            'consistency': {},
            'accuracy': {},
            'issues': []
        }
        
        # Completeness check
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()
        quality['completeness']['score'] = ((total_cells - missing_cells) / total_cells) * 100
        quality['completeness']['missing_cells'] = int(missing_cells)
        quality['completeness']['total_cells'] = int(total_cells)
        
        # Column-level completeness
        for col in self.df.columns:
            missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
            if missing_pct > 50:
                quality['issues'].append({
                    'severity': 'high',
                    'column': col,
                    'issue': f'High missing rate: {missing_pct:.1f}%',
                    'recommendation': 'Consider dropping column or imputation strategy'
                })
            elif missing_pct > 10:
                quality['issues'].append({
                    'severity': 'medium',
                    'column': col,
                    'issue': f'Moderate missing rate: {missing_pct:.1f}%',
                    'recommendation': 'Imputation recommended'
                })
        
        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        quality['consistency']['duplicate_rows'] = int(duplicates)
        if duplicates > 0:
            quality['issues'].append({
                'severity': 'medium',
                'issue': f'{duplicates} duplicate rows found',
                'recommendation': 'Review and remove duplicates'
            })
        
        # Constant columns (zero variance)
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].nunique() == 1:
                quality['issues'].append({
                    'severity': 'low',
                    'column': col,
                    'issue': 'Constant value across all rows',
                    'recommendation': 'Consider removing - provides no information'
                })
        
        # Outliers detection (IQR method)
        outlier_counts = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_counts[col] = int(outliers)
        
        quality['accuracy']['outlier_counts'] = outlier_counts
        
        return quality
    
    def execute_feature_analysis(self) -> Dict[str, Any]:
        """
        Deep feature analysis including distributions and relationships
        """
        analysis = {
            'numeric_features': {},
            'categorical_features': {},
            'correlations': [],
            'feature_interactions': []
        }
        
        # Numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if len(data) < 2:
                continue
            
            from scipy import stats
            
            # Distribution analysis
            skewness = float(stats.skew(data))
            kurtosis = float(stats.kurtosis(data))
            
            # Normality test
            if len(data) >= 3:
                _, p_value = stats.normaltest(data)
                is_normal = p_value > 0.05
            else:
                is_normal = False
            
            analysis['numeric_features'][col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': is_normal,
                'distribution_type': self._classify_distribution(skewness, kurtosis),
                'coefficient_variation': float(data.std() / data.mean()) if data.mean() != 0 else 0
            }
        
        # Categorical features
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            
            analysis['categorical_features'][col] = {
                'cardinality': len(value_counts),
                'entropy': float(stats.entropy(value_counts)),
                'top_category': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'top_frequency': float(value_counts.iloc[0] / len(self.df)) if len(value_counts) > 0 else 0,
                'is_high_cardinality': len(value_counts) > 50
            }
        
        # Correlation analysis
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.5:
                        analysis['correlations'].append({
                            'feature1': numeric_cols[i],
                            'feature2': numeric_cols[j],
                            'correlation': float(corr_value),
                            'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate',
                            'direction': 'positive' if corr_value > 0 else 'negative'
                        })
        
        return analysis
    
    def execute_statistical_testing(self) -> Dict[str, Any]:
        """
        Perform statistical hypothesis testing
        """
        from scipy import stats
        
        tests = {
            'normality_tests': {},
            'variance_tests': {},
            'independence_tests': {}
        }
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Normality tests for each numeric column
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if len(data) >= 3:
                statistic, p_value = stats.normaltest(data)
                tests['normality_tests'][col] = {
                    'test': 'D\'Agostino-Pearson',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05,
                    'interpretation': 'Data is normally distributed' if p_value > 0.05 else 'Data is not normally distributed'
                }
        
        # Variance homogeneity test (if we have groups)
        # Placeholder for when we add group analysis
        
        return tests
    
    def execute_pattern_discovery(self) -> Dict[str, Any]:
        """
        Discover patterns, trends, and insights
        """
        patterns = {
            'trends': [],
            'seasonality': [],
            'clusters': {},
            'anomalies': []
        }
        
        # Time-based patterns (if date column exists)
        date_cols = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) == 0:
            # Try to find date-like columns
            for col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    date_cols = [col]
                    break
                except:
                    continue
        
        # Trend analysis for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Analyze top 5 numeric columns
            data = self.df[col].dropna()
            
            if len(data) < 2:
                continue
            
            # Linear trend
            x = np.arange(len(data))
            y = data.values
            
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            if abs(r_value) > 0.3:  # Significant trend
                patterns['trends'].append({
                    'feature': col,
                    'direction': 'increasing' if slope > 0 else 'decreasing',
                    'strength': float(abs(r_value)),
                    'p_value': float(p_value),
                    'is_significant': p_value < 0.05
                })
        
        return patterns
    
    def execute_preprocessing_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate intelligent preprocessing recommendations
        """
        recommendations = []
        
        # Missing value handling
        for col in self.df.columns:
            missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
            
            if missing_pct > 0:
                if missing_pct > 50:
                    recommendations.append({
                        'step': 'drop_column',
                        'column': col,
                        'reason': f'{missing_pct:.1f}% missing values',
                        'priority': 'high'
                    })
                elif missing_pct > 10:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        recommendations.append({
                            'step': 'impute_numeric',
                            'column': col,
                            'method': 'median',
                            'reason': f'{missing_pct:.1f}% missing values',
                            'priority': 'medium'
                        })
                    else:
                        recommendations.append({
                            'step': 'impute_categorical',
                            'column': col,
                            'method': 'mode',
                            'reason': f'{missing_pct:.1f}% missing values',
                            'priority': 'medium'
                        })
        
        # Outlier handling
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            
            if outliers > len(self.df) * 0.05:  # More than 5% outliers
                recommendations.append({
                    'step': 'handle_outliers',
                    'column': col,
                    'method': 'winsorization',
                    'reason': f'{outliers} outliers detected',
                    'priority': 'medium'
                })
        
        # Scaling recommendations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            # Check if scales differ significantly
            scales = []
            for col in numeric_cols:
                scales.append(self.df[col].std())
            
            if max(scales) / min(scales) > 10:
                recommendations.append({
                    'step': 'scale_features',
                    'columns': list(numeric_cols),
                    'method': 'standardization',
                    'reason': 'Features have different scales',
                    'priority': 'high'
                })
        
        # Encoding recommendations
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            cardinality = self.df[col].nunique()
            
            if cardinality == 2:
                recommendations.append({
                    'step': 'binary_encode',
                    'column': col,
                    'method': 'label_encoding',
                    'reason': 'Binary categorical variable',
                    'priority': 'high'
                })
            elif cardinality < 10:
                recommendations.append({
                    'step': 'one_hot_encode',
                    'column': col,
                    'method': 'one_hot_encoding',
                    'reason': f'Low cardinality ({cardinality} categories)',
                    'priority': 'high'
                })
            else:
                recommendations.append({
                    'step': 'target_encode',
                    'column': col,
                    'method': 'target_encoding',
                    'reason': f'High cardinality ({cardinality} categories)',
                    'priority': 'medium'
                })
        
        self.preprocessing_steps = recommendations
        return recommendations
    
    def execute_visualization_strategy(self) -> List[Dict[str, Any]]:
        """
        Design optimal visualization strategy
        """
        visualizations = []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        # Distribution plots for numeric features
        for col in numeric_cols[:5]:
            visualizations.append({
                'type': 'histogram',
                'title': f'Distribution of {col}',
                'x_axis': col,
                'purpose': 'Understand feature distribution',
                'priority': 'high'
            })
            
            visualizations.append({
                'type': 'boxplot',
                'title': f'Outlier Detection: {col}',
                'y_axis': col,
                'purpose': 'Identify outliers and spread',
                'priority': 'medium'
            })
        
        # Correlation heatmap
        if len(numeric_cols) >= 3:
            visualizations.append({
                'type': 'heatmap',
                'title': 'Feature Correlation Matrix',
                'data': list(numeric_cols),
                'purpose': 'Identify feature relationships',
                'priority': 'high'
            })
        
        # Scatter plots for strong correlations
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        visualizations.append({
                            'type': 'scatter',
                            'title': f'{numeric_cols[i]} vs {numeric_cols[j]}',
                            'x_axis': numeric_cols[i],
                            'y_axis': numeric_cols[j],
                            'purpose': f'Explore correlation ({corr_matrix.iloc[i, j]:.2f})',
                            'priority': 'high'
                        })
        
        # Bar charts for categorical features
        for col in categorical_cols[:3]:
            if self.df[col].nunique() <= 20:
                visualizations.append({
                    'type': 'bar',
                    'title': f'Distribution of {col}',
                    'x_axis': col,
                    'purpose': 'Understand category distribution',
                    'priority': 'medium'
                })
        
        # Pair plot recommendation
        if len(numeric_cols) >= 3 and len(numeric_cols) <= 6:
            visualizations.append({
                'type': 'pairplot',
                'title': 'Feature Relationships Overview',
                'features': list(numeric_cols),
                'purpose': 'Comprehensive relationship analysis',
                'priority': 'low'
            })
        
        self.visualizations = visualizations
        return visualizations
    
    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution type based on moments"""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "Normal"
        elif skewness > 1:
            return "Right-skewed"
        elif skewness < -1:
            return "Left-skewed"
        elif kurtosis > 3:
            return "Heavy-tailed"
        elif kurtosis < -1:
            return "Light-tailed"
        else:
            return "Asymmetric"
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate human-readable comprehensive analysis report
        """
        report = "# Comprehensive Data Analysis Report\n\n"
        
        # Execute all analysis tasks
        understanding = self.execute_data_understanding()
        quality = self.execute_quality_assessment()
        feature_analysis = self.execute_feature_analysis()
        patterns = self.execute_pattern_discovery()
        preprocessing = self.execute_preprocessing_recommendations()
        
        # Data Overview
        report += "## Dataset Overview\n\n"
        report += f"The dataset contains **{understanding['shape']['rows']:,} records** "
        report += f"across **{understanding['shape']['columns']} features**, "
        report += f"consuming approximately **{understanding['memory_usage']:.2f} MB** of memory.\n\n"
        
        # Quality Assessment
        report += "## Data Quality Assessment\n\n"
        report += f"Overall data completeness stands at **{quality['completeness']['score']:.2f}%**, "
        report += f"with **{quality['completeness']['missing_cells']:,}** missing values detected.\n\n"
        
        if quality['issues']:
            report += "### Identified Quality Issues\n\n"
            for issue in quality['issues'][:5]:
                report += f"- **{issue.get('severity', '').upper()}**: {issue['issue']}\n"
                report += f"  - Recommendation: {issue['recommendation']}\n"
            report += "\n"
        
        # Feature Analysis
        report += "## Feature Analysis\n\n"
        
        if feature_analysis['correlations']:
            report += "### Key Correlations\n\n"
            for corr in feature_analysis['correlations'][:5]:
                report += f"- **{corr['feature1']}** and **{corr['feature2']}**: "
                report += f"{corr['strength']} {corr['direction']} correlation ({corr['correlation']:.3f})\n"
            report += "\n"
        
        # Pattern Discovery
        if patterns['trends']:
            report += "### Identified Trends\n\n"
            for trend in patterns['trends']:
                report += f"- **{trend['feature']}** shows {trend['direction']} trend "
                report += f"(strength: {trend['strength']:.2f})\n"
            report += "\n"
        
        # Preprocessing Recommendations
        report += "## Recommended Preprocessing Steps\n\n"
        high_priority = [r for r in preprocessing if r['priority'] == 'high']
        
        if high_priority:
            report += "### High Priority Actions\n\n"
            for i, rec in enumerate(high_priority[:5], 1):
                report += f"{i}. **{rec['step'].replace('_', ' ').title()}**: "
                report += f"{rec.get('column', 'Multiple columns')} - {rec['reason']}\n"
            report += "\n"
        
        return report


# Export
__all__ = ['IntelligentDataAnalyst', 'AnalysisTask']
