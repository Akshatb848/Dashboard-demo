"""
Semantic Profiling Engine for AI Analytics Dashboard
Provides intelligent column classification, KPI discovery, and forecast eligibility
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
import re


# ======================================================
# COLUMN PROFILE
# ======================================================

@dataclass
class ColumnProfile:
    """Statistical and semantic profile of a data column"""
    name: str
    dtype: str
    null_ratio: float
    unique_ratio: float
    is_numeric: bool
    is_datetime: bool
    variance: Optional[float]
    monotonic: bool
    cardinality: int
    sample_values: List[str] = field(default_factory=list)
    detected_patterns: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"ColumnProfile(name={self.name}, type={self.dtype}, null={self.null_ratio:.2%})"


class DatasetProfiler:
    """Profiles entire datasets with statistical and structural analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._profiles: Dict[str, ColumnProfile] = {}
    
    def profile(self) -> Dict[str, ColumnProfile]:
        """Generate comprehensive profile for all columns"""
        profiles = {}
        
        for col in self.df.columns:
            series = self.df[col].dropna()
            
            if len(series) == 0:
                profiles[col] = self._create_empty_profile(col)
                continue
            
            # Basic statistics
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)
            
            # Advanced detection
            patterns = self._detect_patterns(series)
            
            profiles[col] = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                null_ratio=self.df[col].isnull().mean(),
                unique_ratio=series.nunique() / max(len(self.df), 1),
                is_numeric=is_numeric,
                is_datetime=is_datetime,
                variance=float(series.var()) if is_numeric and len(series) > 1 else None,
                monotonic=self._check_monotonic(series) if is_numeric else False,
                cardinality=series.nunique(),
                sample_values=self._get_sample_values(series, n=5),
                detected_patterns=patterns
            )
        
        self._profiles = profiles
        return profiles
    
    def _create_empty_profile(self, col: str) -> ColumnProfile:
        """Create profile for empty column"""
        return ColumnProfile(
            name=col,
            dtype="unknown",
            null_ratio=1.0,
            unique_ratio=0.0,
            is_numeric=False,
            is_datetime=False,
            variance=None,
            monotonic=False,
            cardinality=0,
            sample_values=[],
            detected_patterns=[]
        )
    
    def _check_monotonic(self, series: pd.Series) -> bool:
        """Check if series is monotonically increasing"""
        try:
            return series.is_monotonic_increasing or series.is_monotonic_decreasing
        except:
            return False
    
    def _get_sample_values(self, series: pd.Series, n: int = 5) -> List[str]:
        """Get sample values for inspection"""
        try:
            samples = series.value_counts().head(n).index.tolist()
            return [str(v) for v in samples]
        except:
            return []
    
    def _detect_patterns(self, series: pd.Series) -> List[str]:
        """Detect semantic patterns in column data"""
        patterns = []
        
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's a percentage
            if series.min() >= 0 and series.max() <= 100:
                patterns.append("percentage")
            
            # Check if it's currency-like (large numbers)
            if series.mean() > 100 and series.std() > 10:
                patterns.append("currency")
            
            # Check if it's a count (integers)
            if series.dtype in ['int64', 'int32']:
                patterns.append("count")
        
        elif pd.api.types.is_string_dtype(series):
            samples = series.astype(str).head(100)
            
            # Email pattern
            if samples.str.contains(r'^[\w\.-]+@[\w\.-]+\.\w+$', regex=True).any():
                patterns.append("email")
            
            # URL pattern
            if samples.str.contains(r'^https?://', regex=True).any():
                patterns.append("url")
            
            # Phone pattern
            if samples.str.contains(r'^\+?\d[\d\s\-\(\)]+$', regex=True).any():
                patterns.append("phone")
        
        return patterns


# ======================================================
# SEMANTIC CLASSIFIER
# ======================================================

@dataclass
class SemanticModel:
    """Business model representation of dataset structure"""
    time_columns: List[str] = field(default_factory=list)
    measures: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)
    rates: List[str] = field(default_factory=list)
    currencies: List[str] = field(default_factory=list)
    counts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary representation"""
        return {
            "time": self.time_columns,
            "measures": self.measures,
            "dimensions": self.dimensions,
            "identifiers": self.identifiers,
            "rates": self.rates,
            "currencies": self.currencies,
            "counts": self.counts
        }
    
    def __repr__(self):
        return (f"SemanticModel(time={len(self.time_columns)}, "
                f"measures={len(self.measures)}, "
                f"dimensions={len(self.dimensions)})")


class SemanticClassifier:
    """Classifies columns into business-meaningful categories"""
    
    # Keywords for semantic classification
    TIME_KEYWORDS = {'date', 'time', 'timestamp', 'created', 'updated', 'dt', 'day', 'month', 'year'}
    MEASURE_KEYWORDS = {'amount', 'total', 'sum', 'revenue', 'sales', 'price', 'cost', 'value', 'count', 'quantity', 'qty'}
    IDENTIFIER_KEYWORDS = {'id', 'key', 'code', 'number', 'uuid', 'guid', 'reference', 'ref'}
    RATE_KEYWORDS = {'rate', 'ratio', 'percent', 'percentage', 'margin', 'share', '%'}
    CURRENCY_KEYWORDS = {'price', 'cost', 'revenue', 'sales', 'amount', 'payment', 'salary', 'budget'}
    COUNT_KEYWORDS = {'count', 'quantity', 'qty', 'number', 'total', 'sum'}
    
    def __init__(self, profiles: Dict[str, ColumnProfile]):
        self.profiles = profiles
        self.model = SemanticModel()
    
    def classify(self) -> SemanticModel:
        """Classify all columns into semantic categories"""
        
        for col, profile in self.profiles.items():
            col_lower = col.lower()
            
            # Priority 1: Time columns
            if self._is_time_column(profile, col_lower):
                self.model.time_columns.append(col)
                continue
            
            # Priority 2: Identifiers (high cardinality, unique values)
            if self._is_identifier(profile, col_lower):
                self.model.identifiers.append(col)
                continue
            
            # Priority 3: Rates/Percentages
            if self._is_rate(profile, col_lower):
                self.model.rates.append(col)
                continue
            
            # Priority 4: Measures (numeric with variance)
            if self._is_measure(profile, col_lower):
                # Sub-classify measures
                if self._is_currency(profile, col_lower):
                    self.model.currencies.append(col)
                elif self._is_count(profile, col_lower):
                    self.model.counts.append(col)
                else:
                    self.model.measures.append(col)
                continue
            
            # Default: Dimension
            if not profile.is_numeric or profile.cardinality < len(self.profiles) * 0.5:
                self.model.dimensions.append(col)
        
        return self.model
    
    def _is_time_column(self, profile: ColumnProfile, col_lower: str) -> bool:
        """Check if column represents time/date"""
        if profile.is_datetime:
            return True
        
        # Check keywords
        if any(kw in col_lower for kw in self.TIME_KEYWORDS):
            return True
        
        return False
    
    def _is_identifier(self, profile: ColumnProfile, col_lower: str) -> bool:
        """Check if column is an identifier"""
        # High uniqueness ratio suggests identifier
        if profile.unique_ratio > 0.95:
            return True
        
        # Check keywords
        if any(kw in col_lower for kw in self.IDENTIFIER_KEYWORDS):
            return True
        
        # Monotonic integers often IDs
        if profile.is_numeric and profile.monotonic and profile.cardinality == len(self.profiles):
            return True
        
        return False
    
    def _is_rate(self, profile: ColumnProfile, col_lower: str) -> bool:
        """Check if column represents a rate/percentage"""
        # Check for percentage pattern
        if "percentage" in profile.detected_patterns:
            return True
        
        # Check keywords
        if any(kw in col_lower for kw in self.RATE_KEYWORDS):
            return True
        
        return False
    
    def _is_measure(self, profile: ColumnProfile, col_lower: str) -> bool:
        """Check if column is a numeric measure"""
        if not profile.is_numeric:
            return False
        
        # Must have variance to be interesting
        if profile.variance is None or profile.variance == 0:
            return False
        
        # Check keywords
        if any(kw in col_lower for kw in self.MEASURE_KEYWORDS):
            return True
        
        # Numeric with reasonable cardinality
        if profile.cardinality > 10:
            return True
        
        return False
    
    def _is_currency(self, profile: ColumnProfile, col_lower: str) -> bool:
        """Check if measure represents currency"""
        if "currency" in profile.detected_patterns:
            return True
        
        if any(kw in col_lower for kw in self.CURRENCY_KEYWORDS):
            return True
        
        return False
    
    def _is_count(self, profile: ColumnProfile, col_lower: str) -> bool:
        """Check if measure represents a count"""
        if "count" in profile.detected_patterns:
            return True
        
        if any(kw in col_lower for kw in self.COUNT_KEYWORDS):
            return True
        
        return False


# ======================================================
# METRIC INTELLIGENCE ENGINE
# ======================================================

@dataclass
class KPICandidate:
    """Represents a potential Key Performance Indicator"""
    column: str
    score: float
    category: str
    rationale: List[str]
    
    def __repr__(self):
        return f"KPI({self.column}, score={self.score:.2f}, {self.category})"


class MetricIntelligenceEngine:
    """Discovers and ranks potential KPIs from dataset"""
    
    def __init__(self, df: pd.DataFrame, semantic: SemanticModel):
        self.df = df
        self.semantic = semantic
        self._kpi_candidates: List[KPICandidate] = []
    
    def discover_kpis(self) -> List[KPICandidate]:
        """Discover and rank KPI candidates"""
        candidates = []
        
        # Analyze all measure columns
        all_measures = (self.semantic.measures + 
                       self.semantic.currencies + 
                       self.semantic.counts + 
                       self.semantic.rates)
        
        for col in all_measures:
            if col not in self.df.columns:
                continue
            
            series = self.df[col].dropna()
            
            if len(series) < 2:
                continue
            
            # Score the KPI
            score, rationale = self._score_kpi(series, col)
            
            # Categorize
            category = self._categorize_kpi(col)
            
            candidates.append(KPICandidate(
                column=col,
                score=score,
                category=category,
                rationale=rationale
            ))
        
        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        self._kpi_candidates = candidates
        return candidates
    
    def _score_kpi(self, series: pd.Series, col: str) -> Tuple[float, List[str]]:
        """Score a potential KPI (0-100)"""
        score = 0.0
        rationale = []
        
        # Factor 1: Data quality (max 25 points)
        completeness = 1 - series.isnull().mean()
        score += completeness * 25
        if completeness > 0.9:
            rationale.append(f"{completeness:.0%} data completeness")
        
        # Factor 2: Variance (max 25 points)
        if series.std() > 0:
            cv = series.std() / series.mean() if series.mean() != 0 else 0
            variance_score = min(cv * 10, 25)  # Cap at 25
            score += variance_score
            if variance_score > 15:
                rationale.append("High variability (interesting for analysis)")
        
        # Factor 3: Cardinality (max 20 points)
        unique_ratio = series.nunique() / len(series)
        if 0.1 < unique_ratio < 0.9:  # Sweet spot
            score += 20
            rationale.append("Good range of unique values")
        elif unique_ratio >= 0.9:
            score += 10
        
        # Factor 4: Business relevance keywords (max 30 points)
        col_lower = col.lower()
        important_keywords = {'revenue', 'sales', 'profit', 'cost', 'margin', 
                            'conversion', 'retention', 'churn', 'growth'}
        
        keyword_matches = sum(1 for kw in important_keywords if kw in col_lower)
        keyword_score = min(keyword_matches * 15, 30)
        score += keyword_score
        
        if keyword_matches > 0:
            rationale.append(f"Contains business keyword(s)")
        
        return min(score, 100), rationale
    
    def _categorize_kpi(self, col: str) -> str:
        """Categorize KPI type"""
        col_lower = col.lower()
        
        if col in self.semantic.currencies:
            return "Financial"
        elif col in self.semantic.rates:
            return "Performance"
        elif col in self.semantic.counts:
            return "Volume"
        elif any(kw in col_lower for kw in ['growth', 'change', 'increase']):
            return "Growth"
        elif any(kw in col_lower for kw in ['efficiency', 'productivity']):
            return "Efficiency"
        else:
            return "Operational"
    
    def get_top_kpis(self, n: int = 5) -> List[str]:
        """Get top N KPI column names"""
        if not self._kpi_candidates:
            self.discover_kpis()
        
        return [kpi.column for kpi in self._kpi_candidates[:n]]


# ======================================================
# FORECAST ELIGIBILITY ENGINE
# ======================================================

@dataclass
class ForecastEligibility:
    """Results of forecast eligibility check"""
    is_eligible: bool
    score: float  # 0-100
    reason: str
    warnings: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"Eligible={self.is_eligible} (score={self.score:.0f})"


class ForecastEligibilityEngine:
    """Determines if a metric is suitable for forecasting"""
    
    MIN_OBSERVATIONS = 15
    MIN_SCORE = 60  # Minimum score to be eligible
    
    def __init__(self, df: pd.DataFrame, time_col: str, metric: str):
        self.df = df
        self.time_col = time_col
        self.metric = metric
    
    def check(self) -> ForecastEligibility:
        """Comprehensive eligibility check"""
        
        # Prepare time series
        ts = self.df[[self.time_col, self.metric]].dropna().copy()
        ts = ts.sort_values(self.time_col)
        
        score = 0.0
        warnings = []
        reason = ""
        
        # Check 1: Sufficient data (0-25 points)
        n_obs = len(ts)
        if n_obs < self.MIN_OBSERVATIONS:
            return ForecastEligibility(
                is_eligible=False,
                score=0,
                reason=f"Insufficient data: {n_obs} observations (need {self.MIN_OBSERVATIONS}+)",
                warnings=warnings
            )
        
        data_score = min((n_obs / 50) * 25, 25)  # Max at 50+ observations
        score += data_score
        
        if n_obs < 30:
            warnings.append(f"Limited history ({n_obs} points)")
        
        # Check 2: Time regularity (0-25 points)
        time_deltas = ts[self.time_col].diff().dropna()
        if len(time_deltas) > 0:
            delta_cv = time_deltas.std() / time_deltas.mean() if time_deltas.mean() > pd.Timedelta(0) else float('inf')
            
            if delta_cv < 0.1:  # Very regular
                score += 25
            elif delta_cv < 0.3:  # Moderately regular
                score += 15
                warnings.append("Time intervals slightly irregular")
            else:
                score += 5
                warnings.append("Time intervals highly irregular")
        
        # Check 3: Stationarity (0-25 points)
        metric_series = ts[self.metric]
        mean = metric_series.mean()
        std = metric_series.std()
        
        if mean == 0:
            return ForecastEligibility(
                is_eligible=False,
                score=score,
                reason="Zero mean series - cannot forecast",
                warnings=warnings
            )
        
        cv = std / abs(mean)
        
        if cv < 0.5:  # Low volatility
            score += 25
        elif cv < 1.0:  # Moderate volatility
            score += 20
        elif cv < 2.0:  # High volatility
            score += 10
            warnings.append(f"High volatility (CV={cv:.2f})")
        else:  # Extreme volatility
            score += 5
            warnings.append(f"Extreme volatility (CV={cv:.2f})")
        
        # Check 4: Trend presence (0-25 points)
        if n_obs > 3:
            x = np.arange(len(metric_series))
            y = metric_series.values
            
            try:
                from scipy import stats
                slope, _, r_value, _, _ = stats.linregress(x, y)
                
                r_squared = r_value ** 2
                
                if r_squared > 0.7:  # Strong trend
                    score += 25
                elif r_squared > 0.4:  # Moderate trend
                    score += 20
                elif r_squared > 0.2:  # Weak trend
                    score += 15
                else:  # No clear trend
                    score += 10
                    warnings.append("No clear trend detected")
            except:
                score += 10
        
        # Determine eligibility
        is_eligible = score >= self.MIN_SCORE
        
        if is_eligible:
            reason = f"Forecast eligible (score: {score:.0f}/100)"
        else:
            reason = f"Score too low: {score:.0f}/100 (need {self.MIN_SCORE}+)"
        
        return ForecastEligibility(
            is_eligible=is_eligible,
            score=score,
            reason=reason,
            warnings=warnings
        )
    
    def get_recommendation(self, eligibility: ForecastEligibility) -> str:
        """Get human-readable recommendation"""
        if eligibility.is_eligible:
            if eligibility.score > 80:
                return "✅ Excellent candidate for forecasting"
            elif eligibility.score > 70:
                return "✅ Good candidate for forecasting"
            else:
                return "⚠️ Acceptable for forecasting (with caution)"
        else:
            return "❌ Not recommended for forecasting"
