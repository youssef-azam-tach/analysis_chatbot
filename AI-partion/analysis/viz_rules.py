"""
Visualization Rules Engine — Power BI Best Practices
=====================================================
Central module enforcing chart quality rules across ALL pages:
- Column classification (ID vs Measure vs Dimension vs Date)
- Advanced ID detection (cardinality, uniqueness, sequential patterns,
  aggregation usefulness, distribution, entropy — weighted scoring)
- Chart type validation (line only for time, bar for categories, etc.)
- KPI label formatting (always include aggregation function)
- Smart column selection (never use IDs)
"""

import re
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
from scipy.stats import skew as scipy_skew


# ═══════════════════════════════════════════════════════════
#  ADVANCED ID CONFIDENCE ENGINE
# ═══════════════════════════════════════════════════════════
# Each sub-scorer returns 0-100.  The final weighted total
# decides whether a column is an identifier (threshold ≥ 60).

# Weights ---------------------------------------------------
_W_CARDINALITY  = 0.25
_W_UNIQUENESS   = 0.25
_W_SEQUENTIAL   = 0.20
_W_AGGREGATION  = 0.10
_W_DISTRIBUTION = 0.10
_W_ENTROPY      = 0.10
_ID_THRESHOLD   = 60        # total_score ≥ this → ID


def _cardinality_ratio(series: pd.Series) -> float:
    """Unique values / total values (0.0 – 1.0)."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    return non_null.nunique() / len(non_null)


def _uniqueness_score(series: pd.Series) -> float:
    """Score 0–100: how likely the column is an ID based on uniqueness."""
    ratio = _cardinality_ratio(series)
    if ratio >= 0.99:
        return 100
    elif ratio >= 0.95:
        return 90
    elif ratio >= 0.90:
        return 75
    elif ratio >= 0.75:
        return 40
    return 0


def _sequential_score(series: pd.Series) -> float:
    """Detect sequential numbers like auto-increment IDs (0–100)."""
    if not pd.api.types.is_numeric_dtype(series):
        return 0
    non_null = series.dropna()
    if len(non_null) < 10:
        return 0
    sorted_vals = np.sort(non_null.values)
    diffs = np.diff(sorted_vals)
    if len(diffs) == 0:
        return 0
    return float(np.mean(diffs == 1)) * 100


def _aggregation_usefulness_score(series: pd.Series) -> float:
    """
    IDs yield meaningless aggregations (high CV).
    Measures yield meaningful aggregations (moderate CV).
    Score 0–100: higher = more ID-like.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return 0
    non_null = series.dropna()
    if len(non_null) < 10:
        return 0
    mean = non_null.mean()
    if mean == 0:
        return 0
    cv = non_null.std() / mean
    if cv > 1.5:
        return 80
    elif cv > 1.0:
        return 60
    elif cv > 0.5:
        return 30
    return 0


def _distribution_score(series: pd.Series) -> float:
    """
    IDs usually have uniform distribution (low skewness).
    Measures usually skewed.  Score 0–100: higher = more ID-like.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return 0
    non_null = series.dropna()
    if len(non_null) < 10:
        return 0
    skewness = abs(float(scipy_skew(non_null)))
    if skewness < 0.3:
        return 70
    elif skewness < 0.7:
        return 40
    return 0


def _entropy_score(series: pd.Series) -> float:
    """
    High character-level entropy → random IDs / UUIDs / hashes.
    Score 0–100: higher = more ID-like.
    """
    non_null = series.dropna().astype(str)
    if len(non_null) < 10:
        return 0
    sample = ''.join(non_null.head(200))
    if not sample:
        return 0
    freq = Counter(sample)
    total = len(sample)
    probs = [v / total for v in freq.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    if entropy > 4:
        return 80
    elif entropy > 3:
        return 50
    return 0


def id_confidence_score(series: pd.Series, col_name: str) -> Dict[str, Any]:
    """
    Compute a detailed confidence breakdown for whether *series* is an ID.

    Returns dict with individual scores, total_score (0-100), and is_id bool.
    """
    scores: Dict[str, Any] = {}

    # ── Name-based fast-path (certainty = 100) ────────────
    name = str(col_name).lower().strip()
    name_score = _name_based_id_score(name, col_name)
    scores['name_match'] = name_score  # informational

    if name_score >= 100:
        # Name alone is conclusive — skip statistics
        scores.update({
            'cardinality': 100, 'uniqueness': 100, 'sequential': 0,
            'aggregation': 0, 'distribution': 0, 'entropy': 0,
            'total_score': 100.0, 'is_id': True,
        })
        return scores

    # ── Statistical sub-scores ────────────────────────────
    scores['cardinality'] = round(_cardinality_ratio(series) * 100, 2)
    scores['uniqueness'] = _uniqueness_score(series)
    scores['sequential'] = _sequential_score(series)
    scores['aggregation'] = _aggregation_usefulness_score(series)
    scores['distribution'] = _distribution_score(series)
    scores['entropy'] = _entropy_score(series)

    # ── Dtype adjustment ──────────────────────────────────
    # IDs are almost NEVER floating-point.  Continuous floats naturally
    # have near-unique values, so cardinality/uniqueness are misleading.
    is_float = pd.api.types.is_float_dtype(series)
    if is_float:
        scores['cardinality'] *= 0.25   # heavy penalty
        scores['uniqueness']  *= 0.25
        scores['sequential']   = 0      # floats aren't sequential IDs

    # ── Weighted total ────────────────────────────────────
    total = (
        scores['cardinality']  * _W_CARDINALITY  +
        scores['uniqueness']   * _W_UNIQUENESS   +
        scores['sequential']   * _W_SEQUENTIAL   +
        scores['aggregation']  * _W_AGGREGATION  +
        scores['distribution'] * _W_DISTRIBUTION +
        scores['entropy']      * _W_ENTROPY
    )

    # Boost if the column name partially looks like an ID
    if name_score > 0:
        total = max(total, name_score)  # name hint wins if higher

    scores['total_score'] = round(total, 2)
    scores['is_id'] = total >= _ID_THRESHOLD
    return scores


def classify_id_columns(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Run the confidence engine on every column.
    Returns {col_name: {cardinality, uniqueness, …, total_score, is_id}}.
    """
    return {col: id_confidence_score(df[col], col) for col in df.columns}


# ─── Name-Based Heuristics (fast path) ────────────────────

_EXACT_ID_NAMES = frozenset({
    'id', 'uuid', 'guid', 'rowid', 'row_id', 'index',
    'pk', 'fk', 'key', 'rowguid',
})

_ID_SUFFIXES = ('_id', 'id', '_key', '_pk', '_fk', '_guid', '_uuid',
                '_code', 'code', '_number', 'number')

_ID_TOKENS = frozenset({
    'id', 'key', 'pk', 'fk', 'uuid', 'guid', 'rowguid', 'entityid', 'rowid',
    'code', 'hash', 'token',
})


def _name_based_id_score(name_lower: str, original_name: str) -> float:
    """
    Return 0-100 based purely on column name patterns.
    100 = definitely ID by name,  50 = suspicious,  0 = no name signal.
    """
    if name_lower in _EXACT_ID_NAMES:
        return 100

    # Suffix check  (e.g. customer_id, productid)
    for suffix in _ID_SUFFIXES:
        if name_lower.endswith(suffix) and len(name_lower) > len(suffix):
            return 100

    # CamelCase ending in ID  (BusinessEntityID, SalesTaxRateID)
    if re.search(r'[a-z]ID$', str(original_name)):
        return 100

    # Token-level check  (e.g. sale_key, row_guid)
    tokens = [t for t in re.split(r'[^a-z0-9]+', name_lower) if t]
    if any(t in _ID_TOKENS for t in tokens):
        return 100

    # Partial hints (weaker)
    if 'num' in name_lower and 'number' not in name_lower:
        return 30  # "item_num" is ambiguous

    return 0


# ═══════════════════════════════════════════════════════════
#  PUBLIC COLUMN CLASSIFICATION API  (unchanged signature)
# ═══════════════════════════════════════════════════════════

def is_id_column(series: pd.Series, col_name: str) -> bool:
    """
    Detect if a column is an identifier — uses the full confidence engine.
    These MUST NEVER be used in chart axes, legends, or KPIs.
    """
    return id_confidence_score(series, col_name)['is_id']


def is_date_column(series: pd.Series, col_name: str) -> bool:
    """Detect if a column is date/time-based."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    
    name_lower = str(col_name).lower()
    date_tokens = ('date', 'time', 'timestamp', 'datetime', 'year', 'month', 'day',
                   'quarter', 'week', 'period', 'created', 'updated', 'modified')
    if any(token in name_lower for token in date_tokens):
        # Verify by trying to parse
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = series.dropna().head(100)
            if not sample.empty:
                parsed = pd.to_datetime(sample, errors='coerce')
                if float(parsed.notna().mean()) >= 0.6:
                    return True
        # Even if it's numeric (e.g., Year column), the name suggests date
        if pd.api.types.is_integer_dtype(series):
            vals = series.dropna()
            if not vals.empty:
                min_val, max_val = vals.min(), vals.max()
                if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                    return True  # Likely a year column
        return True if pd.api.types.is_datetime64_any_dtype(series) else False
    
    return False


def is_measure_column(series: pd.Series, col_name: str) -> bool:
    """Detect if a column is a numeric measure (aggregatable business value)."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    if is_id_column(series, col_name):
        return False
    if is_date_column(series, col_name):
        return False
    return True


def is_dimension_column(series: pd.Series, col_name: str) -> bool:
    """Detect if a column is a categorical dimension (descriptive category)."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    if is_id_column(series, col_name):
        return False
    if is_date_column(series, col_name):
        return False
    return True


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify all columns into: identifiers, measures, dimensions, dates.
    This is the SINGLE SOURCE OF TRUTH for column classification.
    """
    identifiers = []
    measures = []
    dimensions = []
    dates = []
    
    for col in df.columns:
        series = df[col]
        
        if is_id_column(series, col):
            identifiers.append(col)
        elif is_date_column(series, col):
            dates.append(col)
        elif is_measure_column(series, col):
            measures.append(col)
        elif is_dimension_column(series, col):
            dimensions.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # Fallback: numeric but not classified = check more carefully
            measures.append(col)
        else:
            dimensions.append(col)
    
    return {
        'identifiers': identifiers,
        'measures': measures,
        'dimensions': dimensions,
        'dates': dates,
    }


def get_measures(df: pd.DataFrame) -> List[str]:
    """Get only measure columns (safe for aggregation and chart Y-axis)."""
    return classify_columns(df)['measures']


def get_dimensions(df: pd.DataFrame) -> List[str]:
    """Get only dimension columns (safe for chart X-axis, legends, grouping)."""
    return classify_columns(df)['dimensions']


def get_dates(df: pd.DataFrame) -> List[str]:
    """Get only date columns (safe for line chart X-axis)."""
    return classify_columns(df)['dates']


# ─── Chart Type Validation ─────────────────────────────────

VALID_CHART_TYPES = {
    'bar', 'column', 'line', 'pie', 'donut', 'area',
    'scatter', 'bubble', 'histogram', 'boxplot', 'heatmap',
    'treemap', 'funnel', 'waterfall', 'combo', 'distribution',
    'trend', 'card', 'kpi', 'table',
}


def validate_chart_spec(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate a chart specification against Power BI best practices.
    Returns (is_valid, reason).
    """
    chart = (chart_type or '').lower().strip()
    classes = classify_columns(df)
    
    id_set = set(classes['identifiers'])
    measure_set = set(classes['measures'])
    dimension_set = set(classes['dimensions'])
    date_set = set(classes['dates'])
    
    # RULE: Never use ID columns
    if x_col and x_col in id_set:
        return False, f"BLOCKED: '{x_col}' is an identifier column. IDs must NEVER be used in chart axes."
    if y_col and y_col in id_set:
        return False, f"BLOCKED: '{y_col}' is an identifier column. IDs must NEVER be plotted as measures."
    
    # RULE: Bar/Column chart validation
    if chart in ('bar', 'column'):
        if not x_col or not y_col:
            return False, "Bar chart requires both a category (X) and a measure (Y)."
        if y_col not in measure_set:
            return False, f"Bar chart Y-axis must be a numeric measure, but '{y_col}' is not."
        if x_col not in dimension_set and x_col not in date_set:
            return False, f"Bar chart X-axis should be a category or date, but '{x_col}' is not."
        return True, ""
    
    # RULE: Line chart MUST have time on X-axis
    if chart in ('line', 'trend', 'area'):
        if not x_col or not y_col:
            return False, "Line chart requires a date (X) and a measure (Y)."
        if y_col not in measure_set:
            return False, f"Line chart Y must be a numeric measure, but '{y_col}' is not."
        if x_col not in date_set:
            return False, f"Line chart X MUST be a date/time column. '{x_col}' is not a date column. Use Bar chart for categories."
        return True, ""
    
    # RULE: Pie chart needs a dimension
    if chart in ('pie', 'donut'):
        if not x_col:
            return False, "Pie chart needs a categorical dimension."
        if x_col not in dimension_set:
            return False, f"Pie chart slices must be a business dimension. '{x_col}' is not suitable."
        if y_col and y_col not in measure_set:
            return False, f"Pie chart values must be a numeric measure when provided."
        return True, ""
    
    # RULE: Histogram needs a numeric measure
    if chart in ('histogram', 'distribution'):
        col = x_col or y_col
        if not col or col not in measure_set:
            return False, "Histogram/distribution requires a numeric measure column."
        return True, ""
    
    # RULE: Scatter needs two measures (NOT IDs)
    if chart in ('scatter', 'bubble'):
        if not x_col or not y_col:
            return False, "Scatter chart requires two numeric measures."
        if x_col not in measure_set or y_col not in measure_set:
            return False, "Scatter chart axes must both be numeric measures, not IDs or categories."
        if x_col == y_col:
            return False, "Scatter chart needs two DIFFERENT measures."
        return True, ""
    
    # RULE: Heatmap - correlation matrix of measures only
    if chart == 'heatmap':
        return True, ""
    
    # RULE: Boxplot needs a measure
    if chart == 'boxplot':
        col = y_col or x_col
        if not col or col not in measure_set:
            return False, "Boxplot requires a numeric measure."
        return True, ""
    
    # Default: allow (for unknown chart types)
    return True, ""


# ─── Smart Chart Recommendation ───────────────────────────

def recommend_best_chart(
    df: pd.DataFrame,
    purpose: str = 'general',
) -> List[Dict[str, Any]]:
    """
    Recommend the best chart types based on available data.
    Follows the Power BI decision tree.
    """
    classes = classify_columns(df)
    measures = classes['measures']
    dimensions = classes['dimensions']
    dates = classes['dates']
    
    recommendations = []
    
    # Decision tree: time-based data → Line chart
    if dates and measures:
        recommendations.append({
            'chart_type': 'line',
            'x_column': dates[0],
            'y_column': measures[0],
            'aggregation': 'SUM',
            'reason': f'Trend analysis: SUM of {measures[0]} over {dates[0]}',
            'title': f'SUM of {measures[0]} Trend Over {dates[0]}',
        })
    
    # Categories + measures → Bar chart (top 10)
    if dimensions and measures:
        recommendations.append({
            'chart_type': 'bar',
            'x_column': dimensions[0],
            'y_column': measures[0],
            'aggregation': 'SUM',
            'reason': f'Compare SUM of {measures[0]} across {dimensions[0]}',
            'title': f'SUM of {measures[0]} by {dimensions[0]}',
        })
    
    # Proportions → Pie chart (if categories ≤ 6)
    if dimensions:
        n_unique = df[dimensions[0]].nunique()
        if n_unique <= 6 and measures:
            recommendations.append({
                'chart_type': 'pie',
                'x_column': dimensions[0],
                'y_column': measures[0],
                'aggregation': 'SUM',
                'reason': f'Proportion of {measures[0]} by {dimensions[0]} ({n_unique} categories)',
                'title': f'SUM of {measures[0]} Distribution by {dimensions[0]}',
            })
    
    # Distribution → Histogram
    if measures:
        recommendations.append({
            'chart_type': 'histogram',
            'x_column': measures[0],
            'y_column': None,
            'aggregation': None,
            'reason': f'Distribution analysis of {measures[0]}',
            'title': f'Distribution of {measures[0]}',
        })
    
    # Correlation → Scatter (if 2+ measures)
    if len(measures) >= 2:
        recommendations.append({
            'chart_type': 'scatter',
            'x_column': measures[0],
            'y_column': measures[1],
            'aggregation': None,
            'reason': f'Relationship between {measures[0]} and {measures[1]}',
            'title': f'{measures[0]} vs {measures[1]}',
        })
    
    return recommendations


# ─── KPI Label Formatting ─────────────────────────────────

def format_kpi_label(column_name: str, aggregation: str) -> str:
    """
    Format KPI labels to ALWAYS include the aggregation function.
    E.g., 'SUM of Sales', 'COUNT of Customers', 'AVG Revenue per Order'
    """
    agg = (aggregation or 'SUM').upper()
    col = str(column_name).strip()
    
    # Clean up column name
    col = col.replace('_', ' ').title()
    
    return f"{agg} of {col}"


def format_chart_title(
    chart_type: str,
    x_col: Optional[str],
    y_col: Optional[str],
    aggregation: str = 'SUM',
) -> str:
    """
    Format chart titles with aggregation function included.
    E.g., 'SUM of Sales by Product Category'
    """
    agg = (aggregation or 'SUM').upper()
    
    if y_col and x_col:
        return f"{agg} of {y_col} by {x_col}"
    elif y_col:
        return f"{agg} of {y_col}"
    elif x_col:
        return f"Distribution of {x_col}"
    return "Chart"


# ─── LLM Prompt Injection ─────────────────────────────────

VISUALIZATION_RULES_PROMPT = """
## CRITICAL VISUALIZATION RULES — MUST FOLLOW STRICTLY

### RULE 1: NEVER USE ID COLUMNS
The following are STRICTLY FORBIDDEN in charts, axes, legends, grouping, or KPIs:
- Primary Keys (CustomerID, ProductID, OrderID, BusinessEntityID, etc.)
- Foreign Keys (TerritoryID, RegionID, StateProvinceID, etc.)
- System IDs (SalesTaxRateID, RowID, etc.)
- Any column ending with "ID" or containing "Key", "PK", "FK"
- Any column with near-unique integer values

If a dataset only contains ID columns and no meaningful descriptive columns, DO NOT recommend a chart.

### RULE 2: ONLY USE MEANINGFUL BUSINESS COLUMNS
Allowed columns for charts:
- Measures: Sales, Revenue, Profit, Cost, Quantity, Amount, Price, Rate, Score (numeric, aggregatable)
- Dimensions: Product Name, Category, Region, Country, Department, Segment (categorical, descriptive)
- Dates: Date, Month, Year, Quarter, OrderDate, CreatedDate (time-based)

### RULE 3: ALWAYS USE AGGREGATION FUNCTIONS
All measures MUST specify aggregation: SUM, COUNT, AVG, MIN, MAX
KPI labels MUST include aggregation: "SUM of Sales", "COUNT of Customers", "AVG Order Value"
NEVER label a KPI as just "Sales" or "Customers" — always include the function.

### RULE 4: MATCH CHART TYPE TO DATA TYPE
- Time-based data → Line Chart or Area Chart (X-axis MUST be date/time)
- Comparing categories → Bar Chart or Column Chart
- Proportions (≤6 categories) → Pie Chart or Donut Chart
- Correlation between measures → Scatter Chart
- Distribution of a measure → Histogram
- Single key value → Card/KPI with aggregation label

### RULE 5: LINE CHART ONLY FOR TIME SERIES
NEVER use line chart with categorical data or IDs on X-axis.
Line chart X-axis MUST be: Date, Month, Year, Quarter, Week, Time.
If you want to compare categories, use Bar Chart instead.

### RULE 6: LIMIT VISUAL CLUTTER
- Bar chart: max 15 categories (use Top N)
- Pie chart: max 6 slices (group rest into "Other")
- Line chart: max 4 lines

### RULE 7: VALIDATE BEFORE RECOMMENDING
Before recommending any visualization:
Step 1: Remove ALL ID columns from consideration
Step 2: Identify which columns are measures, dimensions, dates
Step 3: Choose chart type matching the data type
Step 4: Specify aggregation function
Step 5: Ensure the chart answers a real business question
"""


def get_viz_rules_for_prompt() -> str:
    """Return visualization rules as a string to inject into LLM prompts."""
    return VISUALIZATION_RULES_PROMPT


def get_column_classification_for_prompt(df: pd.DataFrame, dataset_name: str = "dataset") -> str:
    """
    Return a formatted string describing column classifications for LLM context.
    Includes confidence scores for detected IDs so the LLM understands *why*.
    """
    classes = classify_columns(df)
    id_details = classify_id_columns(df)
    
    lines = [f"\n=== Column Classification for {dataset_name} ==="]
    
    if classes['identifiers']:
        id_parts = []
        for col in classes['identifiers']:
            score = id_details[col]['total_score']
            id_parts.append(f"{col} (confidence: {score:.0f}%)")
        lines.append(f"⛔ IDENTIFIER COLUMNS (NEVER use in charts): {', '.join(id_parts)}")
    if classes['measures']:
        lines.append(f"✅ MEASURE COLUMNS (use for Y-axis, values, aggregation): {', '.join(classes['measures'])}")
    if classes['dimensions']:
        lines.append(f"✅ DIMENSION COLUMNS (use for X-axis, categories, grouping): {', '.join(classes['dimensions'])}")
    if classes['dates']:
        lines.append(f"✅ DATE COLUMNS (use for line chart X-axis, time trends): {', '.join(classes['dates'])}")
    
    if not classes['measures'] and not classes['dimensions'] and not classes['dates']:
        lines.append("⚠️ WARNING: No meaningful chart columns found. This dataset contains only identifiers.")
    
    return '\n'.join(lines)
