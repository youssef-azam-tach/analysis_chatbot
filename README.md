# 📊 Data Analysis & AI Chatbot Platform

> **Comprehensive Data Analysis Solution for SME Businesses**  
> Built with Streamlit • Powered by Local LLMs via Ollama • Power Query-like Operations

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Application Pages](#application-pages)
5. [Core Modules](#core-modules)
6. [Intelligent Column Detection](#intelligent-column-detection)
7. [Data Quality Assessment](#data-quality-assessment)
8. [Data Cleaning Operations](#data-cleaning-operations)
9. [Power Query Operations](#power-query-operations)
10. [KPI Intelligence](#kpi-intelligence)
11. [Visualization System](#visualization-system)
12. [AI Chatbot](#ai-chatbot)
13. [Session State Variables](#session-state-variables)
14. [Configuration](#configuration)
15. [Dependencies](#dependencies)

---

## Overview

This platform provides end-to-end data analysis capabilities including:

- **Multi-file Excel loading** with sheet-level management
- **Intelligent data quality assessment** with proper column role detection
- **Power Query-like transformations** (Merge, Append, Custom Columns)
- **AI-powered business analysis** using local LLMs
- **Interactive visualizations** with Plotly
- **KPI generation with intelligent aggregation** (keys are NEVER summed)

### Key Principles

| Principle | Implementation |
|-----------|----------------|
| **Row-level duplicates only** | Only FULL ROW duplicates are flagged, not column value repetitions |
| **Keys are never summed** | Primary/Foreign Keys use COUNT or DISTINCT COUNT only |
| **Measures are aggregated** | Amount, Price, Revenue columns use SUM, AVG, etc. |
| **Categories have natural duplicates** | Brand, Category columns are NOT flagged for duplicates |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Ollama with at least one model installed

### Installation

```bash
# Clone and navigate
cd analysis-everything

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with a model
ollama pull qwen2.5:7b
ollama serve
```

### Run the Application

```bash
# Linux/Mac
./run.sh

# Windows
run.bat

# Or directly
streamlit run ui/streamlit/app.py
```

Access at: **http://localhost:8501**

---

## Project Structure

```
analysis-everything/
├── analysis/                    # Analytics Engine
│   ├── __init__.py
│   ├── advanced_analyzer.py     # Senior analyst deep analysis
│   ├── automatic_analyzer.py    # Complete automatic analysis pipeline
│   ├── business_intelligence.py # Business type inference & KPI detection
│   ├── data_quality.py          # Data quality assessment with column roles
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── kpi_intelligence.py      # Intelligent KPI generation
│   ├── report_generator.py      # PDF report generation
│   └── visualization.py         # Chart creation toolkit
│
├── models/                      # AI & Chatbot Models
│   ├── __init__.py
│   ├── dashboard_builder.py     # AI-assisted dashboard generation
│   ├── data_to_text.py          # DataFrame to natural language
│   ├── enhanced_chatbot.py      # Chatbot with auto visualization
│   ├── hybrid_chatbot.py        # Pandas agent + visualization hybrid
│   ├── llm_chatbot.py           # Direct LLM chatbot
│   ├── pandas_agent_chatbot.py  # Code execution on DataFrames
│   ├── rag_pipeline.py          # Retrieval-Augmented Generation
│   └── schema_analyzer.py       # Multi-table relationship detection
│
├── pipelines/                   # Data Processing
│   ├── __init__.py
│   └── cleaning.py              # Data cleaning & Power Query operations
│
├── app/                         # Data Loading
│   ├── __init__.py
│   ├── data_loader.py           # Excel file loading
│   └── multi_file_loader.py     # Multi-file/sheet management
│
├── ui/streamlit/                # User Interface
│   ├── app.py                   # Main Streamlit application (4094 lines)
│   └── streamlit_app.py         # Entry point wrapper
│
├── config/
│   └── settings.yml             # Application configuration
│
├── data/                        # Data storage folder
├── requirements.txt             # Python dependencies
├── run.sh                       # Linux/Mac startup script
└── run.bat                      # Windows startup script
```

---

## Application Pages

### 1. 🏠 Home

**Purpose:** Welcome page with feature overview and system status

**Components:**
- Feature overview cards
- Ollama connection status indicator
- Quick start guidance

---

### 2. 📤 Multi-File Loading

**Purpose:** Upload and manage multiple Excel files

**Features:**
| Feature | Description |
|---------|-------------|
| File Upload | Drag-and-drop or browse for `.xlsx`/`.xls` files |
| Sheet Preview | View all sheets in each uploaded file |
| Data Preview | Preview first N rows of any sheet |
| Column Info | View column names, types, and basic stats |

**User Flow:**
1. Click "Browse files" or drag Excel files
2. Select which sheets to load from each file
3. Preview data to verify correct loading
4. Proceed to analysis pages

---

### 3. 🔗 Schema Analysis

**Purpose:** Discover relationships between tables (ERD-style)

**Features:**
| Feature | Description |
|---------|-------------|
| Auto-detection | AI analyzes column names and data overlaps |
| Key Detection | Identifies primary and foreign key candidates |
| Relationship View | Visual display of table connections |
| Manual Override | Add/edit relationships manually |

**Detection Logic:**
- Analyzes column name patterns (`*_id`, `*Key`, etc.)
- Computes actual data value overlaps between columns
- Uses LLM to infer semantic relationships

---

### 4. 🎯 Business Goals

**Purpose:** Define business context for AI analysis

**Input Fields:**
| Field | Description | Example |
|-------|-------------|---------|
| Business Problem | The challenge being addressed | "Declining sales in Q4" |
| Objective | What you want to achieve | "Identify top-performing products" |
| Target Audience | Who will use the insights | "Sales leadership team" |

---

### 5. ⚠️ Data Quality

**Purpose:** Comprehensive data quality assessment

**Quality Score:**
- 0-100 scale based on detected issues
- Weighted by issue severity

**Issue Categories:**

| Category | Detection Logic | Severity Levels |
|----------|-----------------|-----------------|
| Missing Values | Per-column null percentage | >50% Critical, >25% High, >10% Medium, ≤10% Low |
| Duplicate Rows | **FULL ROW duplicates only** | Any duplicates = Medium |
| Outliers | IQR method on **MEASURE columns only** | Based on percentage |
| Type Mismatch | Non-numeric in numeric columns | High |
| High Cardinality | >90% unique in categorical | Medium |
| Consistency | Whitespace/case variations | Low |

**Critical Rule:** Column value repetitions (e.g., "Nike" appearing 1000 times in Brand column) are **NOT** flagged as duplicates.

**UI Components:**
- Overall quality score gauge
- Issues grouped by severity (Critical → Low)
- Expandable issue details
- Priority action list
- Column role summary display

---

### 6. 🧹 Data Cleaning

**Purpose:** Clean data and perform Power Query operations

**Tabs:**

#### Tab 1: File-by-File Cleaning

**Per-Dataset Operations:**
| Operation | Options | Notes |
|-----------|---------|-------|
| Handle Missing Values | drop, fill_mean, fill_median, fill_mode, fill_zero, forward_fill, backward_fill | Applied to selected columns |
| Remove Outliers | IQR, Z-Score | **Only on MEASURE columns** (keys excluded) |
| Encode Categorical | Label Encoding, One-Hot Encoding | Converts categories to numbers |
| Scale Numeric | Standard, MinMax | Standardize or normalize |
| Remove Duplicates | Full Row | **Only removes identical rows** |

**Column Role Display:**
```
📊 Column Roles Detected:
  🔑 Keys: CustomerID, OrderID, ProductKey
  📏 Measures: SalesAmount, Quantity, Price
  🏷️ Categories: Brand, Category, Region
```

#### Tab 2: Merge Queries (VLOOKUP)

**Purpose:** Join two tables on matching columns (like Excel VLOOKUP/XLOOKUP)

| Parameter | Description |
|-----------|-------------|
| Left Table | Primary table to keep all rows from |
| Right Table | Lookup table to bring values from |
| Left Key | Column in left table to match on |
| Right Key | Column in right table to match on |
| Join Type | left, right, inner, outer |

**Output:** Combined table with all left table rows + matched right table columns

#### Tab 3: Append Queries

**Purpose:** Stack tables vertically (UNION operation)

| Parameter | Description |
|-----------|-------------|
| Tables | Select 2+ tables to append |
| Ignore Index | Reset row numbers (recommended) |

**Output:** Single table with all rows from selected tables

#### Tab 4: Add Custom Columns

**Purpose:** Create calculated columns using expressions

| Parameter | Description |
|-----------|-------------|
| Target Dataset | Which dataset to modify |
| New Column Name | Name for the new column |
| Expression | Python/Pandas expression |

**Supported Expressions:**
```python
# Arithmetic
col['Price'] * col['Quantity']
col['Revenue'] - col['Cost']

# String operations
col['FirstName'] + ' ' + col['LastName']
col['Name'].str.upper()
col['Email'].str.contains('@')

# Conditional
np.where(col['Amount'] > 1000, 'High', 'Low')

# Date extraction
pd.to_datetime(col['Date']).dt.year
pd.to_datetime(col['Date']).dt.month_name()
```

---

### 7. 🤖 Strategic AI Analyst

**Purpose:** Full business analysis powered by AI

**Analysis Components:**
| Component | Description |
|-----------|-------------|
| Executive Summary | AI-generated business overview |
| Pattern Detection | Correlations, distributions, outliers |
| Business Insights | LLM-generated insights based on data |
| Recommendations | Actionable next steps |
| Visualizations | Auto-generated relevant charts |

**Output Sections:**
1. Overview Statistics
2. Business Context (industry, entities, KPIs)
3. Deep Pattern Analysis
4. Executive Summary
5. Action Plan

---

### 8. 📊 KPIs Dashboard

**Purpose:** Display and create KPIs with intelligent aggregation

#### Master KPIs Section

Automatically generates KPIs across all data with proper aggregation:

| Column Type | Allowed Functions | Forbidden Functions |
|-------------|-------------------|---------------------|
| Primary Key | COUNT, DISTINCT COUNT | SUM, AVERAGE, MIN, MAX |
| Foreign Key | COUNT, DISTINCT COUNT | SUM, AVERAGE, MIN, MAX |
| Identifier | COUNT, DISTINCT COUNT | SUM, AVERAGE, MIN, MAX |
| Measure | SUM, AVERAGE, MIN, MAX, MEDIAN | - |
| Category | COUNT, DISTINCT COUNT | SUM, AVERAGE |
| Date | MIN, MAX, COUNT | SUM, AVERAGE |

**KPI Card Display:**
```
┌─────────────────────────────────────┐
│  💰 Total Sales Revenue             │
│  ═══════════════════════════════════│
│  $1,234,567.89                      │
│  ─────────────────────────────────  │
│  Column: SalesAmount                │
│  Function: SUM                      │
│  Definition: Sum of all sales       │
│  amount values - represents total   │
│  monetary value                     │
└─────────────────────────────────────┘
```

#### Per-Dataset KPIs

For each loaded dataset:
- Auto-generates relevant KPIs
- Shows column role detection results
- Displays aggregation function used

#### Custom KPI Builder

| Field | Description |
|-------|-------------|
| Select Column | Choose from available columns |
| Aggregation Function | COUNT, DISTINCT_COUNT, SUM, AVERAGE, MIN, MAX, MEDIAN |
| Custom Name | Optional display name |

**Validation:**
- If user selects SUM on a key column, shows error:
  ```
  ⚠️ Cannot SUM column 'CustomerID' - it's a KEY column.
  Recommendation: Use DISTINCT COUNT instead.
  ```

---

### 9. 📊 Custom Dashboard

**Purpose:** Power BI-style dashboard builder

**Features:**
| Feature | Description |
|---------|-------------|
| Multiple Pages | Create separate dashboard pages |
| KPI Cards | Add metric cards with auto-formatting |
| Charts | Add various visualization types |
| Layout | Arrange components in grid |
| Pin Charts | Pin charts from other pages |

---

### 10. 📈 Visualization

**Purpose:** Create and explore visualizations

#### Smart Visualizations

Auto-generates appropriate charts based on data characteristics:
- Time series → Line charts with trends
- Categorical distribution → Bar/Pie charts
- Numeric correlation → Scatter plots, heatmaps
- Distribution analysis → Histograms, boxplots

#### Cross-Dataset Chart Builder

| Parameter | Description |
|-----------|-------------|
| Chart Type | Line, Bar, Pie, Scatter, Histogram, Box, Heatmap |
| Dataset | Select source dataset |
| X-Axis | Column for x-axis |
| Y-Axis | Column for y-axis (if applicable) |
| Color | Optional grouping column |
| Title | Chart title |

---

### 11. 💬 Enhanced Chatbot

**Purpose:** AI chatbot with code execution and auto-visualization

**Capabilities:**
| Capability | Description |
|------------|-------------|
| Natural Language Queries | Ask questions in plain English |
| Code Execution | Executes pandas code for accurate answers |
| Auto Visualization | Generates charts when relevant |
| Conversation History | Maintains context across messages |

**Example Queries:**
- "What are the top 5 products by revenue?"
- "Show me sales trend over time"
- "Which customers have the highest order count?"
- "Compare sales across regions"

---

### 12. 📄 Monthly Report

**Purpose:** Generate exportable reports

**Export Formats:**
| Format | Contents |
|--------|----------|
| PDF | Full formatted report with charts |
| Excel | Data tables with analysis results |
| Markdown | Text-based summary |

---

## Core Modules

### analysis/data_quality.py

#### ColumnRole Enum
```python
class ColumnRole(Enum):
    PRIMARY_KEY = "primary_key"    # Unique identifier for each row
    FOREIGN_KEY = "foreign_key"    # Reference to another table
    IDENTIFIER = "identifier"      # Non-unique ID (e.g., SKU)
    CATEGORY = "category"          # Categorical/dimension column
    DIMENSION = "dimension"        # Grouping column
    MEASURE = "measure"            # Numeric value for aggregation
    DATE = "date"                  # Date/datetime column
    TEXT = "text"                  # Free-form text
    UNKNOWN = "unknown"            # Could not determine
```

#### IntelligentColumnAnalyzer

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `detect_column_role` | `df: DataFrame, column: str` | `ColumnRole` | Detects role of single column |
| `get_all_column_roles` | `df: DataFrame` | `Dict[str, ColumnRole]` | Maps all columns to roles |
| `get_key_columns` | `df: DataFrame` | `Set[str]` | Returns key/identifier columns |
| `get_measure_columns` | `df: DataFrame` | `Set[str]` | Returns measure columns |
| `get_category_columns` | `df: DataFrame` | `Set[str]` | Returns category columns |

**Detection Patterns:**

```python
# Primary Key Detection
KEY_PATTERNS = [
    r'.*_id$',           # customer_id, order_id
    r'^id$',             # id
    r'.*key$',           # CustomerKey, ProductKey
    r'^key$',            # key
    r'.*code$',          # product_code
    r'^sku$',            # sku
    r'^barcode$',        # barcode
    r'.*number$',        # order_number
    r'^serial.*',        # serial_number
]

# Additional checks:
# - High uniqueness ratio (>95%)
# - Sequential integer pattern
# - No null values
```

#### DataQualityAssessor

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `assess_missing_values()` | `List[Dict]` | Missing value issues by column |
| `assess_duplicates()` | `List[Dict]` | **FULL ROW duplicates only** |
| `assess_outliers()` | `List[Dict]` | Outliers in **MEASURE columns only** |
| `assess_data_types()` | `List[Dict]` | Type validation issues |
| `assess_categorical_consistency()` | `List[Dict]` | Whitespace/case issues |
| `assess_relationships()` | `List[Dict]` | High correlation detection |
| `assess_all()` | `Dict` | Complete quality report |
| `get_column_role_summary()` | `Dict` | Summary of detected roles |

**Duplicate Detection Logic:**
```python
def assess_duplicates(self) -> List[Dict]:
    # Count FULL ROW duplicates only
    duplicate_mask = self.df.duplicated(keep=False)
    duplicate_count = duplicate_mask.sum()
    
    # Only flag if there are actual full-row duplicates
    if duplicate_count > 0:
        # Report as issue
    
    # DO NOT flag column-level duplicates
    # Categories naturally have repeated values
```

**Outlier Detection Logic:**
```python
def assess_outliers(self) -> List[Dict]:
    # Get MEASURE columns only
    measure_cols = IntelligentColumnAnalyzer.get_measure_columns(self.df)
    
    # Exclude key columns from outlier detection
    for col in self.df.select_dtypes(include=[np.number]).columns:
        if col not in measure_cols:
            continue  # Skip non-measure columns
        
        # Apply IQR method only to measures
```

---

### analysis/kpi_intelligence.py

#### AggregationFunction Enum
```python
class AggregationFunction(Enum):
    COUNT = "count"
    DISTINCT_COUNT = "distinctcount"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    FIRST = "first"
    LAST = "last"
```

#### KPIDefinition Dataclass
```python
@dataclass
class KPIDefinition:
    name: str                    # Display name
    column: str                  # Source column
    function: AggregationFunction # Aggregation function
    value: Any                   # Calculated value
    formatted_value: str         # Human-readable value
    business_definition: str     # What this KPI means
    column_role: str            # Detected role
    is_valid: bool              # Whether aggregation is valid
    warning: Optional[str]      # Any warnings
```

#### KPIColumnAnalyzer

**Critical Validation Rules:**

| Column Role | Valid Functions | Invalid Functions | Correction |
|-------------|-----------------|-------------------|------------|
| PRIMARY_KEY | COUNT, DISTINCT_COUNT | SUM, AVERAGE, MIN, MAX, MEDIAN | → DISTINCT_COUNT |
| FOREIGN_KEY | COUNT, DISTINCT_COUNT | SUM, AVERAGE, MIN, MAX, MEDIAN | → DISTINCT_COUNT |
| IDENTIFIER | COUNT, DISTINCT_COUNT | SUM, AVERAGE | → DISTINCT_COUNT |
| MEASURE | SUM, AVERAGE, MIN, MAX, MEDIAN, COUNT | - | - |
| CATEGORY | COUNT, DISTINCT_COUNT | SUM, AVERAGE | → DISTINCT_COUNT |
| DATE | MIN, MAX, COUNT, DISTINCT_COUNT | SUM, AVERAGE | → MAX |

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `detect_column_role` | `df, column` | `ColumnRole` | Detect column role |
| `get_valid_aggregations` | `role` | `List[AggregationFunction]` | Valid functions for role |
| `get_recommended_aggregation` | `df, column` | `(AggregationFunction, str)` | Best function + explanation |
| `is_aggregation_valid` | `df, column, aggregation` | `(bool, str)` | Validate + warning |

#### IntelligentKPIGenerator

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_column_role` | `column` | `ColumnRole` | Get role for column |
| `get_key_columns` | - | `List[str]` | All key columns |
| `get_measure_columns` | - | `List[str]` | All measure columns |
| `get_category_columns` | - | `List[str]` | All category columns |
| `generate_kpi` | `column, aggregation, custom_name` | `KPIDefinition` | Generate single KPI |
| `generate_all_kpis` | `max_kpis` | `List[KPIDefinition]` | Auto-generate KPIs |
| `get_column_summary` | - | `Dict` | Summary by role |

#### validate_kpi_request Function

```python
def validate_kpi_request(
    df: pd.DataFrame,
    column: str,
    requested_function: AggregationFunction
) -> Tuple[bool, AggregationFunction, str]:
    """
    Validate a KPI request and return corrected function if invalid.
    
    Returns:
        - is_valid: Whether the request is valid
        - corrected_function: The function to use (same if valid, corrected if not)
        - message: Explanation message
    """
```

**Example:**
```python
# User requests SUM on CustomerID
is_valid, corrected, message = validate_kpi_request(df, "CustomerID", AggregationFunction.SUM)
# Returns: (False, AggregationFunction.DISTINCT_COUNT, "Cannot SUM a key column...")
```

---

### pipelines/cleaning.py

#### IntelligentColumnDetector

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `is_key_column` | `df, column` | `bool` | Check if column is a key |
| `is_measure_column` | `df, column` | `bool` | Check if column is a measure |
| `get_key_columns` | `df` | `List[str]` | List of key columns |
| `get_measure_columns` | `df` | `List[str]` | List of measure columns |

#### DataCleaner

**Constructor:**
```python
cleaner = DataCleaner(df: pd.DataFrame)
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `handle_missing_values` | `strategy, columns` | `DataFrame` | Fill or drop missing values |
| `remove_outliers` | `method, columns` | `DataFrame` | Remove outliers (MEASURES only) |
| `encode_categorical` | `method, columns` | `DataFrame` | Encode categories |
| `scale_numeric` | `method, columns` | `DataFrame` | Scale numeric columns |
| `remove_duplicates` | `subset` | `DataFrame` | Remove **FULL ROW** duplicates |
| `get_cleaning_log` | - | `List[str]` | Get log of operations |
| `get_cleaned_df` | - | `DataFrame` | Get cleaned data |
| `reset` | - | `DataFrame` | Reset to original |

**Missing Value Strategies:**
```python
STRATEGIES = {
    'drop': 'Remove rows with missing values',
    'fill_mean': 'Fill with column mean (numeric only)',
    'fill_median': 'Fill with column median (numeric only)',
    'fill_mode': 'Fill with most frequent value',
    'fill_zero': 'Fill with zero',
    'forward_fill': 'Fill with previous row value',
    'backward_fill': 'Fill with next row value'
}
```

#### PowerQueryOperations

All methods are **static** and return modified DataFrames.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `merge_queries` | `left_df, right_df, left_on, right_on, how, suffixes` | `DataFrame` | VLOOKUP/JOIN |
| `append_queries` | `dfs: List[DataFrame], ignore_index` | `DataFrame` | Stack vertically (UNION) |
| `add_custom_column` | `df, new_column_name, expression` | `DataFrame` | Calculated column |
| `add_column_from_lookup` | `main_df, lookup_df, main_key, lookup_key, value_column, new_column_name` | `DataFrame` | XLOOKUP |
| `group_and_aggregate` | `df, group_by, aggregations` | `DataFrame` | Group By |
| `pivot_table` | `df, index, columns, values, aggfunc` | `DataFrame` | Pivot |
| `unpivot` | `df, id_vars, value_vars, var_name, value_name` | `DataFrame` | Melt/Unpivot |
| `split_column` | `df, column, delimiter, new_column_names` | `DataFrame` | Split by delimiter |
| `change_column_type` | `df, column, new_type` | `DataFrame` | Change type |
| `fill_down` | `df, columns` | `DataFrame` | Forward fill |
| `fill_up` | `df, columns` | `DataFrame` | Backward fill |
| `remove_duplicates_full_row` | `df` | `DataFrame` | Remove exact duplicates |
| `remove_duplicates_by_keys` | `df, key_columns, keep` | `DataFrame` | Dedupe by business key |

**Merge Types:**
```python
HOW_OPTIONS = {
    'left': 'Keep all rows from left table (VLOOKUP behavior)',
    'right': 'Keep all rows from right table',
    'inner': 'Keep only matching rows',
    'outer': 'Keep all rows from both tables'
}
```

---

### analysis/visualization.py

#### Visualizer

**Constructor:**
```python
visualizer = Visualizer(df: pd.DataFrame)
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `line_chart` | `x, y, title, color` | `plotly.Figure` | Line chart |
| `bar_chart` | `x, y, title, orientation, color` | `plotly.Figure` | Bar chart |
| `pie_chart` | `values, names, title` | `plotly.Figure` | Pie chart |
| `histogram` | `column, nbins, title` | `plotly.Figure` | Histogram |
| `boxplot` | `y, x, title` | `plotly.Figure` | Box plot |
| `scatter_plot` | `x, y, color, size, title, trendline` | `plotly.Figure` | Scatter plot |
| `heatmap` | `data, title` | `plotly.Figure` | Heatmap |
| `correlation_heatmap` | `title` | `plotly.Figure` | Correlation matrix |
| `distribution_plot` | `column, title` | `plotly.Figure` | Histogram + box marginal |
| `trend_plot` | `x, y, title` | `plotly.Figure` | Line with 7-day MA |
| `comparison_plot` | `columns, title` | `plotly.Figure` | Multi-column box |
| `multi_histogram` | `columns, title` | `plotly.Figure` | Subplot histograms |
| `get_chart_types` | - | `List[str]` | Available chart types |

---

## Intelligent Column Detection

### Detection Hierarchy

```
1. Name Pattern Matching
   ├── *_id, *Key, *code → PRIMARY_KEY / FOREIGN_KEY
   ├── *amount*, *price*, *revenue* → MEASURE
   └── *category*, *type*, *brand* → CATEGORY

2. Data Characteristics
   ├── >95% unique + no nulls → PRIMARY_KEY
   ├── Numeric + low cardinality → CATEGORY
   └── Numeric + high variance → MEASURE

3. Column Position (fallback)
   └── First column often ID
```

### Key Pattern Matching (Regex)

```python
KEY_PATTERNS = [
    r'.*_id$',              # customer_id, order_id
    r'^id$',                # id
    r'.*key$',              # CustomerKey
    r'^key$',               # key
    r'.*code$',             # product_code
    r'^sku$',               # sku
    r'^barcode$',           # barcode
    r'.*number$',           # order_number
    r'^serial.*',           # serial_number
    r'^index$',             # index
]

MEASURE_PATTERNS = [
    r'.*amount.*',          # SalesAmount
    r'.*price.*',           # UnitPrice
    r'.*cost.*',            # TotalCost
    r'.*revenue.*',         # Revenue
    r'.*quantity.*',        # Quantity
    r'.*total.*',           # Total
    r'.*sum.*',             # Sum
    r'.*value.*',           # Value
    r'.*sales.*',           # Sales
    r'.*profit.*',          # Profit
]

CATEGORY_PATTERNS = [
    r'.*category.*',        # Category
    r'.*type.*',            # Type
    r'.*status.*',          # Status
    r'.*brand.*',           # Brand
    r'.*region.*',          # Region
    r'.*country.*',         # Country
    r'.*segment.*',         # Segment
    r'.*group.*',           # Group
]
```

---

## Data Quality Assessment

### Severity Levels

```python
class Severity(Enum):
    LOW = "low"           # Minor issues, informational
    MEDIUM = "medium"     # Should be addressed
    HIGH = "high"         # Significant data issues
    CRITICAL = "critical" # Must be fixed before analysis
```

### Issue Types and Thresholds

| Issue Type | Critical | High | Medium | Low |
|------------|----------|------|--------|-----|
| Missing Values | >50% | >25% | >10% | ≤10% |
| Full Row Duplicates | N/A | N/A | Any | - |
| Outliers | >30% | >20% | >10% | <10% |
| Type Mismatch | - | Any | - | - |
| High Cardinality | - | - | >90% unique | - |

### Quality Score Calculation

```python
def calculate_quality_score(issues: List[Dict]) -> int:
    """
    Calculate overall quality score (0-100)
    """
    base_score = 100
    
    for issue in issues:
        severity = issue['severity']
        deduction = {
            'critical': 25,
            'high': 15,
            'medium': 5,
            'low': 2
        }.get(severity, 0)
        base_score -= deduction
    
    return max(0, base_score)
```

---

## Data Cleaning Operations

### Operation Details

#### Handle Missing Values

```python
cleaner.handle_missing_values(
    strategy='fill_mean',  # Options: drop, fill_mean, fill_median, fill_mode, fill_zero, forward_fill, backward_fill
    columns=['Amount', 'Price']  # Specific columns or None for all
)
```

#### Remove Outliers

```python
# Only applied to MEASURE columns
cleaner.remove_outliers(
    method='iqr',  # Options: iqr, zscore
    columns=['SalesAmount']  # Must be measure columns
)

# IQR Method:
# Q1 = 25th percentile
# Q3 = 75th percentile
# IQR = Q3 - Q1
# Lower bound = Q1 - 1.5 * IQR
# Upper bound = Q3 + 1.5 * IQR

# Z-Score Method:
# Remove if |z-score| > 3
```

#### Remove Duplicates

```python
# ONLY removes full row duplicates
cleaner.remove_duplicates()

# Or by specific columns (business key deduplication)
PowerQueryOperations.remove_duplicates_by_keys(
    df,
    key_columns=['CustomerID', 'OrderDate'],
    keep='first'  # Options: first, last, False (remove all)
)
```

---

## Power Query Operations

### Merge Queries (VLOOKUP)

```python
# Like Excel VLOOKUP: bring data from one table to another
result = PowerQueryOperations.merge_queries(
    left_df=orders_df,
    right_df=customers_df,
    left_on='CustomerID',
    right_on='CustomerID',
    how='left',  # Keep all orders, bring in customer data
    suffixes=('', '_customer')
)
```

### Append Queries (UNION)

```python
# Stack multiple tables vertically
result = PowerQueryOperations.append_queries(
    dfs=[jan_sales, feb_sales, mar_sales],
    ignore_index=True  # Reset row numbers
)
```

### Add Custom Column

```python
# Create calculated columns
result = PowerQueryOperations.add_custom_column(
    df=sales_df,
    new_column_name='TotalAmount',
    expression="col['Quantity'] * col['UnitPrice']"
)

# Supported expression variables:
# - col: Reference to DataFrame (e.g., col['ColumnName'])
# - np: NumPy library
# - pd: Pandas library
```

### Group and Aggregate

```python
result = PowerQueryOperations.group_and_aggregate(
    df=sales_df,
    group_by=['Region', 'Category'],
    aggregations={
        'SalesAmount': 'sum',
        'Quantity': 'sum',
        'OrderID': 'count'  # Count of orders
    }
)
```

### Pivot Table

```python
result = PowerQueryOperations.pivot_table(
    df=sales_df,
    index='Region',
    columns='Category',
    values='SalesAmount',
    aggfunc='sum'
)
```

### Unpivot (Melt)

```python
# Convert wide format to long format
result = PowerQueryOperations.unpivot(
    df=wide_df,
    id_vars=['Region', 'Year'],
    value_vars=['Q1', 'Q2', 'Q3', 'Q4'],
    var_name='Quarter',
    value_name='Sales'
)
```

---

## KPI Intelligence

### Aggregation Selection Logic

```python
def get_recommended_aggregation(df, column) -> Tuple[AggregationFunction, str]:
    role = detect_column_role(df, column)
    
    if role in [ColumnRole.PRIMARY_KEY, ColumnRole.FOREIGN_KEY]:
        return (AggregationFunction.DISTINCT_COUNT, 
                "Key columns should be counted, not summed")
    
    if role == ColumnRole.MEASURE:
        # Check if column represents counts vs. amounts
        if 'count' in column.lower() or 'qty' in column.lower():
            return (AggregationFunction.SUM, "Sum of quantities")
        return (AggregationFunction.SUM, "Total of all values")
    
    if role == ColumnRole.CATEGORY:
        return (AggregationFunction.DISTINCT_COUNT,
                "Count of unique categories")
    
    if role == ColumnRole.DATE:
        return (AggregationFunction.MAX, "Most recent date")
```

### KPI Generation Example

```python
generator = IntelligentKPIGenerator(df)

# Auto-generate all KPIs
kpis = generator.generate_all_kpis(max_kpis=20)

for kpi in kpis:
    print(f"""
    {kpi.name}
    ═══════════════════
    Value: {kpi.formatted_value}
    Column: {kpi.column}
    Function: {kpi.function.value.upper()}
    Definition: {kpi.business_definition}
    """)

# Output:
# Total Customers
# ═══════════════════
# Value: 1,234
# Column: CustomerID
# Function: DISTINCT COUNT
# Definition: Count of unique CustomerID values
```

---

## Visualization System

### Chart Type Selection Guide

| Data Type | Recommended Charts |
|-----------|-------------------|
| Time Series | Line Chart, Trend Plot |
| Categorical | Bar Chart, Pie Chart |
| Distribution | Histogram, Box Plot |
| Correlation | Scatter Plot, Heatmap |
| Comparison | Grouped Bar, Multi-Histogram |

### Auto-Visualization Logic

```python
def suggest_visualizations(df):
    suggestions = []
    
    # Time series detection
    date_cols = df.select_dtypes(include=['datetime']).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        suggestions.append({
            'type': 'line_chart',
            'x': date_cols[0],
            'y': numeric_cols[0],
            'title': f'{numeric_cols[0]} over Time'
        })
    
    # Categorical distribution
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols[:3]:
        if df[col].nunique() <= 10:
            suggestions.append({
                'type': 'pie_chart',
                'values': col,
                'title': f'{col} Distribution'
            })
    
    return suggestions
```

---

## AI Chatbot

### Chatbot Types

| Type | File | Use Case |
|------|------|----------|
| LLM Chatbot | `llm_chatbot.py` | Direct LLM with data context |
| Enhanced Chatbot | `enhanced_chatbot.py` | Auto-generates visualizations |
| Hybrid Chatbot | `hybrid_chatbot.py` | Code execution + visualization |
| Pandas Agent | `pandas_agent_chatbot.py` | Accurate code-based answers |

### Hybrid Chatbot Flow

```
User Question
     │
     ▼
┌─────────────────────┐
│ Detect Query Type   │
│ (calculation/viz)   │
└─────────────────────┘
     │
     ├──► Calculation ──► Execute Pandas Code ──► Format Answer
     │
     └──► Visualization ──► Generate Chart ──► Display
```

### Example Interactions

```
User: "What is the total sales by region?"

Chatbot:
1. Detects: calculation + visualization
2. Executes: df.groupby('Region')['Sales'].sum()
3. Generates: Bar chart of sales by region
4. Returns: "Total sales by region: North $1.2M, South $900K..."
   + Interactive bar chart
```

---

## Session State Variables

| Variable | Type | Description |
|----------|------|-------------|
| `df` | `DataFrame` | Current active dataset |
| `original_df` | `DataFrame` | Original unchanged data |
| `selected_datasets` | `Dict[str, DataFrame]` | Multiple selected datasets |
| `cleaned_datasets` | `Dict[str, DataFrame]` | Cleaned versions of datasets |
| `multi_file_loader` | `MultiFileLoader` | File management instance |
| `schema_analyzer` | `SchemaAnalyzer` | Relationship detection instance |
| `business_goals` | `Dict` | Problem, objective, target |
| `pinned_charts` | `List[Dict]` | Charts pinned to dashboard |
| `chat_history` | `List[Dict]` | Chatbot conversation history |
| `chatbot` | `HybridChatbot` | Chatbot instance |
| `dashboard_pages` | `List[Dict]` | Custom dashboard pages |

---

## Configuration

### settings.yml

```yaml
app:
  name: "Data Analysis & AI Chatbot"
  version: "0.1.0"
  debug: true

data_sources:
  enabled:
    - excel

excel:
  supported_formats:
    - .xlsx
    - .xls
  max_file_size_mb: 100

llm:
  provider: "ollama"
  model: "qwen2.5:7b"           # Default model
  embedding_model: "nomic-embed-text:latest"
  temperature: 0.5
  max_tokens: 1024

rag:
  chunk_size: 500
  chunk_overlap: 50
  similarity_threshold: 0.5
  top_k: 5

database:
  chroma_db_path: "./data/chroma_db"
  persist: true

ui:
  theme: "light"
  page_icon: "📊"
  layout: "wide"
```

### Supported LLM Models

| Model | Size | Best For |
|-------|------|----------|
| `qwen2.5:7b` | 7B | General purpose (default) |
| `llama3.2` | 8B | Fast responses |
| `mistral` | 7B | Code generation |
| `phi3` | 3.8B | Lightweight tasks |
| `gemma2` | 9B | Detailed analysis |

---

## Dependencies

```
streamlit==1.38.0
pandas==2.2.3
numpy==1.26.4
openpyxl==3.1.5
plotly==5.24.1
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.5.2
langchain==0.3.7
langchain-community==0.3.7
chromadb==0.5.15
sentence-transformers==3.2.1
ollama==0.3.1
pyyaml==6.0.2
python-dotenv==1.0.1
```

---

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Cannot SUM column X" | Attempting to sum a key column | Use COUNT or DISTINCT COUNT |
| "No data loaded" | No Excel files uploaded | Upload files on Multi-File Loading page |
| "Ollama connection failed" | Ollama not running | Run `ollama serve` in terminal |
| "Model not found" | Model not installed | Run `ollama pull qwen2.5:7b` |

### Validation Messages

```python
# Key column aggregation attempt
"⚠️ Cannot SUM column 'CustomerID' - it's a KEY column. 
Keys represent unique identifiers and should not be summed.
Recommendation: Use DISTINCT COUNT instead."

# Missing data warning
"⚠️ Column 'Email' has 45% missing values (High severity).
Recommendation: Fill with default value or investigate data source."

# Full row duplicates
"⚠️ Found 123 duplicate rows (exact matches across all columns).
Recommendation: Review and remove if not intentional."
```

---

## License

MIT License

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following existing patterns
4. Test with sample data
5. Submit pull request

---

*Documentation generated from codebase analysis - January 2026*
# analysis_chatbot
