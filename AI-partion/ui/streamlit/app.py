import sys
from pathlib import Path

# IMPORTANT: Fix sys.path to prevent app.py from shadowing the app package
# This must happen before any app.* imports
project_root = Path(__file__).resolve().parent.parent.parent
script_dir = Path(__file__).resolve().parent

# Remove ALL occurrences of script directory from sys.path
sys.path = [p for p in sys.path if Path(p).resolve() != script_dir]

# Ensure project root is at the front
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import os
import re

# Import from modules
from app.data_loader import ExcelLoader
from app.multi_file_loader import MultiFileLoader
from analysis.eda import EDAAnalyzer
from analysis.business_intelligence import BusinessIntelligence
from analysis.data_quality import DataQualityAssessor, IntelligentColumnAnalyzer, ColumnRole
from analysis.automatic_analyzer import AutomaticAnalyzer
from analysis.kpi_intelligence import IntelligentKPIGenerator, KPIColumnAnalyzer, AggregationFunction, validate_kpi_request
from pipelines.cleaning import DataCleaner, PowerQueryOperations, IntelligentColumnDetector
from analysis.visualization import Visualizer
from models.data_to_text import DataToText
from models.llm_chatbot import LLMChatbot
from models.enhanced_chatbot import EnhancedChatbot
from models.hybrid_chatbot import HybridChatbot
from models.schema_analyzer import SchemaAnalyzer
from analysis.advanced_analyzer import AdvancedAnalyzer
from models.dashboard_builder import DashboardBuilder
import plotly.express as px
import plotly.graph_objects as go

# Helper function to get data (prefers cleaned version over original)
def get_dataset(file_name: str, sheet_name: str) -> pd.DataFrame:
    """
    Get dataset - returns cleaned version if available, otherwise original.
    This ensures all pages use cleaned data after Data Cleaning stage.
    """
    dataset_key = f"{file_name} → {sheet_name}"
    # Also try without arrow for single-sheet files
    simple_key = file_name
    
    # Check for cleaned version first
    if st.session_state.get('cleaned_datasets'):
        if dataset_key in st.session_state.cleaned_datasets:
            return st.session_state.cleaned_datasets[dataset_key].copy()
        if simple_key in st.session_state.cleaned_datasets:
            return st.session_state.cleaned_datasets[simple_key].copy()
    
    # Fall back to original
    if st.session_state.multi_file_loader:
        df = st.session_state.multi_file_loader.get_sheet_data(file_name, sheet_name)
        if df is not None:
            return df.copy()
    
    return None

def get_all_datasets() -> dict:
    """
    Get all datasets for analysis.
    🎯 GOLDEN RULE: If pipeline final dataset exists, return ONLY that.
    Otherwise, returns cleaned versions where available, original otherwise.
    """
    # GOLDEN RULE: If final dataset from Data Cleaning pipeline exists, use ONLY that
    if st.session_state.get('pipeline_final_dataset') is not None:
        final_name = st.session_state.get('pipeline_final_dataset_name', 'Cleaned_Data')
        return {final_name: st.session_state.pipeline_final_dataset.copy()}
    
    # Otherwise, return available cleaned/original datasets
    all_datasets = {}
    if not st.session_state.multi_file_loader:
        return all_datasets
    
    # First check if there's a _final_dataset marker (from cleaning pipeline)
    if st.session_state.get('cleaned_datasets') and '_final_dataset' in st.session_state.cleaned_datasets:
        return {'Cleaned_Data': st.session_state.cleaned_datasets['_final_dataset'].copy()}
    
    # Otherwise build from available sources
    for file_name in st.session_state.multi_file_loader.get_loaded_files():
        sheets = st.session_state.multi_file_loader.get_file_sheets(file_name)
        for sheet in sheets:
            dataset_name = f"{file_name} → {sheet}" if len(sheets) > 1 else file_name
            
            # Use cleaned version if available
            if st.session_state.get('cleaned_datasets') and dataset_name in st.session_state.cleaned_datasets:
                all_datasets[dataset_name] = st.session_state.cleaned_datasets[dataset_name].copy()
            else:
                df = st.session_state.multi_file_loader.get_sheet_data(file_name, sheet)
                if df is not None:
                    all_datasets[dataset_name] = df.copy()
    
    return all_datasets



# Helper function to get the selected color palette
def get_color_palette():
    """
    Get the selected color palette from session state.
    Returns a list of colors that can be used in visualizations.
    🎯 GOLDEN RULE: Returns the EXACT palette selected by user in Multi-File Loading page
    """
    color_palettes = {
        'Vibrancy': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'],
        'Ocean': ['#0066CC', '#0099FF', '#00CCFF', '#006699', '#3399FF', '#66CCFF', '#0052A3', '#004D99'],
        'Sunset': ['#FF6B35', '#F7931E', '#FDB833', '#F37335', '#C1272D', '#F15A24', '#E74C3C', '#EC7063'],
        'Forest': ['#2D5016', '#3E7C17', '#6FA876', '#A4C639', '#52B788', '#40916C', '#2D6A4F', '#1B4332']
    }
    
    # Get user's selected palette from session state (set in Multi-File Loading page)
    selected = st.session_state.get('selected_color_palette', 'Vibrancy')
    return color_palettes.get(selected, color_palettes['Vibrancy'])

def get_palette_color_scale():
    """
    Get color scale for continuous data (like color_continuous_scale in Plotly).
    Returns a palette that works well with Plotly's color_continuous_scale parameter.
    """
    palette = st.session_state.get('selected_color_palette', 'Vibrancy')
    palette_scales = {
        'Vibrancy': 'Turbo',      # Vibrant continuous scale
        'Ocean': 'Blues',          # Ocean blues
        'Sunset': 'Reds',          # Sunset reds
        'Forest': 'Greens'         # Forest greens
    }
    return palette_scales.get(palette, 'Turbo')

# Helper function to apply color palette to Plotly figure
def apply_color_palette_to_figure(fig, palette_colors=None):
    """
    Apply the selected color palette to a Plotly figure.
    This ensures consistent colors across all visualizations.
    
    Args:
        fig: Plotly figure object
        palette_colors: Optional custom color list. If None, uses selected palette.
    
    Returns:
        Updated Plotly figure with applied colors
    """
    if fig is None:
        return fig
    
    if palette_colors is None:
        palette_colors = get_color_palette()
    
    # Update the color palette for bar charts, line charts, etc.
    for idx, trace in enumerate(fig.data):
        if idx < len(palette_colors):
            # For different trace types, apply color appropriately
            if hasattr(trace, 'marker'):
                trace.marker.color = palette_colors[idx]
            if hasattr(trace, 'line'):
                trace.line.color = palette_colors[idx]
    
    return fig

def create_bar_chart_with_palette(data, x, y, title="", color_col=None, **kwargs):
    """
    Create a bar chart with the selected color palette applied automatically.
    Wrapper around px.bar() to ensure consistent palette usage.
    """
    palette = get_color_palette()
    color_scale = get_palette_color_scale()
    
    # Create base figure
    if color_col and color_col in data.columns:
        fig = px.bar(data, x=x, y=y, title=title, color=color_col, 
                     color_continuous_scale=color_scale, **kwargs)
    else:
        fig = px.bar(data, x=x, y=y, title=title, **kwargs)
        # Apply discrete palette to traces
        fig = apply_color_palette_to_figure(fig, palette)
    
    return fig

def create_line_chart_with_palette(data, x, y, title="", **kwargs):
    """
    Create a line chart with the selected color palette applied automatically.
    Wrapper around px.line() to ensure consistent palette usage.
    """
    palette = get_color_palette()
    fig = px.line(data, x=x, y=y, title=title, **kwargs)
    fig = apply_color_palette_to_figure(fig, palette)
    return fig

def create_pie_chart_with_palette(data=None, values=None, names=None, title="", **kwargs):
    """
    Create a pie chart with the selected color palette applied automatically.
    Wrapper around px.pie() to ensure consistent palette usage.
    """
    palette = get_color_palette()
    fig = px.pie(values=values, names=names, title=title, 
                 color_discrete_sequence=palette, **kwargs)
    return fig

# Helper function for dataset selection with multi-select
def render_dataset_selector(page_key, allow_multi=True, show_select_all=False):
    """Render dataset selector - keeps datasets separate when multiple selected"""
    if not st.session_state.multi_file_loader or len(st.session_state.multi_file_loader.get_loaded_files()) == 0:
        return False
    
    st.markdown("---")
    st.subheader("📁 Select Dataset(s)")
    
    # Show cleaned datasets indicator
    cleaned_count = len(st.session_state.get('cleaned_datasets', {}))
    if cleaned_count > 0:
        st.success(f"✨ **{cleaned_count} dataset(s) have been cleaned.** Cleaned versions will be used automatically.")
    
    # Get all available datasets
    available_datasets = []
    for file_name in st.session_state.multi_file_loader.get_loaded_files():
        sheets = st.session_state.multi_file_loader.get_file_sheets(file_name)
        for sheet in sheets:
            dataset_name = f"{file_name} → {sheet}" if len(sheets) > 1 else file_name
            # Add indicator if cleaned
            is_cleaned = dataset_name in st.session_state.get('cleaned_datasets', {})
            display = f"✨ {dataset_name} (cleaned)" if is_cleaned else dataset_name
            available_datasets.append({
                'display_name': display,
                'file_name': file_name,
                'sheet_name': sheet,
                'dataset_key': dataset_name,
                'is_cleaned': is_cleaned
            })
    
    if not available_datasets:
        return False
    
    # Handle Select All button functionality
    select_all_key = f"select_all_{page_key}"
    if select_all_key not in st.session_state:
        st.session_state[select_all_key] = False
    
    # Create multi-selection or single selection
    if allow_multi:
        # Add Select All button if enabled
        if show_select_all and len(available_datasets) > 1:
            btn_col1, btn_col2 = st.columns([1, 4])
            with btn_col1:
                if st.button("📋 Select All Sheets", key=f"select_all_btn_{page_key}", use_container_width=True):
                    st.session_state[select_all_key] = True
                    st.rerun()
        
        col1, col2 = st.columns([4, 1])
        with col1:
            # Check if we should select all
            if st.session_state.get(select_all_key, False):
                default_selection = list(range(len(available_datasets)))
                st.session_state[select_all_key] = False  # Reset flag
            else:
                # Get current selection or default to first
                current_key = f"dataset_selector_{page_key}"
                if current_key in st.session_state:
                    default_selection = st.session_state[current_key]
                else:
                    default_selection = [0]
            
            selected_datasets = st.multiselect(
                "Choose dataset(s) - multiple selection keeps them separate:",
                range(len(available_datasets)),
                format_func=lambda i: available_datasets[i]['display_name'],
                default=default_selection,
                key=f"dataset_selector_{page_key}"
            )
        
        with col2:
            load_button = st.button("🔄 Load", use_container_width=True, key=f"load_{page_key}")
    else:
        # Single selection mode - add Select All to load all one by one
        if show_select_all and len(available_datasets) > 1:
            btn_col1, btn_col2 = st.columns([1, 4])
            with btn_col1:
                if st.button("📋 Load All Sheets", key=f"load_all_btn_{page_key}", use_container_width=True):
                    # Load ALL datasets when this button is clicked
                    datasets_dict = {}
                    names = []
                    for idx, ds in enumerate(available_datasets):
                        df_original = st.session_state.multi_file_loader.get_sheet_data(
                            ds['file_name'], 
                            ds['sheet_name']
                        )
                        if df_original is not None:
                            dataset_key = ds['display_name']
                            if st.session_state.get('cleaned_datasets') and dataset_key in st.session_state.cleaned_datasets:
                                datasets_dict[dataset_key] = st.session_state.cleaned_datasets[dataset_key].copy()
                            else:
                                datasets_dict[dataset_key] = df_original.copy()
                            names.append(dataset_key)
                    
                    if datasets_dict:
                        st.session_state.selected_datasets = datasets_dict
                        st.session_state.dataset_names = names
                        # Set first one as active df for compatibility
                        st.session_state.df = list(datasets_dict.values())[0].copy()
                        st.session_state.original_df = st.session_state.df.copy()
                        st.success(f"✅ Loaded ALL {len(datasets_dict)} datasets!")
                        st.rerun()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_idx = st.selectbox(
                "Choose a dataset:",
                range(len(available_datasets)),
                format_func=lambda i: available_datasets[i]['display_name'],
                key=f"dataset_selector_{page_key}"
            )
            selected_datasets = [selected_idx]
        
        with col2:
            load_button = st.button("🔄 Load", use_container_width=True, key=f"load_{page_key}")
    
    if load_button and selected_datasets:
        datasets_dict = {}
        names = []
        
        for idx in selected_datasets:
            selected = available_datasets[idx]
            # Get original data first
            df_original = st.session_state.multi_file_loader.get_sheet_data(
                selected['file_name'], 
                selected['sheet_name']
            )
            if df_original is not None:
                dataset_key = selected['display_name']
                # Use cleaned version if available, otherwise use original
                if st.session_state.get('cleaned_datasets') and dataset_key in st.session_state.cleaned_datasets:
                    datasets_dict[dataset_key] = st.session_state.cleaned_datasets[dataset_key].copy()
                else:
                    datasets_dict[dataset_key] = df_original.copy()
                names.append(dataset_key)
        
        if datasets_dict:
            if len(datasets_dict) == 1:
                # Single dataset
                st.session_state.df = list(datasets_dict.values())[0].copy()
                st.session_state.original_df = st.session_state.df.copy()
                st.session_state.selected_datasets = datasets_dict
                st.session_state.dataset_names = names
                st.success(f"✅ Loaded: {names[0]}")
            else:
                # Multiple datasets - keep them separate!
                st.session_state.selected_datasets = datasets_dict
                st.session_state.dataset_names = names
                # Set first one as active df for compatibility
                st.session_state.df = list(datasets_dict.values())[0].copy()
                st.session_state.original_df = st.session_state.df.copy()
                st.success(f"✅ Loaded {len(datasets_dict)} datasets: {', '.join(names)}")
                st.info("💡 Datasets are kept separate - each will be processed individually")
            
            # Clear cache for specific pages
            keys_to_clear = ['cleaned_df', 'cleaned_datasets', 'quality_issues', 'complete_analysis', 'automatic_analysis']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.rerun()
    
    # Show current datasets info
    if 'selected_datasets' in st.session_state and st.session_state.selected_datasets:
        datasets = st.session_state.selected_datasets
        if len(datasets) == 1:
            name = list(datasets.keys())[0]
            df = list(datasets.values())[0]
            st.info(f"📊 Dataset: **{name}** - {df.shape[0]} rows × {df.shape[1]} columns")
        else:
            st.info(f"📊 **{len(datasets)} Datasets Loaded** - Processing each separately")
            with st.expander("📋 View Datasets Details"):
                for name, df in datasets.items():
                    st.write(f"- **{name}**: {df.shape[0]} rows × {df.shape[1]} columns")
    
    st.markdown("---")
    return True

def render_pin_button(title, figure, key_suffix, item_type="chart"):
    """Render a Pin to Dashboard button for visualizations
    
    IMPORTANT: Pinning a chart only saves UI state + configuration, 
    it does NOT trigger re-analysis or regeneration of visualizations.
    """
    # Use a callback to prevent re-renders on button click
    def pin_chart():
        # Check if chart is already pinned to prevent duplicates
        chart_ids = [f"{c.get('title')}_{c.get('type')}" for c in st.session_state.pinned_charts]
        chart_id = f"{title}_{item_type}"
        
        if chart_id not in chart_ids:
            st.session_state.pinned_charts.append({
                "title": title,
                "figure": figure,
                "type": item_type
            })
            st.session_state.show_pin_success = (title, item_type)
            # CRITICAL: Mark that a chart was pinned, but do NOT trigger re-analysis
            st.session_state.chart_pinned_this_session = True
        else:
            st.session_state.show_pin_warning = title
    
    # Use on_click callback to prevent re-render issues
    st.button(
        f"📌 Pin to Dashboard", 
        key=f"pin_{key_suffix}",
        on_click=pin_chart,
        use_container_width=False
    )
    
    # Show success message from session state if exists
    if st.session_state.get('show_pin_success'):
        title_shown, item_type_shown = st.session_state.show_pin_success
        st.success(f"📌 Pinned '{title_shown}' to dashboard!")
        st.session_state.show_pin_success = None
    
    if st.session_state.get('show_pin_warning'):
        st.info(f"ℹ️ '{st.session_state.show_pin_warning}' is already pinned!")
        st.session_state.show_pin_warning = None

def compute_dataset_hash():
    """
    Compute a hash of all loaded datasets to detect when data changes.
    Used to determine if visualizations need to be regenerated.
    
    CRITICAL RULE: Re-analysis is allowed ONLY if the dataset itself changes.
    """
    all_datasets = get_all_datasets()
    if not all_datasets:
        return None
    
    try:
        # Create a stable hash based on:
        # - Number of datasets
        # - Shape of each dataset (rows × columns)
        # - Data types
        hash_components = []
        for name in sorted(all_datasets.keys()):
            df = all_datasets[name]
            hash_components.append(f"{name}_{df.shape[0]}_{df.shape[1]}_{'_'.join(map(str, df.dtypes))}")
        
        import hashlib
        combined = "_".join(hash_components)
        return hashlib.md5(combined.encode()).hexdigest()
    except:
        return None

def should_regenerate_visualizations():
    """
    Determine if visualizations should be regenerated.
    
    GOLDEN RULE FOR VISUALIZATION PAGE:
    ✅ DO regenerate if: dataset itself changed (shape, columns, data)
    ❌ DON'T regenerate if:
       - A chart is pinned (pinning = save only, no re-analysis)
       - A column/file is selected in Custom Chart Builder (reuse existing analysis)
       - Only UI state changed (chart selection, filter, etc.)
    """
    # Check if pinning happened - if so, don't regenerate
    if st.session_state.chart_pinned_this_session:
        st.session_state.chart_pinned_this_session = False  # Reset for next check
        return False
    
    # Check if only column/file selection changed (not dataset)
    current_selection = {
        "chart_type": st.session_state.custom_chart_builder_selection.get("chart_type"),
        "use_cross_dataset": st.session_state.custom_chart_builder_selection.get("use_cross_dataset"),
        "selected_dataset": st.session_state.custom_chart_builder_selection.get("selected_dataset"),
        "x_col": st.session_state.custom_chart_builder_selection.get("x_col"),
        "y_col": st.session_state.custom_chart_builder_selection.get("y_col")
    }
    
    # Get previous selection
    previous_key = "custom_chart_builder_selection_prev"
    if previous_key in st.session_state:
        if st.session_state[previous_key] == current_selection:
            # Selection hasn't changed - don't regenerate
            return False
    
    # Store current selection for next comparison
    st.session_state[previous_key] = current_selection.copy()
    
    # Check if DATASET changed (most important check)
    current_hash = compute_dataset_hash()
    previous_hash = st.session_state.viz_dataset_state
    
    # If dataset changed, regenerate
    if previous_hash is not None and current_hash != previous_hash:
        st.session_state.viz_dataset_state = current_hash
        return True
    
    # Store hash for next comparison
    if current_hash is not None:
        st.session_state.viz_dataset_state = current_hash
    
    # If this is first run and no cached analysis, regenerate
    if previous_hash is None and 'viz_all_generated' not in st.session_state:
        return True
    
    # Default: don't regenerate without explicit user action
    return False

def render_eda_page():
    """Detailed Exploratory Data Analysis page"""
    st.title("🔍 Data Understanding & EDA")
    st.markdown("### Deep dive into your data structure and patterns")
    
    render_dataset_selector("eda", allow_multi=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first")
        return

    datasets = st.session_state.selected_datasets
    
    for name, df in datasets.items():
        st.markdown(f"## 📁 {name}")
        eda = EDAAnalyzer(df)
        viz = Visualizer(df)
        
        tabs = st.tabs(["📊 Statistics", "🧹 Missing Values", "🔗 Correlations", "❗ Outliers", "🏷️ Categorical"])
        
        with tabs[0]:
            st.subheader("Statistical Summary")
            st.dataframe(eda.get_numerical_stats(), use_container_width=True)
            
            st.subheader("Data Overview")
            summary = eda.get_summary_report()
            cols = st.columns(3)
            cols[0].metric("Rows", summary['rows'])
            cols[1].metric("Columns", summary['columns'])
            cols[2].metric("Duplicates", summary['duplicates'])
            
        with tabs[1]:
            st.subheader("Missing Values Analysis")
            missing_df = eda.get_missing_values()
            st.dataframe(missing_df, use_container_width=True)
            
            if missing_df['missing_count'].sum() > 0:
                fig = px.bar(missing_df[missing_df['missing_count'] > 0], 
                            x='column', y='missing_count', title="Missing Values per Column")
                st.plotly_chart(fig, use_container_width=True)
                render_pin_button(f"Missing Values: {name}", fig, f"eda_missing_{name}")
            else:
                st.success("✨ No missing values detected!")
                
        with tabs[2]:
            st.subheader("Correlation Matrix")
            corr = eda.get_correlation_matrix()
            if not corr.empty:
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
                render_pin_button(f"Correlations: {name}", fig, f"eda_corr_{name}")
                
                high_corr = eda.get_high_correlations()
                if high_corr:
                    st.write("**High Correlations Detected:**")
                    for c in high_corr:
                        st.write(f"- `{c['col1']}` & `{c['col2']}`: **{c['correlation']:.2f}**")
            else:
                st.info("Not enough numeric columns for correlation analysis")
                
        with tabs[3]:
            st.subheader("Outlier Detection (IQR Method)")
            outliers = eda.detect_outliers()
            if outliers:
                outlier_data = []
                for col, stats in outliers.items():
                    if stats['count'] > 0:
                        outlier_data.append({
                            "Column": col,
                            "Outliers": stats['count'],
                            "Percentage": f"{stats['percentage']:.1f}%",
                            "Bounds": f"[{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]"
                        })
                
                if outlier_data:
                    st.table(pd.DataFrame(outlier_data))
                    
                    # Boxplots for top 4 outlier columns
                    outlier_cols = [d['Column'] for d in outlier_data[:4]]
                    if outlier_cols:
                        fig = px.box(df, y=outlier_cols, title="Outlier Visualization")
                        st.plotly_chart(fig, use_container_width=True)
                        render_pin_button(f"Outliers: {name}", fig, f"eda_outliers_{name}")
                else:
                    st.success("✨ No significant outliers detected!")
            
        with tabs[4]:
            st.subheader("Categorical Data Analysis")
            cat_stats = eda.get_categorical_stats()
            if cat_stats:
                for col, stats in cat_stats.items():
                    with st.expander(f"Variable: {col}"):
                        st.write(f"**Unique Values:** {stats['unique_count']}")
                        st.write(f"**Top Value:** {stats['top_value']} ({stats['top_value_count']} times)")
                        
                        # Show top 10 distribution
                        counts = df[col].value_counts().head(10)
                        fig = px.pie(values=counts.values, names=counts.index, title=f"Top 10 categories for {col}")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns detected")

def render_advanced_stats_page():
    """Advanced Statistical Analysis page"""
    st.title("📈 Advanced Statistical Insights")
    st.markdown("### Deep statistical properties and patterns")
    
    render_dataset_selector("stats", allow_multi=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first")
        return

    datasets = st.session_state.selected_datasets
    
    for name, df in datasets.items():
        st.markdown(f"## 📁 {name}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 1:
            st.warning(f"No numeric data available for analysis in {name}")
            continue
            
        col_to_analyze = st.selectbox(f"Select column for deep analysis ({name})", numeric_cols, key=f"sel_stats_{name}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🕒 Distribution & Skewness")
            fig = px.histogram(df, x=col_to_analyze, marginal="box", title=f"Distribution of {col_to_analyze}")
            st.plotly_chart(fig, use_container_width=True)
            render_pin_button(f"Distribution: {col_to_analyze} ({name})", fig, f"stats_dist_{name}_{col_to_analyze}")
            
        with col2:
            st.subheader("📉 Quantile Analysis")
            quantiles = df[col_to_analyze].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            st.table(quantiles)
            
            # Cumulative Distribution
            fig = px.ecdf(df, x=col_to_analyze, title=f"Cumulative Distribution: {col_to_analyze}")
            st.plotly_chart(fig, use_container_width=True)

    st.info("💡 Pro Tip: Use the 'Complete Analysis' page for AI-driven interpretation of these stats.")

def is_id_column(column_name: str, df: pd.DataFrame = None) -> bool:
    """
    Identify if a column is an ID/Primary Key/Foreign Key column.
    Should NOT be used as axis in visualizations.
    
    Returns True if column appears to be an ID column.
    """
    col_lower = column_name.lower()
    
    # Check column name patterns for ID indicators
    id_patterns = [
        'id',  # Ends with ID or contains ID
        'key',  # Primary/Foreign keys
        'code',  # Product codes, postal codes that are identifiers
        'ref',  # References
        'pk',  # Primary Key prefix
        'fk',  # Foreign Key prefix
    ]
    
    # Check if column name contains ID patterns
    if any(pattern in col_lower for pattern in id_patterns):
        return True
    
    # Additional check: if column is numeric and has high cardinality relative to dataset size
    if df is not None and column_name in df.columns:
        try:
            if df[column_name].dtype.kind in 'iuf':  # numeric
                cardinality = df[column_name].nunique()
                total_rows = len(df)
                # If >80% unique values and numeric, likely an ID
                if cardinality / total_rows > 0.8:
                    return True
        except:
            pass
    
    return False

def get_valid_visualization_columns(df: pd.DataFrame, exclude_ids: bool = True) -> dict:
    """
    Get columns that are safe to use in visualizations.
    Filters out ID columns and validates against Power BI best practices.
    
    Returns dict with 'categorical', 'numeric', 'datetime' columns.
    """
    categorical_cols = []
    numeric_cols = []
    datetime_cols = []
    
    for col in df.columns:
        # Skip ID columns if requested
        if exclude_ids and is_id_column(col, df):
            continue
        
        dtype = df[col].dtype
        
        if dtype.kind == 'M':  # datetime
            datetime_cols.append(col)
        elif dtype.kind in 'iuf':  # numeric
            numeric_cols.append(col)
        else:  # categorical/object
            categorical_cols.append(col)
    
    return {
        'categorical': categorical_cols,
        'numeric': numeric_cols,
        'datetime': datetime_cols,
        'all': categorical_cols + numeric_cols + datetime_cols
    }

def validate_chart_specification(chart_type: str, x_col: str, y_col: str, 
                                 values_col: str, names_col: str, 
                                 df: pd.DataFrame, dataset_name: str) -> tuple:
    """
    Validate chart specification against Power BI best practices.
    Returns (is_valid: bool, error_message: str, corrected_spec: dict)
    
    RULES:
    - NO ID columns as axes
    - Line charts ONLY for time series (date/time X-axis)
    - Pie charts need numeric values and categorical names
    - Scatter needs numeric axes (no IDs)
    - Bar/Column can use categorical x-axis (but not IDs)
    """
    error_messages = []
    corrected_spec = {'chart_type': chart_type, 'x': x_col, 'y': y_col, 'values': values_col, 'names': names_col}
    
    valid_cols = get_valid_visualization_columns(df, exclude_ids=True)
    
    # Check chart type specific rules
    if chart_type == 'line':
        # Line charts ONLY for time series data
        if x_col and is_id_column(x_col, df):
            error_messages.append(f"❌ Line chart: X-axis cannot be ID column '{x_col}'")
            return False, error_messages, corrected_spec
        
        # X should be datetime for line charts
        if x_col and x_col in df.columns:
            if df[x_col].dtype.kind != 'M':  # Not datetime
                error_messages.append(f"⚠️ Line chart '{x_col}' is not time-based. Use Bar chart instead.")
                corrected_spec['chart_type'] = 'bar'
    
    elif chart_type in ['bar', 'column']:
        # Bar/column X-axis shouldn't be ID
        if x_col and is_id_column(x_col, df):
            error_messages.append(f"❌ Bar chart: Cannot use ID column '{x_col}' as category axis")
            # Try to find valid categorical column
            if valid_cols['categorical']:
                corrected_spec['x'] = valid_cols['categorical'][0]
            else:
                return False, error_messages, corrected_spec
    
    elif chart_type == 'scatter':
        # Both axes must be numeric (not IDs)
        if x_col and is_id_column(x_col, df):
            error_messages.append(f"❌ Scatter chart: X-axis cannot be ID '{x_col}'")
            if valid_cols['numeric']:
                corrected_spec['x'] = valid_cols['numeric'][0]
        
        if y_col and is_id_column(y_col, df):
            error_messages.append(f"❌ Scatter chart: Y-axis cannot be ID '{y_col}'")
            if len(valid_cols['numeric']) > 1:
                corrected_spec['y'] = valid_cols['numeric'][1]
            elif valid_cols['numeric']:
                corrected_spec['y'] = valid_cols['numeric'][0]
    
    elif chart_type == 'pie':
        # Values must be numeric, names must be categorical (not IDs)
        if values_col and is_id_column(values_col, df):
            error_messages.append(f"⚠️ Pie chart: Using ID column '{values_col}' as values (unusual)")
        
        if names_col and is_id_column(names_col, df):
            error_messages.append(f"❌ Pie chart: Cannot use ID column '{names_col}' for slice names")
            if valid_cols['categorical']:
                corrected_spec['names'] = valid_cols['categorical'][0]
    
    elif chart_type == 'histogram':
        # X should be numeric (not ID)
        if x_col and is_id_column(x_col, df):
            error_messages.append(f"⚠️ Histogram: Using ID column '{x_col}' for distribution (unusual)")
            # Try to find numeric non-ID column
            if valid_cols['numeric']:
                corrected_spec['x'] = valid_cols['numeric'][0]
    
    is_valid = len(error_messages) == 0
    return is_valid, error_messages, corrected_spec

def parse_and_generate_charts(llm_response: str, datasets: dict) -> list:
    """Parse LLM response and generate charts from CHART specifications
    
    CRITICAL: Validates ALL chart specs against Power BI best practices.
    Rejects any chart using ID columns or violating visualization rules.
    Generates smart defaults if LLM fails validation.
    """
    import re
    charts = []
    
    # Find all CHART specifications - handle both quoted and unquoted titles
    # Pattern matches: CHART_N: type=X, dataset=Y, x=A, y=B, title="Title" or title=Title
    chart_patterns = re.findall(
        r'CHART_\d+:\s*type=(\w+),\s*dataset=([^,]+),\s*'
        r'(?:x=([^,]+),\s*)?'
        r'(?:y=([^,]+),\s*)?'
        r'(?:values=([^,]+),\s*)?'
        r'(?:names=([^,]+),\s*)?'
        r'title=["\']?([^"\'\n]+)["\']?',
        llm_response, re.IGNORECASE
    )
    
    for match in chart_patterns:
        chart_type, dataset_name, x_col, y_col, values_col, names_col, title = match
        dataset_name = dataset_name.strip()
        title = title.strip()
        
        # Find the matching dataset
        df = None
        for name, data in datasets.items():
            if dataset_name.lower() in name.lower() or name.lower() in dataset_name.lower():
                df = data
                dataset_name = name
                break
        
        if df is None:
            # Try first dataset as fallback
            if len(datasets) == 0:
                continue
            dataset_name = list(datasets.keys())[0]
            df = datasets[dataset_name]
        
        try:
            fig = None
            chart_type = chart_type.lower().strip()
            # Clean column names - remove quotes if present
            x_col = x_col.strip().strip('"').strip("'") if x_col else None
            y_col = y_col.strip().strip('"').strip("'") if y_col else None
            values_col = values_col.strip().strip('"').strip("'") if values_col else None
            names_col = names_col.strip().strip('"').strip("'") if names_col else None
            
            # ============ CRITICAL: VALIDATE CHART SPEC BEFORE RENDERING ============
            is_valid, validation_errors, corrected = validate_chart_specification(
                chart_type, x_col, y_col, values_col, names_col, df, dataset_name
            )
            
            # REJECT if any column is an ID column (strict validation)
            if x_col and x_col in df.columns and is_id_column(x_col, df):
                continue  # Skip this chart
            if y_col and y_col in df.columns and is_id_column(y_col, df):
                continue  # Skip this chart
            if values_col and values_col in df.columns and is_id_column(values_col, df):
                continue  # Skip this chart
            if names_col and names_col in df.columns and is_id_column(names_col, df):
                continue  # Skip this chart
            
            # Skip if validation fails
            if not is_valid:
                continue
            
            # Use corrected spec if validation made changes
            chart_type = corrected['chart_type']
            x_col = corrected['x']
            y_col = corrected['y']
            values_col = corrected['values']
            names_col = corrected['names']
            
            # ============ GENERATE CHART WITH VALIDATED COLUMNS ============
            if chart_type == 'bar':
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    agg_df = df.groupby(x_col)[y_col].sum().nlargest(15).reset_index()
                    fig = px.bar(agg_df, x=x_col, y=y_col, title=title, color=y_col, color_continuous_scale=get_palette_color_scale())
                elif x_col and x_col in df.columns:
                    fig = px.bar(df[x_col].value_counts().head(15).reset_index(), x='index', y=x_col, title=title)
                    
            elif chart_type == 'line':
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    # Sort by x for better line chart
                    plot_df = df.sort_values(x_col).head(100)
                    fig = px.line(plot_df, x=x_col, y=y_col, title=title)
                    
            elif chart_type == 'pie':
                if values_col and names_col and values_col in df.columns and names_col in df.columns:
                    agg_df = df.groupby(names_col)[values_col].sum().nlargest(10).reset_index()
                    fig = px.pie(agg_df, values=values_col, names=names_col, title=title)
                elif names_col and names_col in df.columns:
                    fig = px.pie(df[names_col].value_counts().head(10).reset_index(), values=names_col, names='index', title=title)
                    
            elif chart_type == 'scatter':
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    fig = px.scatter(df.head(500), x=x_col, y=y_col, title=title)
                    
            elif chart_type == 'histogram':
                if x_col and x_col in df.columns:
                    fig = px.histogram(df, x=x_col, title=title, nbins=30, color_discrete_sequence=get_color_palette()[:1])
            
            if fig:
                charts.append({
                    'figure': fig,
                    'title': title,
                    'type': chart_type,
                    'dataset': dataset_name
                })
        except Exception as e:
            continue
    
    # If no charts were parsed or all rejected, generate smart defaults using ONLY valid columns
    if not charts:
        for name, df in list(datasets.items())[:2]:
            valid_cols = get_valid_visualization_columns(df, exclude_ids=True)
            
            cat_cols = valid_cols['categorical']
            numeric_cols = valid_cols['numeric']
            
            # Chart 1: Bar chart with categorical data
            if len(cat_cols) > 0 and len(numeric_cols) > 0:
                agg_df = df.groupby(cat_cols[0])[numeric_cols[0]].sum().nlargest(10).reset_index()
                fig = px.bar(agg_df, x=cat_cols[0], y=numeric_cols[0], title=f"Top {cat_cols[0]} by {numeric_cols[0]}", 
                            color=numeric_cols[0], color_continuous_scale=get_palette_color_scale())
                charts.append({'figure': fig, 'title': f"Top {cat_cols[0]} by {numeric_cols[0]}", 'type': 'bar', 'dataset': name})
            
            # Chart 2: Scatter chart with numeric data
            if len(numeric_cols) >= 2:
                fig = px.scatter(df.head(300), x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                charts.append({'figure': fig, 'title': f"{numeric_cols[0]} vs {numeric_cols[1]}", 'type': 'scatter', 'dataset': name})
    
    return charts


def render_strategic_analysis_page():
    """Senior Level AI Strategic Analysis page - Full Business Analysis on All Data"""
    st.title("🤖 Strategic AI Analyst")
    st.markdown("### Comprehensive Business Intelligence & Strategic Analysis")
    
    # Check if data is loaded
    if not st.session_state.multi_file_loader or len(st.session_state.multi_file_loader.get_loaded_files()) == 0:
        st.warning("⚠️ Please load data first from 'Multi-File Loading' page")
        return
    
    # Get all datasets info
    all_datasets = get_all_datasets()
    total_files = len(st.session_state.multi_file_loader.get_loaded_files())
    cleaned_datasets = st.session_state.get('cleaned_datasets', {})
    cleaned_count = len([k for k in all_datasets.keys() if k in cleaned_datasets])
    not_cleaned_count = len(all_datasets) - cleaned_count
    
    # Display file statistics
    st.markdown("---")
    st.subheader("📁 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📂 Total Files", total_files)
    with col2:
        st.metric("📊 Total Datasets/Sheets", len(all_datasets))
    with col3:
        st.metric("✨ Cleaned", cleaned_count)
    with col4:
        st.metric("📄 Original (Not Cleaned)", not_cleaned_count)
    
    # Show datasets list
    with st.expander("📋 View All Loaded Datasets"):
        for name, df in all_datasets.items():
            status = "✨ Cleaned" if name in cleaned_datasets else "📄 Original"
            st.write(f"- **{name}**: {df.shape[0]:,} rows × {df.shape[1]} columns | {status}")
    
    # Calculate total statistics
    total_rows = sum(df.shape[0] for df in all_datasets.values())
    total_cols = sum(df.shape[1] for df in all_datasets.values())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Total Rows (All Datasets)", f"{total_rows:,}")
    with col2:
        st.metric("📋 Total Columns (All Datasets)", total_cols)
    
    st.markdown("---")
    
    # Run Full Analysis Button
    st.subheader("🚀 Run Comprehensive Business Analysis")
    st.info("💡 This analysis will combine ALL loaded datasets and provide business insights, problems, recommendations, and visualizations.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_analysis = st.button("🚀 Run Full Strategic Analysis on ALL Data", key="strategic_full_analysis", use_container_width=True)
    
    if run_analysis or st.session_state.get('strategic_full_analysis_result'):
        if run_analysis:
            with st.spinner("🔄 AI Senior Analyst is performing comprehensive business analysis on all data... This may take a few minutes."):
                # Prepare comprehensive data summary
                data_summary = ""
                combined_numeric_stats = {}
                all_columns_info = []
                
                for dataset_name, df in all_datasets.items():
                    data_summary += f"\n\n=== Dataset: {dataset_name} ===\n"
                    data_summary += f"Status: {'CLEANED' if dataset_name in cleaned_datasets else 'ORIGINAL'}\n"
                    data_summary += f"Rows: {len(df):,}, Columns: {len(df.columns)}\n"
                    data_summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                    
                    # Column details
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        unique = df[col].nunique()
                        nulls = df[col].isna().sum()
                        null_pct = (nulls / len(df)) * 100 if len(df) > 0 else 0
                        sample = df[col].dropna().head(3).tolist()
                        data_summary += f"  - {col} ({dtype}): {unique} unique, {null_pct:.1f}% missing, samples: {sample}\n"
                        all_columns_info.append({'dataset': dataset_name, 'column': col, 'dtype': dtype})
                    
                    # Numeric statistics
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        data_summary += f"\nKey Numeric Statistics for {dataset_name}:\n"
                        for col in numeric_cols[:5]:  # Top 5 numeric columns
                            data_summary += f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}, sum={df[col].sum():.2f}\n"
                
                # Get schema relationships if available
                relationship_summary = ""
                if st.session_state.get('schema_analyzer') and st.session_state.schema_analyzer.relationships:
                    relationship_summary = "\n\n=== Detected Table Relationships ===\n"
                    for rel in st.session_state.schema_analyzer.relationships:
                        relationship_summary += f"- {rel['file1']}.{rel['column1']} → {rel['file2']}.{rel['column2']} ({rel['relationship_type']})\n"
                
                # Business Goals context
                goals_context = ""
                if st.session_state.business_goals.get('completed'):
                    goals_context = f"""
=== BUSINESS CONTEXT (User Defined) ===
Business Problem: {st.session_state.business_goals.get('problem', 'Not specified')}
Primary Objective: {st.session_state.business_goals.get('objective', 'Not specified')}
Target Audience: {st.session_state.business_goals.get('target', 'Not specified')}
"""
                
                # Comprehensive analysis prompt with visualization specifications
                analysis_prompt = f"""You are a Senior Business Intelligence Analyst and Strategic Advisor.

{goals_context}

=== ALL LOADED DATA ({len(all_datasets)} datasets, {total_rows:,} total rows) ===
{data_summary}
{relationship_summary}

=== YOUR MISSION ===
Perform a COMPREHENSIVE STRATEGIC BUSINESS ANALYSIS across ALL the data provided. 
Focus on BUSINESS INSIGHTS, not data quality issues.

---
## ANALYSIS FRAMEWORK
---

### 1. EXECUTIVE SUMMARY
- High-level overview of what the data reveals about the business
- Key metrics and their significance
- Overall business health assessment

### 2. KEY BUSINESS METRICS & KPIs
- Identify the most important KPIs from the data
- Calculate and present key metrics (totals, averages, trends)
- Benchmark analysis (internal comparisons)

### 3. BUSINESS INSIGHTS & DISCOVERIES
- What are the top 5-10 most important insights?
- What patterns or trends are visible?
- What correlations exist between different metrics?

### 4. BUSINESS PROBLEMS & CHALLENGES IDENTIFIED
- What problems does the data reveal?
- What areas need immediate attention?
- What risks are visible in the data?

### 5. OPPORTUNITIES & GROWTH AREAS
- What opportunities does the data suggest?
- Which segments show growth potential?

### 6. STRATEGIC RECOMMENDATIONS
For each recommendation:
- What: Clear action to take
- Why: Data-backed reasoning
- Impact: Expected business impact (High/Medium/Low)
- Priority: Urgency level (1-5)

### 7. VISUALIZATION SPECIFICATIONS
CRITICAL: Write EXACTLY 5 chart specifications in PLAIN TEXT (NOT in a code block).
Each on its own line, using this EXACT format:

CHART_1: type=bar, dataset=<dataset_name>, x=<column_name>, y=<column_name>, title=<chart_title>
CHART_2: type=line, dataset=<dataset_name>, x=<column_name>, y=<column_name>, title=<chart_title>
CHART_3: type=pie, dataset=<dataset_name>, values=<column_name>, names=<column_name>, title=<chart_title>
CHART_4: type=scatter, dataset=<dataset_name>, x=<column_name>, y=<column_name>, title=<chart_title>
CHART_5: type=histogram, dataset=<dataset_name>, x=<column_name>, title=<chart_title>

⚠️ ABSOLUTE RULES - MUST FOLLOW OR CHARTS WILL BE REJECTED:

❌ **DO NOT USE** ANY of these patterns in charts:
- Columns ending in: ID, KEY, CODE, REF, PK, FK
- Examples: CustomerID, ProductID, OrderID, TerritoryID, SalesTaxRateID, BusinessEntityID
- System identifiers of ANY kind
- Columns where 80%+ of values are unique (these are IDs)

✅ **MUST USE** descriptive business columns:
- Product Names (not ProductID)
- Category Names (not CategoryID)
- Region Names (not RegionID/TerritoryID)  
- Sales Amounts, Quantities, Revenues (actual numbers)
- Dates (for time-based charts)
- Percentages, Ratios, Metrics

CHART TYPE RULES (MANDATORY):
- **Bar Chart**: Use Category names with SUM/COUNT of measures (e.g., "Sales by Product Category")
- **Line Chart**: ONLY if X-axis is Date/Month/Year (temporal data). NEVER use IDs or categories
- **Pie Chart**: Use meaningful category names (NOT IDs) with percentage/amount values
- **Scatter**: Use actual business metrics on both axes, NOT IDs (e.g., "Price vs Quantity")
- **Histogram**: Distribution of actual measures (Revenue, Quantity), NOT ID counts

⚠️ VALIDATION EXAMPLES:

WRONG (❌ WILL BE REJECTED):
"CHART_1: type=bar, dataset=Sales, x=CustomerID, y=SalesAmount, title=Sales by CustomerID"
"CHART_2: type=line, dataset=Territory, x=TerritoryID, y=Sales, title=Territory Sales Over Time"
"CHART_4: type=scatter, dataset=Tax, x=SalesTaxRateID, y=TaxRate, title=Tax Rate Distribution"

CORRECT (✅ WILL BE ACCEPTED):
"CHART_1: type=bar, dataset=Sales, x=ProductName, y=SalesAmount, title=Top Products by Sales"
"CHART_2: type=line, dataset=Sales, x=MonthYear, y=TotalSales, title=Monthly Sales Trend"
"CHART_4: type=scatter, dataset=Sales, x=UnitPrice, y=OrderQuantity, title=Price vs Quantity Relationship"

REMEMBER: If a column name contains ID, KEY, CODE, or is mostly unique values, DO NOT USE IT IN CHARTS!

### 8. NEXT STEPS & FURTHER ANALYSIS
- What additional data would enhance the analysis?
- What deeper dives are recommended?

---
OUTPUT FORMAT: Use clear markdown with headers, bullet points, and structured sections.
Be specific with numbers and percentages from the actual data.
"""

                try:
                    import ollama
                    response = ollama.chat(
                        model="qwen2.5:7b",
                        messages=[{"role": "user", "content": analysis_prompt}]
                    )
                    
                    analysis_result = response['message']['content']
                    st.session_state.strategic_full_analysis_result = analysis_result
                    st.session_state.strategic_analysis_datasets = all_datasets
                    st.session_state.strategic_analysis_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Parse and generate LLM-suggested charts
                    st.session_state.strategic_llm_charts = parse_and_generate_charts(analysis_result, all_datasets)
                    
                except Exception as e:
                    st.error(f"❌ Error running analysis: {str(e)}")
                    return
        
        # Display Results
        if st.session_state.get('strategic_full_analysis_result'):
            st.markdown("---")
            st.markdown("## 📊 Strategic Analysis Report")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📅 Generated: {st.session_state.get('strategic_analysis_timestamp', 'N/A')}")
            with col2:
                st.caption(f"📊 Datasets Analyzed: {len(all_datasets)}")
            with col3:
                st.caption(f"📈 Total Rows: {total_rows:,}")
            
            # Analysis content
            st.markdown(st.session_state.strategic_full_analysis_result)
            
            # LLM-Generated Charts Section (AI Recommended)
            st.markdown("---")
            st.markdown("## 🤖 AI-Recommended Visualizations")
            st.caption("Charts generated based on AI analysis of your data")
            
            llm_charts = st.session_state.get('strategic_llm_charts', [])
            if llm_charts:
                chart_cols = st.columns(2)
                for idx, chart_info in enumerate(llm_charts):
                    with chart_cols[idx % 2]:
                        st.plotly_chart(chart_info['figure'], use_container_width=True)
                        render_pin_button(f"AI: {chart_info['title']}", chart_info['figure'], f"llm_chart_{idx}")
            else:
                st.info("💡 No specific charts were generated. Using auto-generated visualizations below.")
            
            # Auto-Generated Business Visualizations
            st.markdown("---")
            st.markdown("## 📈 Auto-Generated Business Visualizations")
            
            analysis_datasets = st.session_state.get('strategic_analysis_datasets', all_datasets)
            
            # Store auto-generated charts for export
            auto_charts = []
            
            viz_tabs = st.tabs(["📊 Key Metrics", "📈 Trends", "🔗 Comparisons", "📉 Distributions"])
            
            with viz_tabs[0]:
                st.subheader("Key Business Metrics")
                for dataset_name, df in list(analysis_datasets.items())[:3]:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
                    # Filter out ID columns
                    valid_numeric = [col for col in numeric_cols if not is_id_column(col, df)]
                    if len(valid_numeric) > 0:
                        st.markdown(f"**{dataset_name}**")
                        metric_cols = st.columns(min(4, len(valid_numeric)))
                        for idx, col in enumerate(valid_numeric):
                            with metric_cols[idx]:
                                total = df[col].sum()
                                avg = df[col].mean()
                                st.metric(col, f"{total:,.0f}" if total > 100 else f"{avg:.2f}")
            
            with viz_tabs[1]:
                st.subheader("Trend Analysis")
                # Find date columns and create trend charts
                for dataset_name, df in list(analysis_datasets.items())[:2]:
                    valid_cols = get_valid_visualization_columns(df, exclude_ids=True)
                    date_cols = valid_cols['datetime']
                    numeric_cols = valid_cols['numeric']
                    
                    if len(date_cols) > 0 and len(numeric_cols) > 0:
                        fig = px.line(df.head(100), x=date_cols[0], y=numeric_cols[0], 
                                     title=f"{numeric_cols[0]} Trend - {dataset_name}")
                        st.plotly_chart(fig, use_container_width=True)
                        render_pin_button(f"Trend {dataset_name}", fig, f"strategic_trend_{dataset_name}")
                        auto_charts.append({'figure': fig, 'title': f"{numeric_cols[0]} Trend - {dataset_name}", 'type': 'trend'})
            
            with viz_tabs[2]:
                st.subheader("Comparative Analysis")
                for dataset_name, df in list(analysis_datasets.items())[:2]:
                    valid_cols = get_valid_visualization_columns(df, exclude_ids=True)
                    cat_cols = valid_cols['categorical']
                    numeric_cols = valid_cols['numeric']
                    
                    if len(cat_cols) > 0 and len(numeric_cols) > 0:
                        cat_col = cat_cols[0]
                        num_col = numeric_cols[0]
                        
                        agg_df = df.groupby(cat_col)[num_col].sum().nlargest(10).reset_index()
                        fig = px.bar(agg_df, x=cat_col, y=num_col, 
                                    title=f"Top {cat_col} by {num_col} - {dataset_name}",
                                    color=num_col, color_continuous_scale=get_palette_color_scale())
                        st.plotly_chart(fig, use_container_width=True)
                        render_pin_button(f"Comparison {dataset_name}", fig, f"strategic_comp_{dataset_name}")
                        auto_charts.append({'figure': fig, 'title': f"Top {cat_col} by {num_col} - {dataset_name}", 'type': 'comparison'})
            
            with viz_tabs[3]:
                st.subheader("Distribution Analysis")
                for dataset_name, df in list(analysis_datasets.items())[:2]:
                    valid_cols = get_valid_visualization_columns(df, exclude_ids=True)
                    numeric_cols = valid_cols['numeric']
                    
                    if len(numeric_cols) > 0:
                        col = numeric_cols[0]
                        fig = px.histogram(df, x=col, title=f"Distribution of {col} - {dataset_name}",
                                          nbins=30, color_discrete_sequence=['#4a90d9'])
                        st.plotly_chart(fig, use_container_width=True)
                        render_pin_button(f"Distribution {dataset_name}", fig, f"strategic_dist_{dataset_name}")
                        auto_charts.append({'figure': fig, 'title': f"Distribution of {col} - {dataset_name}", 'type': 'distribution'})
            
            # Combine all charts for export
            all_export_charts = llm_charts + auto_charts
            
            # Download Section
            st.markdown("---")
            st.markdown("## 📥 Download Complete Report")
            
            # Prepare report content
            report_md = f"""# Strategic Business Analysis Report

## Report Metadata
- **Generated:** {st.session_state.get('strategic_analysis_timestamp', 'N/A')}
- **Total Datasets Analyzed:** {len(all_datasets)}
- **Total Rows Analyzed:** {total_rows:,}
- **Cleaned Datasets:** {cleaned_count}
- **Original Datasets:** {not_cleaned_count}

## Datasets Included
"""
            for name in all_datasets.keys():
                status = "✨ Cleaned" if name in cleaned_datasets else "📄 Original"
                report_md += f"- {name} ({status})\n"
            
            if st.session_state.business_goals.get('completed'):
                report_md += f"""
## Business Context
- **Problem:** {st.session_state.business_goals.get('problem', 'N/A')}
- **Objective:** {st.session_state.business_goals.get('objective', 'N/A')}
- **Target Audience:** {st.session_state.business_goals.get('target', 'N/A')}
"""
            
            report_md += f"""
---

{st.session_state.strategic_full_analysis_result}

---
*Report generated by Strategic AI Analyst*
"""
            
            # HTML Report - Clean and Compact
            html_report = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Strategic Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; font-size: 12px; line-height: 1.5; color: #333; }}
        h1 {{ font-size: 18px; color: #1a1a2e; border-bottom: 2px solid #4a90d9; padding-bottom: 8px; margin-bottom: 15px; }}
        h2 {{ font-size: 14px; color: #16213e; margin: 15px 0 8px 0; border-left: 3px solid #4a90d9; padding-left: 8px; }}
        h3 {{ font-size: 12px; color: #0f3460; margin: 10px 0 5px 0; }}
        .header {{ background: #1a1a2e; color: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
        .header h1 {{ color: white; border: none; margin: 0; font-size: 16px; }}
        .header p {{ margin: 5px 0 0 0; font-size: 11px; opacity: 0.9; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 10px 0; }}
        .info-box {{ background: #f5f5f5; padding: 10px; border-radius: 4px; text-align: center; }}
        .info-box .value {{ font-size: 18px; font-weight: bold; color: #4a90d9; }}
        .info-box .label {{ font-size: 10px; color: #666; }}
        .section {{ margin: 15px 0; padding: 12px; background: #fafafa; border-radius: 4px; }}
        ul, ol {{ margin-left: 18px; }}
        li {{ margin-bottom: 4px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
        th {{ background: #4a90d9; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .footer {{ margin-top: 20px; padding-top: 10px; border-top: 1px solid #eee; color: #999; font-size: 10px; text-align: center; }}
        @media print {{ body {{ padding: 10px; }} .header {{ background: #1a1a2e !important; -webkit-print-color-adjust: exact; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Strategic Business Analysis</h1>
        <p>Generated: {st.session_state.get('strategic_analysis_timestamp', 'N/A')}</p>
    </div>
    
    <div class="info-grid">
        <div class="info-box"><div class="value">{len(all_datasets)}</div><div class="label">Datasets</div></div>
        <div class="info-box"><div class="value">{total_rows:,}</div><div class="label">Total Rows</div></div>
        <div class="info-box"><div class="value">{total_cols}</div><div class="label">Columns</div></div>
        <div class="info-box"><div class="value">{cleaned_count}</div><div class="label">Cleaned</div></div>
    </div>"""
            
            if st.session_state.business_goals.get('completed'):
                html_report += f"""
    <div class="section">
        <h2>🎯 Business Context</h2>
        <p><strong>Problem:</strong> {st.session_state.business_goals.get('problem', 'N/A')}</p>
        <p><strong>Objective:</strong> {st.session_state.business_goals.get('objective', 'N/A')}</p>
    </div>"""
            
            # Convert markdown to HTML properly
            import re
            analysis_text = st.session_state.strategic_full_analysis_result
            
            # Convert headers
            analysis_text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', analysis_text, flags=re.MULTILINE)
            analysis_text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', analysis_text, flags=re.MULTILINE)
            analysis_text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', analysis_text, flags=re.MULTILINE)
            
            # Convert bold
            analysis_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', analysis_text)
            
            # Convert bullet points
            analysis_text = re.sub(r'^- (.+)$', r'<li>\1</li>', analysis_text, flags=re.MULTILINE)
            analysis_text = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', analysis_text, flags=re.MULTILINE)
            
            # Convert tables (markdown to HTML)
            lines = analysis_text.split('\n')
            new_lines = []
            in_table = False
            for line in lines:
                if '|' in line and line.strip().startswith('|'):
                    if not in_table:
                        in_table = True
                        new_lines.append('<table>')
                    cells = [c.strip() for c in line.split('|')[1:-1]]
                    if all(c.replace('-', '') == '' for c in cells):
                        continue  # Skip separator row
                    row_tag = 'th' if not any('<tr>' in l for l in new_lines[-5:] if '<table>' not in l) else 'td'
                    if row_tag == 'th':
                        row_tag = 'th'
                    else:
                        row_tag = 'td'
                    new_lines.append('<tr>' + ''.join(f'<{row_tag}>{c}</{row_tag}>' for c in cells) + '</tr>')
                else:
                    if in_table:
                        in_table = False
                        new_lines.append('</table>')
                    new_lines.append(line)
            if in_table:
                new_lines.append('</table>')
            analysis_text = '\n'.join(new_lines)
            
            # Wrap paragraphs
            analysis_text = analysis_text.replace('\n\n', '</p><p>')
            
            html_report += f"""
    <div class="section">
        <h2>📈 Analysis Results</h2>
        <div>{analysis_text}</div>
    </div>"""
            
            # Add Key Business Metrics to HTML report
            html_report += """
    <div class="section">
        <h2>📊 Key Business Metrics</h2>
        <div class="info-grid">
"""
            for dataset_name, df in list(analysis_datasets.items())[:3]:
                numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
                if len(numeric_cols) > 0:
                    for col in numeric_cols:
                        total = df[col].sum()
                        avg = df[col].mean()
                        display_val = f"{total:,.0f}" if total > 100 else f"{avg:.2f}"
                        html_report += f"""
            <div class="info-box">
                <div class="value">{display_val}</div>
                <div class="label">{col}</div>
            </div>
"""
            html_report += """
        </div>
    </div>
"""
            
            # Add charts to HTML report
            html_report += """
    <div class="section">
        <h2>📈 Visualizations</h2>
"""
            for idx, chart_info in enumerate(all_export_charts):
                try:
                    # Convert plotly figure to HTML div
                    chart_html = chart_info['figure'].to_html(include_plotlyjs='cdn' if idx == 0 else False, 
                                                               full_html=False, 
                                                               config={'displayModeBar': False})
                    html_report += f"""
        <div style="margin: 15px 0; page-break-inside: avoid;">
            <h3>{chart_info['title']}</h3>
            {chart_html}
        </div>
"""
                except:
                    pass
            
            html_report += """
    </div>
    
    <div class="footer">
        <p>Report generated by Strategic AI Analyst | Print: Ctrl+P → Save as PDF</p>
    </div>
</body>
</html>"""
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📄 Download Markdown",
                    data=report_md,
                    file_name=f"strategic_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    label="🌐 Download HTML (with Charts)",
                    data=html_report,
                    file_name=f"strategic_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col3:
                st.download_button(
                    label="📝 Download Text",
                    data=report_md,
                    file_name=f"strategic_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            st.success("💡 **To save as PDF:** Download the HTML file, open in browser, and use **Print → Save as PDF**")
            
            # Re-run button
            if st.button("🔄 Re-run Analysis", key="rerun_strategic"):
                st.session_state.strategic_full_analysis_result = None
                st.rerun()

def render_kpi_dashboard():
    """Advanced KPI & Business Intelligence Dashboard - All Data Analysis"""
    st.title("📊 KPIs & Metrics Dashboard")
    st.markdown("### Comprehensive Key Performance Indicators Across All Data")
    
    # Check if data is loaded
    if not st.session_state.multi_file_loader or len(st.session_state.multi_file_loader.get_loaded_files()) == 0:
        st.warning("⚠️ Please load data first from 'Multi-File Loading' page")
        return
    
    # Get all datasets
    all_datasets = get_all_datasets()
    total_files = len(st.session_state.multi_file_loader.get_loaded_files())
    cleaned_datasets = st.session_state.get('cleaned_datasets', {})
    cleaned_count = len([k for k in all_datasets.keys() if k in cleaned_datasets])
    not_cleaned_count = len(all_datasets) - cleaned_count
    
    # Data Overview Section
    st.markdown("---")
    st.subheader("📁 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📂 Total Files", total_files)
    with col2:
        st.metric("📊 Total Datasets", len(all_datasets))
    with col3:
        st.metric("✨ Cleaned", cleaned_count, delta="Ready" if cleaned_count > 0 else None)
    with col4:
        st.metric("📄 Not Cleaned", not_cleaned_count, delta="-" if not_cleaned_count > 0 else None, delta_color="inverse")
    
    # Calculate totals
    total_rows = sum(df.shape[0] for df in all_datasets.values())
    total_cols = sum(df.shape[1] for df in all_datasets.values())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Total Rows", f"{total_rows:,}")
    with col2:
        st.metric("📋 Total Columns", total_cols)
    
    # Load All Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        load_all = st.button("🚀 Analyze All Data - Generate Master KPIs", key="kpi_load_all", use_container_width=True)
    
    if load_all or st.session_state.get('kpi_master_analysis'):
        if load_all:
            st.session_state.kpi_master_analysis = True
        
        with st.spinner("🔄 Analyzing all datasets and generating master KPIs..."):
            
            # ============= MASTER KPIs SECTION =============
            st.markdown("---")
            st.markdown("## 🎯 Master KPIs - All Data Combined")
            st.info("💡 **Intelligent KPI Generation:** Keys are counted (not summed), Measures are aggregated correctly. Each KPI shows its function.")
            
            # Use Intelligent KPI Generator for each dataset
            all_kpis = []
            column_role_summary = {'keys': set(), 'measures': set(), 'categories': set()}
            
            for dataset_name, df in all_datasets.items():
                kpi_gen = IntelligentKPIGenerator(df, dataset_name)
                
                # Get column summaries
                summary = kpi_gen.get_column_summary()
                column_role_summary['keys'].update(summary['keys'])
                column_role_summary['measures'].update(summary['measures'])
                column_role_summary['categories'].update(summary['categories'])
                
                # Generate KPIs with proper aggregation
                dataset_kpis = kpi_gen.generate_all_kpis(max_kpis=10)
                for kpi in dataset_kpis:
                    kpi_dict = kpi.to_dict()
                    kpi_dict['dataset'] = dataset_name
                    all_kpis.append(kpi_dict)
            
            # Show column role detection
            with st.expander("🔍 **Column Role Detection** (Click to expand)", expanded=False):
                st.markdown("The system automatically detects column roles to apply correct aggregation functions:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**🔑 Key/ID Columns** (COUNT only, never SUM):")
                    for col in list(column_role_summary['keys'])[:10]:
                        st.write(f"  • `{col}`")
                with col2:
                    st.markdown("**📊 Measure Columns** (SUM/AVG applies):")
                    for col in list(column_role_summary['measures'])[:10]:
                        st.write(f"  • `{col}`")
                with col3:
                    st.markdown("**📁 Category Columns** (DISTINCT COUNT):")
                    for col in list(column_role_summary['categories'])[:10]:
                        st.write(f"  • `{col}`")
            
            # Group KPIs by measure vs count
            measure_kpis = [k for k in all_kpis if k['function'] in ['SUM', 'AVG', 'MEDIAN']]
            count_kpis = [k for k in all_kpis if k['function'] in ['COUNT', 'DISTINCT COUNT']]
            
            # Display Measure KPIs (TOP Priority)
            st.subheader("📊 Business Metrics (Measures)")
            if measure_kpis:
                for row_start in range(0, min(12, len(measure_kpis)), 4):
                    kpi_cols = st.columns(4)
                    for idx, kpi in enumerate(measure_kpis[row_start:row_start+4]):
                        with kpi_cols[idx]:
                            # Show function in metric label
                            label = f"{kpi['name']}"
                            func_badge = f"({kpi['function']})"
                            
                            st.metric(
                                label,
                                kpi['formatted_value'],
                                delta=func_badge,
                                help=f"📋 Column: {kpi['column']}\n📐 Function: {kpi['function']}\n📖 {kpi['business_definition']}"
                            )
                            
                            # Show warning if any
                            if kpi.get('warning'):
                                st.warning(kpi['warning'])
            else:
                st.info("No measure columns detected in the data.")
            
            # Display Count KPIs
            st.subheader("📈 Entity Counts (Keys & Categories)")
            if count_kpis:
                for row_start in range(0, min(8, len(count_kpis)), 4):
                    kpi_cols = st.columns(4)
                    for idx, kpi in enumerate(count_kpis[row_start:row_start+4]):
                        with kpi_cols[idx]:
                            label = f"{kpi['name']}"
                            func_badge = f"({kpi['function']})"
                            
                            st.metric(
                                label,
                                kpi['formatted_value'],
                                delta=func_badge,
                                help=f"📋 Column: {kpi['column']}\n📐 Function: {kpi['function']}\n📖 {kpi['business_definition']}"
                            )
            else:
                st.info("No key/category columns detected.")
            
            # ============= PER-DATASET KPIs =============
            st.markdown("---")
            st.markdown("## 📁 KPIs by Dataset")
            
            dataset_tabs = st.tabs([f"{'✨' if name in cleaned_datasets else '📄'} {name[:30]}..." if len(name) > 30 else f"{'✨' if name in cleaned_datasets else '📄'} {name}" for name in all_datasets.keys()])
            
            for tab, (dataset_name, df) in zip(dataset_tabs, all_datasets.items()):
                with tab:
                    # Use intelligent KPI generator
                    kpi_gen = IntelligentKPIGenerator(df, dataset_name)
                    col_summary = kpi_gen.get_column_summary()
                    
                    # Quick metrics
                    st.write(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)} | **Keys:** {len(col_summary['keys'])} | **Measures:** {len(col_summary['measures'])}")
                    
                    # Generate proper KPIs
                    dataset_kpis = kpi_gen.generate_all_kpis(max_kpis=6)
                    
                    if dataset_kpis:
                        kpi_cols = st.columns(min(6, len(dataset_kpis)))
                        for i, kpi in enumerate(dataset_kpis):
                            with kpi_cols[i]:
                                st.metric(
                                    kpi.name,
                                    kpi.formatted_value,
                                    delta=f"({kpi.function.value})",
                                    help=kpi.business_definition
                                )
                    
                    # Top performers chart - using correct aggregation
                    cat_cols = kpi_gen.get_category_columns()
                    measure_cols = kpi_gen.get_measure_columns()
                    
                    if cat_cols and measure_cols:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Use SUM for measure columns
                            agg_df = df.groupby(cat_cols[0])[measure_cols[0]].sum().nlargest(10).reset_index()
                            fig = px.bar(agg_df, x=cat_cols[0], y=measure_cols[0],
                                        title=f"SUM of {measure_cols[0]} by {cat_cols[0]}",
                                        color=measure_cols[0], color_continuous_scale=get_palette_color_scale())
                            st.plotly_chart(fig, use_container_width=True)
                            render_pin_button(f"{dataset_name}: SUM {measure_cols[0]}", fig, f"kpi_bar_{dataset_name}")
                        
                        with col2:
                            if len(measure_cols) >= 2:
                                fig = px.scatter(df.head(500), x=measure_cols[0], y=measure_cols[1],
                                               title=f"{measure_cols[0]} vs {measure_cols[1]} (Measures)")
                                st.plotly_chart(fig, use_container_width=True)
                                render_pin_button(f"{dataset_name}: Scatter", fig, f"kpi_scatter_{dataset_name}")
            
            # ============= CUSTOM KPI BUILDER =============
            st.markdown("---")
            st.markdown("## 🛠️ Custom KPI Builder")
            st.info("💡 Create custom KPIs with intelligent aggregation validation. Keys are protected from invalid SUM operations.")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                selected_dataset = st.selectbox("Select Dataset", list(all_datasets.keys()), key="custom_kpi_dataset")
            
            if selected_dataset:
                df = all_datasets[selected_dataset]
                kpi_gen = IntelligentKPIGenerator(df, selected_dataset)
                
                # Show column roles
                col_summary = kpi_gen.get_column_summary()
                
                with st.expander("🔍 Column Roles in This Dataset"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**🔑 Keys (COUNT only):**")
                        for c in col_summary['keys']:
                            st.write(f"  • `{c}`")
                    with col_b:
                        st.markdown("**📊 Measures (SUM/AVG):**")
                        for c in col_summary['measures']:
                            st.write(f"  • `{c}`")
                
                all_cols = df.columns.tolist()
                
                with col2:
                    dim_col = st.selectbox("Dimension (Group By)", all_cols, key="custom_kpi_dim")
                with col3:
                    met_col = st.selectbox("Metric (Aggregate)", all_cols, key="custom_kpi_met")
                with col4:
                    agg_type = st.selectbox("Aggregation", ["Sum", "Mean", "Count", "Distinct Count", "Max", "Min"], key="custom_kpi_agg")
                
                # Validate the aggregation choice
                is_valid, corrected_agg, warning = validate_kpi_request(df, met_col, agg_type)
                
                if not is_valid and warning:
                    st.warning(warning)
                    st.info(f"💡 Recommended aggregation for '{met_col}': **{corrected_agg.upper()}**")
                
                if st.button("📊 Generate Custom KPI", key="gen_custom_kpi"):
                    try:
                        # Use the validated/corrected aggregation
                        final_agg = agg_type.lower() if is_valid else corrected_agg
                        
                        # Map aggregation names
                        agg_map = {
                            'sum': 'sum', 'mean': 'mean', 'count': 'count',
                            'distinct count': 'nunique', 'max': 'max', 'min': 'min',
                            'distinctcount': 'nunique'
                        }
                        agg_func = agg_map.get(final_agg.lower(), 'count')
                        
                        if df[met_col].dtype.kind in 'iuf' or agg_func in ['count', 'nunique']:
                            custom_agg = df.groupby(dim_col)[met_col].agg(agg_func).reset_index()
                            custom_agg = custom_agg.nlargest(15, met_col) if agg_func != 'nunique' else custom_agg.head(15)
                            
                            # Create title with function
                            chart_title = f"{final_agg.upper()} of {met_col} by {dim_col}"
                            
                            fig = px.bar(custom_agg, x=dim_col, y=met_col, color=met_col,
                                       title=chart_title,
                                       color_continuous_scale=get_palette_color_scale())
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show KPI documentation
                            st.success(f"""
                            **📋 KPI Documentation:**
                            - **Column:** {met_col}
                            - **Function:** {final_agg.upper()}
                            - **Grouped By:** {dim_col}
                            - **Column Role:** {kpi_gen.get_column_role(met_col).value}
                            """)
                            
                            render_pin_button(f"Custom: {final_agg.upper()} {met_col} by {dim_col}", fig, f"custom_kpi_{dim_col}_{met_col}")
                        else:
                            st.warning(f"Column '{met_col}' is not numeric. Use 'Count' or 'Distinct Count' aggregation.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

def render_custom_dashboard():
    """Power BI-like Dashboard Builder with true drag-drop, pages, and advanced layout"""
    st.title("📊 Custom Dashboard Builder")
    
    # Initialize dashboard state with enhanced layout system
    if 'dashboard_pages' not in st.session_state:
        st.session_state.dashboard_pages = [{
            "name": "Overview",
            "charts": [],
            "cards": [],
            "layout": "freeform"  # freeform, grid_2col, grid_3col, grid_1col
        }]
    if 'current_dashboard_page' not in st.session_state:
        st.session_state.current_dashboard_page = 0
    if 'chart_titles' not in st.session_state:
        st.session_state.chart_titles = {}
    if 'dashboard_edit_mode' not in st.session_state:
        st.session_state.dashboard_edit_mode = False
    if 'dashboard_item_positions' not in st.session_state:
        st.session_state.dashboard_item_positions = {}  # Track positions of items
    if 'dashboard_item_sizes' not in st.session_state:
        st.session_state.dashboard_item_sizes = {}  # Track sizes of items
    if 'dashboard_z_order' not in st.session_state:
        st.session_state.dashboard_z_order = {}  # Track layering (z-index)
    
    # Get all datasets
    all_datasets = get_all_datasets()
    
    # ========== TOP TOOLBAR ==========
    st.markdown("---")
    
    # Mode toggle (Edit/View)
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 2])
    
    with col1:
        edit_mode = st.checkbox(
            "🎛️ Edit Mode",
            value=st.session_state.dashboard_edit_mode,
            key="edit_mode_toggle"
        )
        st.session_state.dashboard_edit_mode = edit_mode
    
    if st.session_state.dashboard_edit_mode:
        with col2:
            if st.button("➕ Add Page", use_container_width=True):
                page_num = len(st.session_state.dashboard_pages) + 1
                st.session_state.dashboard_pages.append({
                    "name": f"Page {page_num}",
                    "charts": [],
                    "cards": [],
                    "layout": "freeform"
                })
                st.session_state.current_dashboard_page = len(st.session_state.dashboard_pages) - 1
                st.rerun()
        
        with col3:
            if st.button("🤖 AI Layout", use_container_width=True):
                if all_datasets:
                    with st.spinner("🔄 Creating AI layout..."):
                        ai_pages = generate_ai_dashboard_layout(all_datasets, st.session_state.get('business_goals', {}))
                        st.session_state.dashboard_pages = ai_pages
                        st.session_state.current_dashboard_page = 0
                        st.success("✅ Dashboard created!")
                        st.rerun()
                else:
                    st.warning("⚠️ Load data first!")
        
        with col4:
            if st.button("📥 Import Pinned", use_container_width=True):
                if st.session_state.pinned_charts:
                    current_page = st.session_state.current_dashboard_page
                    for chart in st.session_state.pinned_charts:
                        st.session_state.dashboard_pages[current_page]["charts"].append(chart)
                    st.success(f"✅ Added {len(st.session_state.pinned_charts)} charts!")
                    st.rerun()
        
        with col5:
            if st.button("🗑️ Clear", use_container_width=True):
                current_page = st.session_state.current_dashboard_page
                st.session_state.dashboard_pages[current_page]["charts"] = []
                st.session_state.dashboard_pages[current_page]["cards"] = []
                st.rerun()
    
    with col6:
        # Layout selector
        layout_options = ["Freeform (Drag)", "2 Columns", "3 Columns", "1 Column"]
        layout_type = st.selectbox(
            "📐 Layout",
            layout_options,
            index=0 if st.session_state.dashboard_pages[st.session_state.current_dashboard_page].get("layout", "freeform") == "freeform" else 1,
            key="dashboard_layout",
            label_visibility="collapsed"
        )
        
        # Update layout type
        layout_map = {
            "Freeform (Drag)": "freeform",
            "2 Columns": "grid_2col",
            "3 Columns": "grid_3col",
            "1 Column": "grid_1col"
        }
        st.session_state.dashboard_pages[st.session_state.current_dashboard_page]["layout"] = layout_map.get(layout_type, "freeform")
    
    # ========== PAGE TABS & MANAGEMENT ==========
    st.markdown("---")
    page_names = [p["name"] for p in st.session_state.dashboard_pages]
    
    if len(page_names) > 1:
        page_tabs = st.tabs(page_names)
        for idx, tab in enumerate(page_tabs):
            with tab:
                st.session_state.current_dashboard_page = idx
    
    current_page_idx = st.session_state.current_dashboard_page
    current_page = st.session_state.dashboard_pages[current_page_idx]
    
    # ========== PAGE CONTROLS (EDIT MODE) ==========
    if st.session_state.dashboard_edit_mode:
        st.markdown("---")
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        
        with ctrl_col1:
            new_title = st.text_input(
                "Page Title",
                value=current_page["name"],
                key=f"page_title_{current_page_idx}",
                placeholder="Enter page title..."
            )
            if new_title != current_page["name"]:
                st.session_state.dashboard_pages[current_page_idx]["name"] = new_title
        
        with ctrl_col2:
            if st.button("📋 Duplicate Page", use_container_width=True, key=f"dup_page_{current_page_idx}"):
                import copy
                new_page = copy.deepcopy(current_page)
                new_page["name"] = f"{current_page['name']} (Copy)"
                st.session_state.dashboard_pages.append(new_page)
                st.success("✅ Page duplicated!")
                st.rerun()
        
        with ctrl_col3:
            if len(st.session_state.dashboard_pages) > 1:
                if st.button("🗑️ Delete", use_container_width=True, key=f"delete_page_{current_page_idx}"):
                    st.session_state.dashboard_pages.pop(current_page_idx)
                    st.session_state.current_dashboard_page = max(0, current_page_idx - 1)
                    st.rerun()
    
    # ========== ADD ITEMS (EDIT MODE) ==========
    if st.session_state.dashboard_edit_mode:
        st.markdown("---")
        add_tabs = st.tabs(["📊 Add Chart", "📈 Add KPI Card"])
        
        with add_tabs[0]:
            if all_datasets:
                add_chart_cols = st.columns([1, 1, 1])
                
                with add_chart_cols[0]:
                    chart_dataset = st.selectbox("Dataset", list(all_datasets.keys()), key="new_chart_dataset")
                
                with add_chart_cols[1]:
                    chart_type = st.selectbox(
                        "Chart Type",
                        ["Bar", "Line", "Pie", "Scatter", "Histogram", "Area", "Box"],
                        key="new_chart_type"
                    )
                
                with add_chart_cols[2]:
                    if chart_dataset:
                        df = all_datasets[chart_dataset]
                        all_cols = df.columns.tolist()
                        x_col = st.selectbox("X Axis", all_cols, key="new_chart_x")
                
                chart_cols2 = st.columns([1, 1, 1])
                with chart_cols2[0]:
                    if chart_dataset:
                        numeric_cols = all_datasets[chart_dataset].select_dtypes(include=[np.number]).columns.tolist()
                        y_col = st.selectbox("Y Axis", numeric_cols, key="new_chart_y")
                
                with chart_cols2[1]:
                    chart_title = st.text_input("Chart Title", key="new_chart_title")
                
                with chart_cols2[2]:
                    chart_width = st.slider("Width (%)", 25, 100, 50, key="new_chart_width")
                
                if st.button("➕ Add Chart to Page", use_container_width=True, key="create_new_chart"):
                    if chart_dataset and x_col and y_col:
                        df = all_datasets[chart_dataset]
                        
                        try:
                            if chart_type == "Bar":
                                agg_df = df.groupby(x_col)[y_col].sum().nlargest(10).reset_index()
                                fig = px.bar(agg_df, x=x_col, y=y_col, title=chart_title or f"{y_col} by {x_col}")
                            elif chart_type == "Line":
                                fig = px.line(df.head(100), x=x_col, y=y_col, title=chart_title or f"{y_col} Trend")
                            elif chart_type == "Pie":
                                agg_df = df.groupby(x_col)[y_col].sum().nlargest(10)
                                fig = px.pie(values=agg_df.values, names=agg_df.index, title=chart_title or f"{y_col} by {x_col}")
                            elif chart_type == "Scatter":
                                fig = px.scatter(df.head(500), x=x_col, y=y_col, title=chart_title or f"{x_col} vs {y_col}")
                            elif chart_type == "Area":
                                fig = px.area(df.head(100), x=x_col, y=y_col, title=chart_title or f"{y_col} Area")
                            elif chart_type == "Box":
                                fig = px.box(df, x=x_col, y=y_col, title=chart_title or f"Distribution of {y_col}")
                            else:
                                fig = px.histogram(df, x=y_col, title=chart_title or f"Distribution of {y_col}")
                            
                            # Apply color palette
                            fig = apply_color_palette_to_figure(fig)
                            
                            chart_id = f"chart_{current_page_idx}_{len(current_page['charts'])}"
                            st.session_state.dashboard_pages[current_page_idx]["charts"].append({
                                "id": chart_id,
                                "title": chart_title or f"{y_col} by {x_col}",
                                "figure": fig,
                                "type": chart_type,
                                "dataset": chart_dataset,
                                "x": x_col,
                                "y": y_col,
                                "width": chart_width
                            })
                            
                            # Initialize position (default layout)
                            st.session_state.dashboard_item_positions[chart_id] = {
                                "x": 0,
                                "y": len(current_page["charts"]) * 300,
                                "width": chart_width,
                                "height": 400
                            }
                            
                            st.success("✅ Chart added!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with add_tabs[1]:
            if all_datasets:
                kpi_cols = st.columns([1, 1, 1, 1])
                
                with kpi_cols[0]:
                    card_dataset = st.selectbox("Dataset", list(all_datasets.keys()), key="card_dataset")
                
                with kpi_cols[1]:
                    if card_dataset:
                        numeric_cols = all_datasets[card_dataset].select_dtypes(include=[np.number]).columns.tolist()
                        card_column = st.selectbox("Column", numeric_cols, key="card_column")
                
                with kpi_cols[2]:
                    card_agg = st.selectbox(
                        "Aggregation",
                        ["Sum", "Average", "Count", "Max", "Min", "Median"],
                        key="card_agg"
                    )
                
                with kpi_cols[3]:
                    card_title = st.text_input(
                        "KPI Title",
                        value=f"{card_agg} of {card_column}" if 'card_column' in dir() else "",
                        key="card_title"
                    )
                
                if st.button("➕ Add KPI Card", use_container_width=True):
                    if card_dataset and card_column:
                        df = all_datasets[card_dataset]
                        if card_agg == "Sum":
                            value = df[card_column].sum()
                        elif card_agg == "Average":
                            value = df[card_column].mean()
                        elif card_agg == "Count":
                            value = df[card_column].count()
                        elif card_agg == "Max":
                            value = df[card_column].max()
                        elif card_agg == "Median":
                            value = df[card_column].median()
                        else:
                            value = df[card_column].min()
                        
                        card_id = f"card_{current_page_idx}_{len(current_page['cards'])}"
                        st.session_state.dashboard_pages[current_page_idx]["cards"].append({
                            "id": card_id,
                            "title": card_title,
                            "value": value,
                            "column": card_column,
                            "agg": card_agg
                        })
                        
                        st.session_state.dashboard_item_positions[card_id] = {
                            "x": 0,
                            "y": len(current_page["cards"]) * 120,
                            "width": 25,
                            "height": 100
                        }
                        
                        st.success("✅ KPI Card added!")
                        st.rerun()
    
    # ========== DISPLAY DASHBOARD ==========
    st.markdown("---")
    
    layout_type = current_page.get("layout", "freeform")
    
    # ========== DISPLAY KPI CARDS ==========
    cards = current_page.get("cards", [])
    if cards:
        st.markdown(f"### 📊 Key Metrics ({len(cards)})")
        
        # Display cards in responsive grid
        card_cols = st.columns(min(len(cards), 4))
        for idx, card in enumerate(cards):
            with card_cols[idx % 4]:
                card_colors = [
                    "#667eea", "#764ba2", "#f093fb", "#4facfe",
                    "#43e97b", "#fa709a", "#feca57", "#ff9566"
                ]
                color = card_colors[idx % len(card_colors)]
                
                # Format the value properly
                if isinstance(card['value'], float):
                    formatted_value = f"{card['value']:,.2f}"
                else:
                    formatted_value = f"{card['value']:,}"
                
                container_html = f"""
                <div style="background: linear-gradient(135deg, {color} 0%, rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.7) 100%); 
                            padding: 20px; border-radius: 12px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <p style="margin: 0; font-size: 12px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;">{card['title']}</p>
                    <h2 style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{formatted_value}</h2>
                    <p style="margin: 5px 0 0 0; font-size: 10px; opacity: 0.8;">{card['agg']} of {card['column']}</p>
                </div>
                """
                
                st.markdown(container_html, unsafe_allow_html=True)
                
                if st.session_state.dashboard_edit_mode:
                    if st.button("🗑️ Remove", key=f"remove_card_{idx}", use_container_width=True):
                        st.session_state.dashboard_pages[current_page_idx]["cards"].pop(idx)
                        st.rerun()
        
        st.markdown("---")
    
    # ========== DISPLAY CHARTS ==========
    charts = current_page.get("charts", [])
    
    if charts:
        st.markdown(f"### 📈 Visualizations ({len(charts)})")
        
        if layout_type == "freeform":
            # Free-form layout - display in responsive grid with edit controls
            for idx, chart in enumerate(charts):
                with st.container():
                    # Chart controls (edit mode)
                    if st.session_state.dashboard_edit_mode:
                        ctrl_cols = st.columns([2, 1, 1, 1, 1, 1])
                        
                        with ctrl_cols[0]:
                            chart_key = f"chart_title_{current_page_idx}_{idx}"
                            new_title = st.text_input(
                                "Chart Title",
                                value=chart.get('title', ''),
                                key=f"edit_title_{current_page_idx}_{idx}",
                                label_visibility="collapsed"
                            )
                            if new_title != chart.get('title', ''):
                                st.session_state.dashboard_pages[current_page_idx]["charts"][idx]['title'] = new_title
                        
                        with ctrl_cols[1]:
                            if st.button("📖 Edit", key=f"edit_{current_page_idx}_{idx}", use_container_width=True):
                                st.info("📝 To edit this chart, delete and recreate with different settings")
                        
                        with ctrl_cols[2]:
                            width_val = st.slider("Width %", 20, 100, value=chart.get('width', 50), key=f"width_{current_page_idx}_{idx}")
                            if width_val != chart.get('width', 50):
                                st.session_state.dashboard_pages[current_page_idx]["charts"][idx]['width'] = width_val
                        
                        with ctrl_cols[3]:
                            if idx > 0:
                                if st.button("⬆️", key=f"up_{current_page_idx}_{idx}", use_container_width=True):
                                    charts[idx], charts[idx-1] = charts[idx-1], charts[idx]
                                    st.rerun()
                        
                        with ctrl_cols[4]:
                            if idx < len(charts) - 1:
                                if st.button("⬇️", key=f"down_{current_page_idx}_{idx}", use_container_width=True):
                                    charts[idx], charts[idx+1] = charts[idx+1], charts[idx]
                                    st.rerun()
                        
                        with ctrl_cols[5]:
                            if st.button("🗑️", key=f"del_{current_page_idx}_{idx}", use_container_width=True):
                                st.session_state.dashboard_pages[current_page_idx]["charts"].pop(idx)
                                st.rerun()
                    
                    # Display chart with responsive width
                    col_width = chart.get('width', 50) / 100
                    st.plotly_chart(
                        chart['figure'],
                        use_container_width=True,
                        key=f"dash_chart_{current_page_idx}_{idx}"
                    )
                    st.markdown("---")
        
        elif layout_type == "grid_2col":
            # 2-column grid
            col1, col2 = st.columns(2)
            cols = [col1, col2]
            
            for idx, chart in enumerate(charts):
                with cols[idx % 2]:
                    if st.session_state.dashboard_edit_mode:
                        edit_col1, edit_col2 = st.columns([3, 1])
                        with edit_col1:
                            chart['title'] = st.text_input(
                                "Title",
                                value=chart.get('title', ''),
                                key=f"2col_title_{current_page_idx}_{idx}",
                                label_visibility="collapsed"
                            )
                        with edit_col2:
                            if st.button("🗑️", key=f"2col_del_{current_page_idx}_{idx}"):
                                st.session_state.dashboard_pages[current_page_idx]["charts"].pop(idx)
                                st.rerun()
                    
                    st.plotly_chart(chart['figure'], use_container_width=True, key=f"2col_{current_page_idx}_{idx}")
        
        elif layout_type == "grid_3col":
            # 3-column grid
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            
            for idx, chart in enumerate(charts):
                with cols[idx % 3]:
                    if st.session_state.dashboard_edit_mode:
                        edit_col1, edit_col2 = st.columns([3, 1])
                        with edit_col1:
                            chart['title'] = st.text_input(
                                "Title",
                                value=chart.get('title', ''),
                                key=f"3col_title_{current_page_idx}_{idx}",
                                label_visibility="collapsed"
                            )
                        with edit_col2:
                            if st.button("🗑️", key=f"3col_del_{current_page_idx}_{idx}"):
                                st.session_state.dashboard_pages[current_page_idx]["charts"].pop(idx)
                                st.rerun()
                    
                    st.plotly_chart(chart['figure'], use_container_width=True, key=f"3col_{current_page_idx}_{idx}")
        
        else:  # grid_1col
            # Single column
            for idx, chart in enumerate(charts):
                if st.session_state.dashboard_edit_mode:
                    edit_col1, edit_col2 = st.columns([3, 1])
                    with edit_col1:
                        chart['title'] = st.text_input(
                            "Title",
                            value=chart.get('title', ''),
                            key=f"1col_title_{current_page_idx}_{idx}",
                            label_visibility="collapsed"
                        )
                    with edit_col2:
                        if st.button("🗑️", key=f"1col_del_{current_page_idx}_{idx}"):
                            st.session_state.dashboard_pages[current_page_idx]["charts"].pop(idx)
                            st.rerun()
                
                st.plotly_chart(chart['figure'], use_container_width=True, key=f"1col_{current_page_idx}_{idx}")
    
    else:
        if st.session_state.dashboard_edit_mode:
            st.info("💡 Use the 'Add Chart' or 'Add KPI Card' tabs above to get started")
        else:
            st.info("💡 Click '🎛️ Edit Mode' to add visualizations to this dashboard")
    
    # ========== EXPORT & ACTIONS ==========
    st.markdown("---")
    st.markdown("### 📥 Export & Share")
    
    export_cols = st.columns(4)
    
    with export_cols[0]:
        if st.button("📄 Export HTML", use_container_width=True):
            html_content = generate_dashboard_html(st.session_state.dashboard_pages[current_page_idx])
            st.download_button(
                "💾 Download HTML",
                data=html_content,
                file_name=f"dashboard_{current_page['name'].replace(' ', '_')}.html",
                mime="text/html"
            )
    
    with export_cols[1]:
        if st.button("🖨️ Print Mode", use_container_width=True):
            st.info("💡 Use Ctrl+P or Cmd+P → Save as PDF for best results")
    
    with export_cols[2]:
        if st.button("📊 Export Data", use_container_width=True):
            st.info("💡 Data is tied to pinned charts. Charts are linked to live data.")
    
    with export_cols[3]:
        if st.button("💾 Save Config", use_container_width=True):
            st.success("✅ Dashboard configuration saved to session!")


def render_quick_excel_analysis():
    """Quick Excel Analysis page - Upload ONE file for immediate analysis without pipeline"""
    st.title("📁 Quick Excel Analysis")
    st.markdown("### Analyze a single Excel file instantly (for accountants & quick analysis)")
    
    # Initialize session state for quick analysis
    if 'quick_excel_data' not in st.session_state:
        st.session_state.quick_excel_data = None
    if 'quick_excel_sheets' not in st.session_state:
        st.session_state.quick_excel_sheets = []
    if 'quick_excel_context' not in st.session_state:
        st.session_state.quick_excel_context = {
            "purpose": "",
            "role": "",
            "selected_sheets": [],
            "relationships": False
        }
    if 'quick_excel_analysis' not in st.session_state:
        st.session_state.quick_excel_analysis = None
    if 'quick_excel_chat_settings' not in st.session_state:
        st.session_state.quick_excel_chat_settings = {
            "analysis_mode": "deep",
            "response_style": "deep",
            "strict_mode": False,
            "table_scope": "all",
            "prefer_joins": True,
            "temperature": 0.0,
        }
    
    # ========== FILE UPLOAD ==========
    st.markdown("---")
    st.markdown("### 📤 Upload Excel File")
    
    uploaded_file = st.file_uploader(
        "Choose ONE Excel file",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        key="quick_excel_uploader"
    )
    
    if uploaded_file is not None:
        # Load Excel file
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            available_sheets = excel_file.sheet_names
            
            # Store sheets and basic info
            st.session_state.quick_excel_sheets = available_sheets
            
            st.success(f"✅ File loaded: **{uploaded_file.name}**")
            st.info(f"📊 Found **{len(available_sheets)}** sheet(s): {', '.join(available_sheets)}")
            
            # ========== CONTEXT COLLECTION (MANDATORY) ==========
            st.markdown("---")
            st.markdown("### ❓ Context & Purpose (Mandatory Before Analysis)")
            st.markdown("*This helps us analyze your data correctly*")
            
            context_cols = st.columns(2)
            
            with context_cols[0]:
                st.session_state.quick_excel_context["purpose"] = st.text_input(
                    "📌 What is the purpose of this file?",
                    value=st.session_state.quick_excel_context["purpose"],
                    placeholder="e.g., Sales reports, Accounting records, Expense tracking...",
                    key="quick_excel_purpose"
                )
            
            with context_cols[1]:
                roles = ["Accountant", "Manager", "Finance Officer", "Analyst", "Business Owner", "Other"]
                st.session_state.quick_excel_context["role"] = st.selectbox(
                    "👤 What is your role?",
                    roles,
                    index=0 if st.session_state.quick_excel_context["role"] == "" else roles.index(st.session_state.quick_excel_context["role"]),
                    key="quick_excel_role"
                )
            
            # ========== LLM SUGGESTED QUESTIONS ==========
            if st.session_state.quick_excel_context["purpose"]:
                st.markdown("---")
                st.markdown("### 💡 AI-Suggested Questions (Based on Your Purpose)")
                
                # Initialize session state for suggested questions
                if 'quick_suggested_questions' not in st.session_state:
                    st.session_state.quick_suggested_questions = None
                if 'user_custom_questions' not in st.session_state:
                    st.session_state.user_custom_questions = []
                
                # Generate suggested questions if not already generated
                if st.session_state.quick_suggested_questions is None:
                    try:
                        with st.spinner("🤖 Generating AI suggestions based on your purpose..."):
                            suggested_qs = generate_suggested_questions(
                                sheets_data_preview={sheet: excel_file.parse(sheet).head(5) for sheet in available_sheets},
                                purpose=st.session_state.quick_excel_context["purpose"],
                                role=st.session_state.quick_excel_context["role"],
                                available_sheets=available_sheets
                            )
                            st.session_state.quick_suggested_questions = suggested_qs
                    except Exception as e:
                        # Fallback questions if generation fails
                        st.warning(f"⚠️ Could not generate AI suggestions: {str(e)}")
                        st.session_state.quick_suggested_questions = [
                            f"What are the key metrics in {st.session_state.quick_excel_context['purpose']}?",
                            f"What are the top items by value?",
                            f"What data quality issues exist in this file?"
                        ]
                
                # Display suggested questions
                if st.session_state.quick_suggested_questions:
                    st.info(f"💡 Here are 3 questions a {st.session_state.quick_excel_context['role']} might ask about {st.session_state.quick_excel_context['purpose']}:")
                    question_cols = st.columns(1)
                    for idx, q in enumerate(st.session_state.quick_suggested_questions[:3], 1):
                        with st.container():
                            col_q, col_use = st.columns([4, 1])
                            with col_q:
                                st.write(f"**Q{idx}:** {q}")
                            with col_use:
                                if st.button("✅ Use", key=f"use_q{idx}", use_container_width=True):
                                    if q not in st.session_state.user_custom_questions:
                                        st.session_state.user_custom_questions.append(q)
                                    st.success(f"Added to your questions!")
                
                # Custom questions section
                st.markdown("---")
                st.markdown("### ❓ Your Custom Questions")
                
                new_question = st.text_input(
                    "Add your own question:",
                    placeholder="e.g., What is the average revenue per customer?",
                    key="quick_custom_question_input"
                )
                
                col_add, col_clear = st.columns(2)
                with col_add:
                    if st.button("➕ Add Question", use_container_width=True, key="add_custom_q"):
                        if new_question and new_question not in st.session_state.user_custom_questions:
                            st.session_state.user_custom_questions.append(new_question)
                            st.success("Question added!")
                            st.rerun()
                
                with col_clear:
                    if st.button("🗑️ Clear All", use_container_width=True, key="clear_custom_q"):
                        st.session_state.user_custom_questions = []
                        st.rerun()
                
                # Display user's questions
                if st.session_state.user_custom_questions:
                    st.markdown("**Your Questions:**")
                    for idx, q in enumerate(st.session_state.user_custom_questions, 1):
                        col_text, col_remove = st.columns([4, 1])
                        with col_text:
                            st.write(f"{idx}. {q}")
                        with col_remove:
                            if st.button("❌", key=f"remove_q{idx}", use_container_width=True):
                                st.session_state.user_custom_questions.pop(idx - 1)
                                st.rerun()
            
            # ========== SHEET SELECTION ==========
            st.markdown("---")
            st.markdown("### 📋 Select Sheets to Analyze")
            
            if len(available_sheets) == 1:
                st.session_state.quick_excel_context["selected_sheets"] = available_sheets
                st.info(f"✅ Single sheet file - Will analyze: **{available_sheets[0]}**")
            else:
                selected_sheets = st.multiselect(
                    "Which sheets should be analyzed?",
                    available_sheets,
                    default=available_sheets,
                    key="quick_excel_sheets_select"
                )
                st.session_state.quick_excel_context["selected_sheets"] = selected_sheets
            
            if not st.session_state.quick_excel_context["selected_sheets"]:
                st.warning("⚠️ Please select at least one sheet")
                return
            
            # ========== RELATIONSHIPS QUESTION ==========
            st.markdown("---")
            st.markdown("### 🔗 Sheet Relationships")
            
            if len(st.session_state.quick_excel_context["selected_sheets"]) > 1:
                rel_col1, rel_col2 = st.columns(2)
                
                with rel_col1:
                    has_relationships = st.radio(
                        "Are there relationships between sheets?",
                        ["No (Analyze independently)", "Yes (Need to join sheets)"],
                        index=0 if not st.session_state.quick_excel_context["relationships"] else 1,
                        key="quick_excel_relationships"
                    )
                    st.session_state.quick_excel_context["relationships"] = "Yes" in has_relationships
                
                with rel_col2:
                    if st.session_state.quick_excel_context["relationships"]:
                        st.info("""
                        📌 **Important**: Define which columns join the sheets.
                        
                        Example: Join by 'Customer ID' or 'Invoice Number'
                        """)
            else:
                st.session_state.quick_excel_context["relationships"] = False
                st.info("✅ Single sheet selected - No relationships needed")
            
            # ========== LOAD & ANALYZE BUTTON ==========
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🚀 Analyze This File", use_container_width=True, key="quick_analyze_btn"):
                    # Validate context
                    if not st.session_state.quick_excel_context["purpose"]:
                        st.error("❌ Please enter the purpose of this file")
                        return
                    
                    if not st.session_state.quick_excel_context["selected_sheets"]:
                        st.error("❌ Please select at least one sheet")
                        return
                    
                    # Load selected sheets
                    with st.spinner("📊 Loading data..."):
                        sheets_data = {}
                        for sheet_name in st.session_state.quick_excel_context["selected_sheets"]:
                            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                            sheets_data[sheet_name] = df
                        
                        st.session_state.quick_excel_data = sheets_data
                    
                    with st.spinner("🤖 Generating analysis..."):
                        # Generate quick analysis
                        analysis_results = generate_quick_excel_analysis(
                            sheets_data,
                            st.session_state.quick_excel_context
                        )
                        st.session_state.quick_excel_analysis = analysis_results
                    
                    st.success("✅ Analysis complete!")
                    st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
    
    # ========== DISPLAY ANALYSIS RESULTS ==========
    if st.session_state.quick_excel_data and st.session_state.quick_excel_analysis:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        analysis = st.session_state.quick_excel_analysis
        
        # ========== SUMMARY CARDS ==========
        st.markdown("### 📈 Data Summary")
        
        summary_cols = st.columns(len(st.session_state.quick_excel_data))
        for idx, (sheet_name, df) in enumerate(st.session_state.quick_excel_data.items()):
            with summary_cols[idx]:
                st.metric(f"📄 {sheet_name}", f"{len(df):,} rows × {len(df.columns)} cols")
        
        # ========== KEY FINDINGS ==========
        if analysis.get("key_findings"):
            st.markdown("---")
            st.markdown("### 🔍 Key Findings")
            for idx, finding in enumerate(analysis["key_findings"], 1):
                st.write(f"**{idx}. {finding}**")
        
        # ========== DATA QUALITY ISSUES ==========
        if analysis.get("data_quality"):
            st.markdown("---")
            st.markdown("### ⚠️ Data Quality Issues")
            quality_data = analysis["data_quality"]
            
            for sheet_name, issues in quality_data.items():
                with st.expander(f"📋 {sheet_name}", expanded=False):
                    if issues:
                        for issue_type, details in issues.items():
                            st.write(f"**{issue_type}:** {details}")
                    else:
                        st.success("✅ No major quality issues detected")
        
        # ========== VISUALIZATIONS ==========
        if analysis.get("visualizations"):
            st.markdown("---")
            st.markdown("### 📊 Key Visualizations")
            
            viz_cols = st.columns(2)
            for idx, (viz_title, fig) in enumerate(analysis["visualizations"].items()):
                with viz_cols[idx % 2]:
                    st.markdown(f"**{viz_title}**")
                    st.plotly_chart(fig, use_container_width=True)
        
        # ========== RECOMMENDATIONS ==========
        if analysis.get("recommendations"):
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            for idx, rec in enumerate(analysis["recommendations"], 1):
                st.write(f"**{idx}. {rec}**")
        
        # ========== DOWNLOAD OPTIONS ==========
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        export_cols = st.columns(3)
        
        with export_cols[0]:
            if st.button("📄 Export as PDF", use_container_width=True, key="quick_export_pdf"):
                try:
                    from analysis.report_generator import generate_strategic_pdf
                    import tempfile

                    report_data = {
                        "title": f"Quick Excel Analysis - {context.get('purpose', 'Analysis')}",
                        "executive_summary": "\n".join(analysis.get("key_findings", [])) or "Quick Excel analysis summary",
                        "insights": [
                            {"title": f"Finding {idx+1}", "finding": finding}
                            for idx, finding in enumerate(analysis.get("key_findings", [])[:6])
                        ],
                        "recommendations": [
                            {"title": f"Recommendation {idx+1}", "description": rec, "priority": "Medium"}
                            for idx, rec in enumerate(analysis.get("recommendations", [])[:6])
                        ],
                    }

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        output_path = tmp_file.name

                    generate_strategic_pdf(report_data, output_path)
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "💾 Download PDF",
                            data=f,
                            file_name="quick_excel_analysis.pdf",
                            mime="application/pdf",
                            key="quick_excel_pdf_download",
                        )
                    try:
                        os.remove(output_path)
                    except Exception:
                        pass
                except Exception as pdf_error:
                    st.warning(f"Unable to generate PDF directly: {str(pdf_error)}")
                    st.info("💡 Use Ctrl+P / Cmd+P → Save as PDF as fallback")
        
        with export_cols[1]:
            if st.button("📊 Export Summary", use_container_width=True, key="quick_export_summary"):
                summary_text = generate_quick_analysis_summary(
                    st.session_state.quick_excel_data,
                    st.session_state.quick_excel_context,
                    analysis
                )
                st.download_button(
                    "💾 Download Summary",
                    data=summary_text,
                    file_name="analysis_summary.txt",
                    mime="text/plain"
                )
        
        with export_cols[2]:
            if st.button("💬 Chat About Analysis", use_container_width=True, key="quick_chat_btn"):
                st.session_state.current_page = "💬 Enhanced Chatbot"
                st.info("📌 Switching to chatbot... Ask questions about this analysis!")
        
        # ========== CHATBOT SECTION ==========
        st.markdown("---")
        st.markdown("### 💬 Quick Questions")
        st.markdown("*Ask questions about this Excel file (answers will only reference this data)*")

        settings_cols = st.columns(3)
        with settings_cols[0]:
            st.session_state.quick_excel_chat_settings["analysis_mode"] = st.selectbox(
                "Analysis Mode",
                ["fast", "balanced", "deep"],
                index=["fast", "balanced", "deep"].index(st.session_state.quick_excel_chat_settings.get("analysis_mode", "deep")),
                key="quick_excel_chat_analysis_mode"
            )
            st.session_state.quick_excel_chat_settings["table_scope"] = st.selectbox(
                "Table Scope",
                ["all", "working"],
                index=["all", "working"].index(st.session_state.quick_excel_chat_settings.get("table_scope", "all")),
                key="quick_excel_chat_table_scope"
            )
        with settings_cols[1]:
            st.session_state.quick_excel_chat_settings["response_style"] = st.selectbox(
                "Response Style",
                ["executive", "technical", "deep"],
                index=["executive", "technical", "deep"].index(st.session_state.quick_excel_chat_settings.get("response_style", "deep")),
                key="quick_excel_chat_response_style"
            )
            st.session_state.quick_excel_chat_settings["prefer_joins"] = st.checkbox(
                "Prefer Joins",
                value=bool(st.session_state.quick_excel_chat_settings.get("prefer_joins", True)),
                key="quick_excel_chat_prefer_joins"
            )
        with settings_cols[2]:
            st.session_state.quick_excel_chat_settings["strict_mode"] = st.checkbox(
                "Strict Evidence",
                value=bool(st.session_state.quick_excel_chat_settings.get("strict_mode", False)),
                key="quick_excel_chat_strict_mode"
            )
            st.session_state.quick_excel_chat_settings["temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.quick_excel_chat_settings.get("temperature", 0.0)),
                step=0.1,
                key="quick_excel_chat_temperature"
            )
        
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., What are the top 3 categories? Show me the trend...",
            key="quick_excel_question"
        )
        
        if user_question:
            with st.spinner("🤖 Analyzing..."):
                answer = answer_quick_excel_question(
                    user_question,
                    st.session_state.quick_excel_data,
                    st.session_state.quick_excel_context,
                    analysis,
                    st.session_state.quick_excel_chat_settings,
                )
            
            st.markdown("---")
            st.markdown("### 📝 Answer")
            st.write(answer)
    
    elif uploaded_file is not None:
        st.info("💡 Click 'Analyze This File' above after filling in the context to get started")
    
    else:
        st.info("📤 Upload an Excel file to begin analysis")


def generate_quick_excel_analysis(sheets_data: dict, context: dict) -> dict:
    """Generate comprehensive quick analysis for uploaded Excel sheets"""
    analysis = {
        "key_findings": [],
        "data_quality": {},
        "visualizations": {},
        "recommendations": []
    }
    
    try:
        # ========== COMPREHENSIVE DATA ANALYSIS ==========
        for sheet_name, df in sheets_data.items():
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Data quality assessment - DETAILED
            issues = {}
            
            # Missing values analysis
            missing_pct = (df.isnull().sum() / len(df) * 100)
            high_missing = missing_pct[missing_pct > 10]
            medium_missing = missing_pct[(missing_pct > 5) & (missing_pct <= 10)]
            
            if len(high_missing) > 0:
                issues["⚠️ High Missing Data (>10%)"] = ", ".join([f"{col}: {missing_pct[col]:.1f}%" for col in high_missing.index])
            if len(medium_missing) > 0:
                issues["⚠️ Medium Missing Data (5-10%)"] = ", ".join([f"{col}: {missing_pct[col]:.1f}%" for col in medium_missing.index])
            
            # Duplicates analysis
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                issues["🔄 Duplicate Rows"] = f"{dup_count} duplicates ({dup_count/len(df)*100:.1f}% of data)"
            
            # Data type issues
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check for numeric values stored as text
                    try:
                        pd.to_numeric(df[col], errors='coerce')
                        non_numeric = df[col].notna().sum() - pd.to_numeric(df[col], errors='coerce').notna().sum()
                        if non_numeric > 0 and non_numeric < len(df) * 0.5:
                            issues[f"⚠️ Potential Type Issue: {col}"] = f"Stored as text, may be numeric"
                    except:
                        pass
            
            analysis["data_quality"][sheet_name] = issues if issues else {"✅ Data Quality": "No major issues detected"}
            
            # ========== INTELLIGENT VISUALIZATIONS ==========
            viz_count = 0
            
            try:
                role = context.get('role', 'Business Owner')
                
                # ROLE-SPECIFIC VISUALIZATIONS
                if role == 'Accountant':
                    # Accountant: Show data accuracy and totals
                    if len(numeric_cols) > 0 and viz_count < 2:
                        # Top values by category for verification
                        if len(cat_cols) > 0:
                            top_data = df.groupby(cat_cols[0])[numeric_cols[0]].sum().nlargest(10).reset_index()
                            fig = px.bar(
                                top_data,
                                x=cat_cols[0],
                                y=numeric_cols[0],
                                title=f"💰 Top {numeric_cols[0]} by {cat_cols[0]} (For Verification)",
                                labels={cat_cols[0]: cat_cols[0], numeric_cols[0]: numeric_cols[0]}
                            )
                            fig = apply_color_palette_to_figure(fig)
                            analysis["visualizations"][f"{sheet_name} - {numeric_cols[0]} Breakdown"] = fig
                            viz_count += 1
                        
                        # Distribution to check for outliers
                        fig = px.box(
                            df,
                            y=numeric_cols[0],
                            title=f"📊 {numeric_cols[0]} Range (Outlier Detection)",
                            labels={numeric_cols[0]: numeric_cols[0]}
                        )
                        fig = apply_color_palette_to_figure(fig)
                        analysis["visualizations"][f"{sheet_name} - {numeric_cols[0]} Range"] = fig
                        viz_count += 1
                
                elif role == 'Manager':
                    # Manager: Show trends and performance
                    if len(numeric_cols) >= 2 and viz_count < 2:
                        # Scatter for correlation
                        fig = px.scatter(
                            df,
                            x=numeric_cols[0],
                            y=numeric_cols[1],
                            title=f"📈 Performance: {numeric_cols[0]} vs {numeric_cols[1]}",
                            labels={numeric_cols[0]: numeric_cols[0], numeric_cols[1]: numeric_cols[1]}
                        )
                        fig = apply_color_palette_to_figure(fig)
                        analysis["visualizations"][f"{sheet_name} - Performance Correlation"] = fig
                        viz_count += 1
                    
                    if len(cat_cols) > 0 and len(numeric_cols) > 0 and viz_count < 2:
                        # Top performers
                        top_data = df.groupby(cat_cols[0])[numeric_cols[0]].sum().nlargest(10).reset_index()
                        fig = px.bar(
                            top_data,
                            x=cat_cols[0],
                            y=numeric_cols[0],
                            title=f"🏆 Top Performers by {numeric_cols[0]}",
                            labels={cat_cols[0]: cat_cols[0], numeric_cols[0]: numeric_cols[0]}
                        )
                        fig = apply_color_palette_to_figure(fig)
                        analysis["visualizations"][f"{sheet_name} - Top Performers"] = fig
                        viz_count += 1
                
                elif role == 'Finance Officer':
                    # Finance Officer: Revenue, cost, profitability
                    if len(numeric_cols) > 0:
                        # Revenue breakdown by category
                        if len(cat_cols) > 0 and viz_count < 2:
                            revenue_data = df.groupby(cat_cols[0])[numeric_cols[0]].sum().nlargest(15).reset_index()
                            fig = px.pie(
                                values=revenue_data[numeric_cols[0]],
                                names=revenue_data[cat_cols[0]],
                                title=f"💵 Revenue Distribution by {cat_cols[0]}"
                            )
                            fig = apply_color_palette_to_figure(fig)
                            analysis["visualizations"][f"{sheet_name} - Revenue Distribution"] = fig
                            viz_count += 1
                        
                        # Cumulative revenue trend
                        if viz_count < 2:
                            cumsum = df[numeric_cols[0]].cumsum()
                            fig = px.line(
                                x=range(len(cumsum)),
                                y=cumsum,
                                title=f"📊 Cumulative {numeric_cols[0]} Growth",
                                labels={'x': 'Record #', 'y': f'Cumulative {numeric_cols[0]}'}
                            )
                            fig = apply_color_palette_to_figure(fig)
                            analysis["visualizations"][f"{sheet_name} - Revenue Growth"] = fig
                            viz_count += 1
                
                elif role == 'Analyst':
                    # Analyst: Patterns and correlations
                    if len(numeric_cols) >= 2 and viz_count < 2:
                        fig = px.scatter(
                            df,
                            x=numeric_cols[0],
                            y=numeric_cols[1],
                            title=f"🔍 Correlation: {numeric_cols[0]} vs {numeric_cols[1]}",
                            labels={numeric_cols[0]: numeric_cols[0], numeric_cols[1]: numeric_cols[1]}
                        )
                        fig = apply_color_palette_to_figure(fig)
                        analysis["visualizations"][f"{sheet_name} - Data Correlation"] = fig
                        viz_count += 1
                    
                    if len(numeric_cols) > 0 and viz_count < 2:
                        fig = px.histogram(
                            df,
                            x=numeric_cols[0],
                            nbins=30,
                            title=f"📊 Distribution Pattern: {numeric_cols[0]}",
                            labels={'count': 'Frequency', numeric_cols[0]: numeric_cols[0]}
                        )
                        fig = apply_color_palette_to_figure(fig)
                        analysis["visualizations"][f"{sheet_name} - Distribution Pattern"] = fig
                        viz_count += 1
                
                elif role == 'HR':
                    # HR: Headcount, departments, performance
                    if len(cat_cols) > 0 and viz_count < 2:
                        # Department/Category breakdown
                        cat_counts = df[cat_cols[0]].value_counts().head(10)
                        fig = px.bar(
                            x=cat_counts.index,
                            y=cat_counts.values,
                            title=f"👥 Count by {cat_cols[0]}",
                            labels={'x': cat_cols[0], 'y': 'Count'}
                        )
                        fig = apply_color_palette_to_figure(fig)
                        analysis["visualizations"][f"{sheet_name} - {cat_cols[0]} Breakdown"] = fig
                        viz_count += 1
                
                else:  # Business Owner or Other
                    # Business Owner: Revenue, profitability, growth
                    if len(numeric_cols) > 0 and viz_count < 2:
                        if len(cat_cols) > 0:
                            revenue_data = df.groupby(cat_cols[0])[numeric_cols[0]].sum().nlargest(10).reset_index()
                            fig = px.bar(
                                revenue_data,
                                x=cat_cols[0],
                                y=numeric_cols[0],
                                title=f"💹 Revenue by {cat_cols[0]}",
                                labels={cat_cols[0]: cat_cols[0], numeric_cols[0]: numeric_cols[0]}
                            )
                            fig = apply_color_palette_to_figure(fig)
                            analysis["visualizations"][f"{sheet_name} - Revenue Leaders"] = fig
                            viz_count += 1
                    
                    # Growth trend
                    if len(numeric_cols) > 0 and viz_count < 2:
                        cumsum = df[numeric_cols[0]].cumsum()
                        fig = px.line(
                            x=range(len(cumsum)),
                            y=cumsum,
                            title=f"📈 Growth Trend",
                            labels={'x': 'Record #', 'y': 'Cumulative Value'}
                        )
                        fig = apply_color_palette_to_figure(fig)
                        analysis["visualizations"][f"{sheet_name} - Growth Trend"] = fig
                        viz_count += 1
                
            except Exception as viz_error:
                # Fallback to basic visualizations
                try:
                    if len(numeric_cols) > 0 and viz_count < 2:
                        fig = px.histogram(
                            df,
                            x=numeric_cols[0],
                            nbins=30,
                            title=f"📊 Distribution of {numeric_cols[0]}"
                        )
                        fig = apply_color_palette_to_figure(fig)
                        analysis["visualizations"][f"{sheet_name} - Distribution"] = fig
                        viz_count += 1
                except:
                    pass
        
        # ========== LLM-POWERED KEY FINDINGS ==========
        try:
            import ollama
            
            # Prepare detailed data summary for LLM
            data_summary = []
            for sheet_name, df in sheets_data.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                summary_text = f"""
SHEET: {sheet_name}
SIZE: {len(df):,} rows × {len(df.columns)} columns

NUMERIC COLUMNS INSIGHTS:"""
                
                if numeric_cols:
                    for col in numeric_cols[:5]:
                        col_data = df[col].dropna()
                        summary_text += f"""
  • {col}: Sum={col_data.sum():,.0f}, Avg={col_data.mean():,.2f}, Min={col_data.min():,.2f}, Max={col_data.max():,.2f}"""
                else:
                    summary_text += "\n  • No numeric columns"
                
                if cat_cols:
                    summary_text += f"\n\nCATEGORICAL COLUMNS: {', '.join(cat_cols[:5])}"
                
                data_summary.append(summary_text)
            
            # Create smart LLM prompt - ROLE-SPECIFIC
            role = context.get('role', 'Business Owner')
            purpose = context.get('purpose', 'General analysis')
            
            # Create role-specific instructions
            role_context = {
                'Accountant': 'Focus on financial metrics, accuracy, and compliance issues',
                'Manager': 'Focus on performance metrics, trends, and actionable insights',
                'Finance Officer': 'Focus on revenue, costs, profitability, and cash flow',
                'Analyst': 'Focus on patterns, correlations, and data-driven insights',
                'Business Owner': 'Focus on profitability, growth opportunities, and risks',
                'HR': 'Focus on headcount, performance, turnover, and costs',
                'Other': 'Focus on the most important business metrics'
            }
            
            role_focus = role_context.get(role, role_context['Other'])
            
            llm_prompt = f"""You are a data analyst expert helping a {role}.

FILE PURPOSE: {purpose}
ROLE: {role}
FOCUS: {role_focus}

DETAILED DATA ANALYSIS:
{''.join(data_summary)}

TASK: Provide 3-4 KEY FINDINGS that are:
1. Specific, measurable, and with actual numbers (not generic)
2. Business-relevant for a {role} working with {purpose}
3. Actionable insights that drive decisions
4. Clear and concise - one key point per finding

Format: 
- Start each finding with a number (1. 2. 3. etc)
- Write complete sentences (not bullet points)
- Include specific metrics and values
- End with actionable implication

Example format:
1. Total revenue is $50,000 with average transaction of $125, indicating strong customer purchasing power.
2. Supplier A contributes 60% of revenue compared to Supplier B at 25%, showing supplier concentration risk.

Your findings:"""
            
            response = ollama.chat(model='qwen2.5:7b', messages=[
                {'role': 'user', 'content': llm_prompt}
            ])
            
            findings_text = response['message']['content'].strip()
            # Parse findings - split by numbering and clean up
            findings = []
            for part in findings_text.split('\n'):
                part = part.strip()
                # Match lines starting with numbers
                if part and any(part.startswith(f"{i}.") for i in range(1, 10)):
                    # Remove the number prefix and clean up
                    finding = part.split('. ', 1)[-1].strip() if '. ' in part else part
                    if len(finding) > 20:
                        findings.append(finding)
            
            analysis["key_findings"] = findings[:4]  # Keep up to 4 findings

            
        except Exception as llm_error:
            # Smart fallback findings based on actual data
            all_findings = []
            for sheet_name, df in sheets_data.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if numeric_cols:
                    col = numeric_cols[0]
                    total = df[col].sum()
                    avg = df[col].mean()
                    all_findings.append(f"📊 {sheet_name}: Total {col} = {total:,.0f}, Average = {avg:,.2f}")
                
                if cat_cols:
                    col = cat_cols[0]
                    unique = df[col].nunique()
                    all_findings.append(f"🏷️ {sheet_name}: {unique:,} unique values in {col}")
                
                missing = df.isnull().sum().sum()
                if missing > 0:
                    all_findings.append(f"⚠️ {sheet_name}: {missing:,} missing values across dataset")
            
            analysis["key_findings"] = all_findings[:4]
        
        # ========== SMART RECOMMENDATIONS ==========
        try:
            import ollama
            
            all_issues = list(analysis["data_quality"].values())
            rec_prompt = f"""As a data analyst for a {context.get('role', 'Business Owner')}, 
give 3 specific, actionable NEXT STEPS for analyzing {context.get('purpose', 'this data')}.

Keep recommendations brief, practical, and focused on value. Format as numbered list."""
            
            response = ollama.chat(model='qwen2.5:7b', messages=[
                {'role': 'user', 'content': rec_prompt}
            ])
            
            recs_text = response['message']['content']
            recs = [line.strip() for line in recs_text.split('\n') 
                   if line.strip() and any(c.isalnum() for c in line)][:3]
            analysis["recommendations"] = recs
            
        except:
            analysis["recommendations"] = [
                f"✅ Review the {len(sheets_data)} sheet(s) data quality issues highlighted above",
                f"📊 Explore relationships using 'Quick Questions' section",
                f"💡 Ask specific questions about your {context.get('purpose', 'data')} to get deeper insights"
            ]
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
    
    return analysis


def generate_quick_analysis_summary(sheets_data: dict, context: dict, analysis: dict) -> str:
    """Generate text summary of quick analysis"""
    summary = f"""
QUICK EXCEL ANALYSIS SUMMARY
============================

FILE INFORMATION:
- Purpose: {context.get('purpose', 'Not specified')}
- Role: {context.get('role', 'Not specified')}
- Sheets Analyzed: {', '.join(context.get('selected_sheets', []))}
- Multi-sheet Relationships: {'Yes' if context.get('relationships') else 'No'}

DATA SUMMARY:
"""
    
    for sheet_name, df in sheets_data.items():
        summary += f"""
{sheet_name}:
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""
    
    if analysis.get("key_findings"):
        summary += "\n\nKEY FINDINGS:\n"
        for idx, finding in enumerate(analysis["key_findings"], 1):
            summary += f"{idx}. {finding}\n"
    
    if analysis.get("recommendations"):
        summary += "\n\nRECOMMENDATIONS:\n"
        for idx, rec in enumerate(analysis["recommendations"], 1):
            summary += f"{idx}. {rec}\n"
    
    return summary


def _quick_is_id_like(series: pd.Series, col_name: str) -> bool:
    name = str(col_name).lower()
    tokens = [t for t in re.split(r'[^a-z0-9]+', name) if t]
    if name in {'id', 'uuid', 'guid'} or name.endswith('_id'):
        return True
    if any(t in {'id', 'key', 'pk', 'fk', 'uuid', 'guid', 'rowguid'} for t in tokens):
        return True
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    uniq_ratio = float(non_null.nunique()) / float(len(non_null))
    is_int_like = pd.api.types.is_integer_dtype(series)
    is_text_like = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    if len(non_null) >= 50 and uniq_ratio >= 0.995 and (is_int_like or is_text_like):
        return True
    return is_int_like and uniq_ratio >= 0.98


def _quick_is_date_like(series: pd.Series, col_name: str) -> bool:
    name = str(col_name).lower()
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if any(tok in name for tok in ['date', 'time', 'year', 'month', 'day']):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().head(100)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors='coerce')
        return float(parsed.notna().mean()) >= 0.6
    return False


def _quick_classify_columns(df: pd.DataFrame) -> dict:
    identifiers, measures, dimensions, dates = [], [], [], []
    for col in df.columns:
        s = df[col]
        if _quick_is_id_like(s, col):
            identifiers.append(col)
            continue
        if _quick_is_date_like(s, col):
            dates.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s):
            measures.append(col)
        else:
            dimensions.append(col)
    return {
        "identifiers": identifiers,
        "measures": measures,
        "dimensions": dimensions,
        "dates": dates,
    }


def _quick_join_hints(sheets_data: dict) -> list:
    hints = []
    items = list(sheets_data.items())
    for i in range(len(items)):
        left_name, left_df = items[i]
        left_cols = set(left_df.columns)
        for j in range(i + 1, len(items)):
            right_name, right_df = items[j]
            common = [c for c in left_cols.intersection(set(right_df.columns)) if str(c).strip()]
            for col in common[:5]:
                if _quick_is_id_like(left_df[col], col) or _quick_is_id_like(right_df[col], col):
                    hints.append(f"{left_name}[{col}] ↔ {right_name}[{col}] (same key-like column)")
                elif str(col).lower().endswith('date') or str(col).lower().endswith('time'):
                    hints.append(f"{left_name}[{col}] ↔ {right_name}[{col}] (shared temporal field)")
                else:
                    hints.append(f"{left_name}[{col}] ↔ {right_name}[{col}] (shared business field)")
    return list(dict.fromkeys(hints))[:12]


def answer_quick_excel_question(question: str, sheets_data: dict, context: dict, analysis: dict, chat_settings: dict = None) -> str:
    try:
        import ollama

        cfg = {
            "analysis_mode": "deep",
            "response_style": "deep",
            "strict_mode": False,
            "table_scope": "all",
            "prefer_joins": True,
            "temperature": 0.0,
        }
        if isinstance(chat_settings, dict):
            cfg.update(chat_settings)
        
        # Prepare comprehensive data context with actual analysis
        data_context = []
        semantic_context = []
        
        for sheet_name, df in sheets_data.items():
            # Get column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            sheet_info = f"""
SHEET: {sheet_name}
─────────────────────
📊 DIMENSIONS:
- Total rows: {len(df):,}
- Total columns: {len(df.columns)}
- Column names: {', '.join(df.columns.tolist())}

📈 NUMERIC COLUMNS SUMMARY:"""
            
            # Add numeric column statistics
            if numeric_cols:
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        sheet_info += f"""
  • {col}:
    - Count: {len(col_data):,}
    - Sum: {col_data.sum():,.2f}
    - Average: {col_data.mean():,.2f}
    - Min: {col_data.min():,.2f}
    - Max: {col_data.max():,.2f}
    - Median: {col_data.median():,.2f}"""
            else:
                sheet_info += "\n  - No numeric columns"
            
            # Add categorical data summary
            sheet_info += "\n\n🏷️ CATEGORICAL COLUMNS SUMMARY:"
            if cat_cols:
                for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                    unique_count = df[col].nunique()
                    sheet_info += f"\n  • {col}: {unique_count} unique values"
                    if unique_count <= 10:
                        top_values = df[col].value_counts().head(5)
                        for val, count in top_values.items():
                            sheet_info += f"\n    - {val}: {count} ({count/len(df)*100:.1f}%)"
            else:
                sheet_info += "\n  - No categorical columns"
            
            # Add data quality info
            sheet_info += f"""

⚠️ DATA QUALITY:
- Missing values: {df.isnull().sum().sum()} total
- Duplicate rows: {df.duplicated().sum()}"""
            
            # Show sample of top rows
            sheet_info += f"""

📋 SAMPLE DATA (First 5 rows):
{df.head(5).to_string()}"""
            
            data_context.append(sheet_info)

            classes = _quick_classify_columns(df)
            semantic_context.append(
                f"SHEET {sheet_name}: "
                f"Identifiers={classes['identifiers'][:6] or ['None']} | "
                f"Measures={classes['measures'][:6] or ['None']} | "
                f"Dimensions={classes['dimensions'][:6] or ['None']} | "
                f"Dates={classes['dates'][:6] or ['None']}"
            )

        join_hints = _quick_join_hints(sheets_data) if cfg.get("table_scope") == "all" else []

        mode_directive = {
            "fast": "Provide direct concise answer with essential evidence only.",
            "balanced": "Provide concise answer with supporting evidence and one recommendation.",
            "deep": "Provide deep professional analysis with assumptions, risks, and next actions.",
        }.get(str(cfg.get("analysis_mode", "deep")), "Provide deep professional analysis.")

        style_directive = {
            "executive": "Format as: Executive Answer, Evidence, Recommendation.",
            "technical": "Format as: Method, Calculations, Findings, Caveats.",
            "deep": "Format as: Executive Answer, Evidence, Join Path, Assumptions, Risks, Next Actions.",
        }.get(str(cfg.get("response_style", "deep")), "Format as: Executive Answer, Evidence, Recommendation.")
        
        # Create an intelligent prompt
        llm_prompt = f"""You are a senior Data Engineer + Data Analyst + Data Scientist helping a {context.get('role', 'analyst')}.

PURPOSE OF FILE: {context.get('purpose', 'General analysis')}

ANALYSIS MODE: {cfg.get('analysis_mode', 'deep')}
RESPONSE STYLE: {cfg.get('response_style', 'deep')}
USE ALL TABLES: {cfg.get('table_scope', 'all') == 'all'}
PREFER JOINS: {bool(cfg.get('prefer_joins', True))}

COMPLETE DATA CONTEXT:
{"=" * 60}
{chr(10).join(data_context)}
{"=" * 60}

SEMANTIC COLUMN CLASSIFICATION:
{chr(10).join(semantic_context)}

JOIN CANDIDATES:
{chr(10).join(join_hints) if join_hints else 'No explicit join hints detected.'}

USER QUESTION: {question}

YOUR TASK:
1. Answer ONLY using the data provided.
2. If the answer needs multiple sheets, explain the join path clearly.
3. Prefer measure-vs-dimension analysis and avoid treating identifiers as business measures.
4. Show concrete numbers and evidence from data.
5. If information is insufficient, state exactly what is missing.
6. Be decisive and professional.

{mode_directive}
{style_directive}

IMPORTANT: Give a strong, decision-oriented answer that helps the {context.get('role', 'analyst')} act immediately."""
        
        response = ollama.chat(model=st.session_state.get('ollama_model', 'qwen2.5:7b'), messages=[
            {'role': 'user', 'content': llm_prompt}
        ], options={"temperature": float(cfg.get("temperature", 0.0))})
        
        answer_text = response['message']['content']

        if bool(cfg.get("strict_mode", False)):
            if "Evidence" not in answer_text and "evidence" not in answer_text:
                answer_text = (
                    answer_text.rstrip()
                    + "\n\nEvidence Note: Please verify this answer against the shown sheet summaries and key metrics before operational use."
                )

        return answer_text
    
    except Exception as e:
        return f"Unable to answer: {str(e)}\n\nTry asking a simpler question or checking if the data contains the information you're looking for."


def generate_suggested_questions(sheets_data_preview: dict, purpose: str, role: str, available_sheets: list) -> list:
    """Generate 3 AI-suggested questions based on file purpose and role using LLM"""
    try:
        import ollama
        
        # Prepare data preview
        data_info = []
        for sheet_name, df_preview in sheets_data_preview.items():
            numeric_cols = df_preview.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df_preview.select_dtypes(include=['object', 'category']).columns.tolist()
            
            data_info.append(f"""
Sheet: {sheet_name}
- Numeric columns: {', '.join(numeric_cols[:5]) if numeric_cols else 'None'}
- Categorical columns: {', '.join(cat_cols[:5]) if cat_cols else 'None'}
- Sample: {', '.join(df_preview.columns.tolist()[:10])}
""")
        
        llm_prompt = f"""You are helping a {role} analyze an Excel file.

PURPOSE: {purpose}
SHEETS: {', '.join(available_sheets)}

DATA COLUMNS:
{chr(10).join(data_info)}

Generate EXACTLY 3 specific, practical questions that this {role} would likely want to ask about this data.

Format your response as:
1. First question here?
2. Second question here?
3. Third question here?

Questions should be:
- Specific to their role as {role}
- Related to the purpose: {purpose}
- Answerable from the available data
- Business-focused and actionable"""
        
        response = ollama.chat(model='qwen2.5:7b', messages=[
            {'role': 'user', 'content': llm_prompt}
        ])
        
        # Parse response into questions
        response_text = response['message']['content']
        questions = []
        
        for line in response_text.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 4)):
                # Remove numbering
                q = line.split('. ', 1)[-1].strip()
                if q:
                    questions.append(q)
        
        # Return up to 3 questions
        return questions[:3] if questions else [
            f"What are the key trends in the {purpose}?",
            f"What data quality issues exist in this file?",
            f"What are the top items by value in this {purpose}?"
        ]
    
    except Exception as e:
        # Fallback questions
        return [
            f"What are the top 10 items by value in {purpose}?",
            f"What is the total and average values in this {purpose} file?",
            f"Are there any missing values or data quality issues?"
        ]


def generate_ai_dashboard_layout(datasets: dict, goals: dict) -> list:
    """Use AI to generate intelligent dashboard layout"""
    pages = []
    
    try:
        import ollama
        
        # Prepare data summary
        data_info = []
        for name, df in datasets.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:5]
            data_info.append(f"Dataset '{name}': {df.shape[0]} rows, Numeric: {numeric_cols}, Categories: {cat_cols}")
        
        prompt = f"""
You are a dashboard designer. Create a Power BI style dashboard layout.

Data Available:
{chr(10).join(data_info)}

Business Goals:
- Problem: {goals.get('problem', 'General Analysis')}
- Objective: {goals.get('objective', 'Understand data')}

Create 2-3 dashboard pages with:
1. Page name
2. KPI cards (3-4 important metrics)
3. Charts (2-3 charts per page with specific columns)

Respond in this exact format:
PAGE: Overview
CARD: Total Sales | SalesAmount | Sum
CARD: Avg Order | OrderValue | Average
CHART: Bar | Sales by Region | Region | SalesAmount
CHART: Line | Monthly Trend | Month | Revenue

PAGE: Analysis
...
"""
        
        response = ollama.chat(model='qwen2.5:7b', messages=[
            {'role': 'user', 'content': prompt}
        ])
        
        ai_response = response['message']['content']
        
        # Parse AI response
        current_page = None
        for line in ai_response.split('\n'):
            line = line.strip()
            if line.startswith('PAGE:'):
                if current_page:
                    pages.append(current_page)
                current_page = {
                    "name": line.replace('PAGE:', '').strip(),
                    "charts": [],
                    "cards": [],
                    "layout": "freeform"
                }
            elif line.startswith('CARD:') and current_page:
                parts = line.replace('CARD:', '').split('|')
                if len(parts) >= 3:
                    title = parts[0].strip()
                    col = parts[1].strip()
                    agg = parts[2].strip()
                    
                    # Try to find column in datasets
                    for ds_name, df in datasets.items():
                        if col in df.columns:
                            if agg.lower() == 'sum':
                                value = df[col].sum()
                            elif agg.lower() == 'average':
                                value = df[col].mean()
                            else:
                                value = df[col].count()
                            
                            current_page["cards"].append({
                                "title": title,
                                "value": value,
                                "column": col
                            })
                            break
            
            elif line.startswith('CHART:') and current_page:
                parts = line.replace('CHART:', '').split('|')
                if len(parts) >= 4:
                    chart_type = parts[0].strip()
                    title = parts[1].strip()
                    x_col = parts[2].strip()
                    y_col = parts[3].strip()
                    
                    # Create chart
                    for ds_name, df in datasets.items():
                        if x_col in df.columns and y_col in df.columns:
                            try:
                                if chart_type.lower() == 'bar':
                                    agg_df = df.groupby(x_col)[y_col].sum().nlargest(10).reset_index()
                                    fig = px.bar(agg_df, x=x_col, y=y_col, title=title)
                                elif chart_type.lower() == 'line':
                                    fig = px.line(df.head(100), x=x_col, y=y_col, title=title)
                                elif chart_type.lower() == 'pie':
                                    agg_df = df.groupby(x_col)[y_col].sum().nlargest(10)
                                    fig = px.pie(values=agg_df.values, names=agg_df.index, title=title)
                                else:
                                    fig = px.scatter(df.head(500), x=x_col, y=y_col, title=title)
                                
                                current_page["charts"].append({
                                    "title": title,
                                    "figure": fig,
                                    "type": chart_type
                                })
                            except:
                                pass
                            break
        
        if current_page:
            pages.append(current_page)
        
    except Exception as e:
        # Fallback: Create basic layout from data
        pages = [{"name": "Overview", "charts": [], "cards": [], "layout": "freeform"}]
        
        for ds_name, df in list(datasets.items())[:1]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
            cat_cols = df.select_dtypes(include=['object', 'category']).columns[:2]
            
            # Add cards
            for col in numeric_cols:
                pages[0]["cards"].append({
                    "title": f"Total {col}",
                    "value": df[col].sum(),
                    "column": col,
                    "agg": "Sum"
                })
            
            # Add charts
            if len(cat_cols) > 0 and len(numeric_cols) > 0:
                cat_col = cat_cols[0]
                num_col = numeric_cols[0]
                agg_df = df.groupby(cat_col)[num_col].sum().nlargest(10).reset_index()
                fig = px.bar(agg_df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
                fig = apply_color_palette_to_figure(fig)
                pages[0]["charts"].append({
                    "id": "chart_0_0",
                    "title": f"{num_col} by {cat_col}",
                    "figure": fig,
                    "type": "Bar",
                    "dataset": ds_name,
                    "x": cat_col,
                    "y": num_col,
                    "width": 50
                })
    
    if not pages:
        pages = [{"name": "Overview", "charts": [], "cards": [], "layout": "freeform"}]
    
    return pages


def generate_dashboard_html(page_data) -> str:
    """Generate HTML export of dashboard - supports single page or list of pages"""
    # Handle both single page (dict) and multiple pages (list)
    pages = page_data if isinstance(page_data, list) else [page_data]
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Dashboard Export</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            min-height: 100vh;
        }
        .dashboard { max-width: 1400px; margin: 0 auto; }
        .page { 
            background: white; 
            padding: 30px; 
            margin-bottom: 30px; 
            border-radius: 15px; 
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            page-break-after: always;
        }
        .page-title { 
            font-size: 28px; 
            color: #333; 
            border-bottom: 3px solid #667eea; 
            padding-bottom: 15px; 
            margin-bottom: 30px;
            font-weight: 600;
        }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
        .metric-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 25px; 
            border-radius: 12px; 
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s ease;
        }
        .metric-card:hover { transform: translateY(-5px); }
        .metric-label { font-size: 12px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
        .metric-value { font-size: 32px; font-weight: bold; margin: 10px 0; }
        .metric-detail { font-size: 11px; opacity: 0.8; }
        .charts-section { margin-top: 40px; }
        .charts-title { font-size: 20px; font-weight: 600; color: #333; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #f0f0f0; }
        .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 30px; margin-top: 20px; }
        .chart-container { 
            background: #fafafa; 
            padding: 20px; 
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .chart-title { font-size: 16px; font-weight: 600; margin-bottom: 15px; color: #333; }
        .chart-placeholder { 
            background: white; 
            border: 2px dashed #ddd;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            color: #999;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        footer { text-align: center; margin-top: 40px; color: #666; font-size: 12px; }
        @media print {
            body { background: white; padding: 0; }
            .page { box-shadow: none; border-top: 1px solid #ddd; }
        }
    </style>
</head>
<body>
    <div class="dashboard">
"""
    
    for page in pages:
        html += '<div class="page">'
        html += f'<h1 class="page-title">📊 {page["name"]}</h1>'
        
        # KPI Cards
        if page.get("cards"):
            html += '<div class="metrics">'
            card_colors = [
                "#667eea", "#764ba2", "#f093fb", "#4facfe",
                "#43e97b", "#fa709a", "#feca57", "#ff9566"
            ]
            for idx, card in enumerate(page["cards"]):
                value = f"{card['value']:,.2f}" if isinstance(card['value'], float) else f"{card['value']:,}"
                color = card_colors[idx % len(card_colors)]
                html += f'''
                <div class="metric-card" style="background: linear-gradient(135deg, {color} 0%, rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.7) 100%);">
                    <div class="metric-label">{card['title']}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-detail">{card['agg']} of {card['column']}</div>
                </div>
                '''
            html += '</div>'
        
        # Charts
        if page.get("charts"):
            html += '<div class="charts-section">'
            html += '<div class="charts-title">📈 Visualizations</div>'
            html += '<div class="charts-grid">'
            for chart in page["charts"]:
                html += f'''
                <div class="chart-container">
                    <div class="chart-title">{chart['title']}</div>
                    <div class="chart-placeholder">
                        📊 {chart['type']} Chart<br>
                        <span style="font-size: 11px; margin-top: 10px; display: block;">
                            View in Streamlit for interactive visualization
                        </span>
                    </div>
                </div>
                '''
            html += '</div></div>'
        
        html += '</div>'  # Close page
    
    html += """
    </div>
    <footer>
        <p>Dashboard generated using Streamlit Custom Dashboard Builder</p>
        <p style="margin-top: 10px; font-size: 11px;">For interactive charts, view this dashboard in Streamlit</p>
    </footer>
</body>
</html>
"""
    
    return html

def render_monthly_report_page():
    """Professional Reporting page with PDF/Excel export"""
    st.title("📄 Professional Business Report")
    st.markdown("### Generate PDF/Excel reports for stakeholders")
    
    if not st.session_state.pinned_charts:
        st.warning("⚠️ No items selected for the report. Pin charts first!")
    else:
        st.subheader("📋 Report Content")
        selected_items = []
        for idx, item in enumerate(st.session_state.pinned_charts):
            if st.checkbox(f"Include {item['title']}", value=True, key=f"report_check_{idx}"):
                selected_items.append(item)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 PDF Generation")
            if st.button("📥 Generate Strategic PDF Report"):
                with st.spinner("Compiling professional report..."):
                    from analysis.report_generator import generate_strategic_pdf
                    
                    report_data = {
                        "title": "Monthly Strategic Analysis",
                        "executive_summary": st.session_state.business_goals.get("objective", "Standard Business Analysis"),
                        "insights": [{"title": item['title'], "finding": "Strategic Insight"} for item in selected_items],
                        "recommendations": [{"title": "Optimization", "description": "Based on chart analysis", "priority": "High"}]
                    }
                    
                    output_path = "Business_Report.pdf"
                    generate_strategic_pdf(report_data, output_path)
                    
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="💾 Download PDF Report",
                            data=f,
                            file_name="Business_Report.pdf",
                            mime="application/pdf"
                        )
                        
            if st.button("📄 Print-Friendly Layout"):
                st.info("Rendering report for easy printing (Ctrl+P)...")
                for item in selected_items:
                    st.subheader(item['title'])
                    st.plotly_chart(item['figure'], use_container_width=True)
                    st.markdown("---")
                
        with col2:
            st.subheader("📊 Excel Data Export")
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                found_data = False
                for idx, item in enumerate(selected_items):
                    try:
                        fig_data = item['figure'].data[0]
                        if hasattr(fig_data, 'x') and hasattr(fig_data, 'y'):
                            temp_df = pd.DataFrame({'Label': fig_data.x, 'Value': fig_data.y})
                            sheet_name = f"Chart_{idx+1}"[:31]
                            temp_df.to_excel(writer, index=False, sheet_name=sheet_name)
                            found_data = True
                    except Exception:
                        continue
                if not found_data:
                    pd.DataFrame({"Info": ["No data could be extracted"]}).to_excel(writer, sheet_name="Report")
            
            excel_data = output.getvalue()
            st.download_button(
                label="📥 Download Data as Excel",
                data=excel_data,
                file_name="Business_Report_Data.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
st.set_page_config(
    page_title="Data Analysis & AI Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "selected_datasets" not in st.session_state:
    st.session_state.selected_datasets = {}
if "dataset_names" not in st.session_state:
    st.session_state.dataset_names = []
if "cleaned_datasets" not in st.session_state:
    st.session_state.cleaned_datasets = {}
if "loader" not in st.session_state:
    st.session_state.loader = None
if "multi_file_loader" not in st.session_state:
    st.session_state.multi_file_loader = MultiFileLoader()
if "schema_analyzer" not in st.session_state:
    st.session_state.schema_analyzer = None
if "schema_analysis_complete" not in st.session_state:
    st.session_state.schema_analysis_complete = False
if "schema_analysis_result" not in st.session_state:
    st.session_state.schema_analysis_result = None
if "schema_analyzed" not in st.session_state:
    st.session_state.schema_analyzed = False
if "deleted_relationships" not in st.session_state:
    st.session_state.deleted_relationships = set()
if "automatic_analysis" not in st.session_state:
    st.session_state.automatic_analysis = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "enhanced_chatbot" not in st.session_state:
    st.session_state.enhanced_chatbot = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []
if "pinned_charts" not in st.session_state:
    st.session_state.pinned_charts = []
if "show_pin_success" not in st.session_state:
    st.session_state.show_pin_success = None
if "show_pin_warning" not in st.session_state:
    st.session_state.show_pin_warning = None
if "business_goals" not in st.session_state:
    st.session_state.business_goals = {
        "problem": "",
        "objective": "",
        "target": "",
        "completed": False
    }
if "strategic_analysis_result" not in st.session_state:
    st.session_state.strategic_analysis_result = None
if "strategic_analysis_timestamp" not in st.session_state:
    st.session_state.strategic_analysis_timestamp = None
if "hybrid_chatbot" not in st.session_state:
    st.session_state.hybrid_chatbot = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = None

# ============ SHEET SELECTION STATE ============
# Track which sheets user has selected from multi-sheet files
if "sheet_selection_state" not in st.session_state:
    st.session_state.sheet_selection_state = {}

# ============ VISUALIZATION STATE TRACKING ============
# These track state to prevent unnecessary re-analysis
if "chart_pinned_this_session" not in st.session_state:
    st.session_state.chart_pinned_this_session = False
if "custom_chart_builder_selection" not in st.session_state:
    st.session_state.custom_chart_builder_selection = {
        "chart_type": None,
        "use_cross_dataset": False,
        "selected_dataset": None,
        "x_col": None,
        "y_col": None
    }
if "visualization_analysis_cache" not in st.session_state:
    st.session_state.visualization_analysis_cache = {}
if "viz_dataset_state" not in st.session_state:
    st.session_state.viz_dataset_state = None  # Tracks dataset hash to detect changes
if "viz_analysis_in_progress" not in st.session_state:
    st.session_state.viz_analysis_in_progress = False

# Sidebar
st.sidebar.title("📊 Data Analysis & AI Chatbot")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📤 Multi-File Loading", "� Quick Excel Analysis", "�🔗 Schema Analysis", 
     "🎯 Business Goals", "⚠️ Data Quality", "🧹 Data Cleaning",
     "🤖 Strategic AI Analyst", "📊 KPIs Dashboard", "📊 Custom Dashboard",
     "📈 Visualization", "💬 Enhanced Chatbot", "📄 Monthly Report"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application helps SME businesses analyze their data and get AI-powered insights."
)

# Main content
if page == "🏠 Home":
    st.title("📊 Data Analysis & AI Chatbot")
    st.markdown("### Welcome to Your Data Intelligence Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Status", "Ready" if st.session_state.df is not None else "No Data")
    
    with col2:
        if st.session_state.df is not None:
            st.metric("Rows", len(st.session_state.df))
        else:
            st.metric("Rows", "0")
    
    with col3:
        if st.session_state.df is not None:
            st.metric("Columns", len(st.session_state.df.columns))
        else:
            st.metric("Columns", "0")
    
    st.markdown("---")
    
    st.markdown("""
    ### Features
    
    ✅ **Data Loading** - Upload Excel files and preview data
    
    ✅ **Data Understanding** - Automatic EDA with missing values, correlations, and outliers
    
    ✅ **Data Cleaning** - Handle missing values, remove outliers, encode categories
    
    ✅ **Visualization** - Create interactive charts and visualizations
    
    ✅ **AI Chatbot** - Ask questions about your data using AI
    
    ### Getting Started
    
    1. Go to **Data Loading** to upload your Excel file
    2. Use **Data Understanding** to explore your data
    3. Use **Data Cleaning** to prepare your data
    4. Create visualizations in **Visualization**
    5. Ask questions in **AI Chatbot**
    """)

elif page == "📤 Multi-File Loading":
    st.title("📤 Multi-File Loading")
    st.markdown("Upload and manage multiple Excel files and sheets")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose Excel files", 
            type=["xlsx", "xls"], 
            accept_multiple_files=True,
            key="multi_file_uploader"
        )
    
    with col2:
        if st.button("🗑️ Clear All", key="clear_all_files"):
            st.session_state.multi_file_loader = MultiFileLoader()
            st.session_state.df = None
            st.session_state.original_df = None
            st.session_state.selected_datasets = {}
            st.session_state.dataset_names = []
            st.session_state.cleaned_datasets = {}
            st.rerun()
    
    # Color Palette Selection
    st.markdown("---")
    st.subheader("🎨 Choose Color Palette for Visualizations")
    st.markdown("Select a color palette that will be used for **ALL visualizations throughout the app**")
    st.markdown("""
    ⚠️ **Important Rules:**
    - Select one palette and it will remain consistent across all pages
    - Colors will NOT change automatically
    - Palette can only be changed by explicitly selecting a different one below
    - All visualizations will use your selected palette for consistency
    """)
    
    # Initialize color palette in session state if not exists
    if 'selected_color_palette' not in st.session_state:
        st.session_state.selected_color_palette = 'Vibrancy'
    
    # Define color palettes
    color_palettes = {
        'Vibrancy': {
            'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'],
            'description': '🌈 Vibrant & Bold - Perfect for eye-catching dashboards'
        },
        'Ocean': {
            'colors': ['#0066CC', '#0099FF', '#00CCFF', '#006699', '#3399FF', '#66CCFF', '#0052A3', '#004D99'],
            'description': '🌊 Cool Blues & Teals - Professional & Calm'
        },
        'Sunset': {
            'colors': ['#FF6B35', '#F7931E', '#FDB833', '#F37335', '#C1272D', '#F15A24', '#E74C3C', '#EC7063'],
            'description': '🌅 Warm Oranges & Reds - Energetic & Warm'
        },
        'Forest': {
            'colors': ['#2D5016', '#3E7C17', '#6FA876', '#A4C639', '#52B788', '#40916C', '#2D6A4F', '#1B4332'],
            'description': '🌲 Natural Greens - Calm & Trustworthy'
        }
    }
    
    # Show current selection prominently
    st.markdown("---")
    current_palette = st.session_state.selected_color_palette
    col_current, col_info = st.columns([2, 3])
    
    with col_current:
        st.success(f"✅ **Current Palette:** {current_palette}")
    
    with col_info:
        st.info(f"{color_palettes[current_palette]['description']}")
    
    # Create columns for palette selection (for changing)
    st.markdown("**Change palette (if needed):**")
    palette_cols = st.columns(4)
    
    for idx, (palette_name, palette_info) in enumerate(color_palettes.items()):
        with palette_cols[idx]:
            is_selected = st.session_state.selected_color_palette == palette_name
            
            # Create a visual preview of the palette
            col_preview = st.columns(len(palette_info['colors']), gap='small')
            
            for col_idx, color in enumerate(palette_info['colors'][:4]):  # Show first 4 colors
                with col_preview[col_idx]:
                    st.markdown(
                        f"<div style='background-color: {color}; height: 40px; border-radius: 4px; border: {'3px solid black' if is_selected else '1px solid #ccc'};'></div>",
                        unsafe_allow_html=True
                    )
            
            st.write(f"**{palette_name}**")
            st.caption(palette_info['description'])
            
            # Selection button
            if st.button(
                "✅ Currently Selected" if is_selected else "Select",
                key=f"palette_{palette_name}",
                use_container_width=True,
                disabled=is_selected
            ):
                st.session_state.selected_color_palette = palette_name
                st.success(f"✅ تم تغيير الألوان إلى '{palette_name}'! 🎨")
                st.info("✨ تم تطبيق الألوان الجديدة على جميع الصفحات والرسوم البيانية فوراً!")
                st.balloons()
                st.rerun()
    
    st.markdown("---")
    

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if not st.session_state.multi_file_loader.is_file_loaded(uploaded_file.name):
                with st.spinner(f"Loading {uploaded_file.name}..."):
                    success = st.session_state.multi_file_loader.load_file(uploaded_file)
                    if success:
                        st.success(f"✅ Loaded: {uploaded_file.name}")
    
    # Show loaded files
    if st.session_state.multi_file_loader and len(st.session_state.multi_file_loader.get_loaded_files()) > 0:
        st.markdown("---")
        st.subheader("📁 Loaded Files & Sheet Selection")
        
        # Initialize sheet selection state if not exists
        if 'sheet_selection_state' not in st.session_state:
            st.session_state.sheet_selection_state = {}
        
        for file_name in st.session_state.multi_file_loader.get_loaded_files():
            with st.expander(f"📄 {file_name}"):
                sheets = st.session_state.multi_file_loader.get_file_sheets(file_name)
                
                if len(sheets) > 1:
                    st.write(f"**📑 Available Sheets:** {len(sheets)}")
                    
                    # Sheet selection for multi-sheet files
                    st.markdown("**Select sheets to import from this file:**")
                    
                    # Initialize selection for this file if not exists
                    if file_name not in st.session_state.sheet_selection_state:
                        st.session_state.sheet_selection_state[file_name] = sheets.copy()  # Default: select all
                    
                    # Multi-select for sheets
                    selected_sheets = st.multiselect(
                        f"Choose sheets from {file_name}",
                        sheets,
                        default=st.session_state.sheet_selection_state.get(file_name, sheets),
                        key=f"sheet_select_{file_name}",
                        help="Select one or more sheets to work with"
                    )
                    
                    # Update selection state
                    if selected_sheets:
                        st.session_state.sheet_selection_state[file_name] = selected_sheets
                    
                    # Create tabs for each sheet preview
                    if selected_sheets:
                        st.write("**Sheet Previews:**")
                        sheet_tabs = st.tabs(selected_sheets)
                        
                        for sheet_idx, (sheet_name, sheet_tab) in enumerate(zip(selected_sheets, sheet_tabs)):
                            with sheet_tab:
                                df = st.session_state.multi_file_loader.get_sheet_data(file_name, sheet_name)
                                if df is not None:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Rows", f"{len(df):,}")
                                    with col2:
                                        st.metric("Columns", len(df.columns))
                                    with col3:
                                        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                                    
                                    st.dataframe(df.head(10), use_container_width=True)
                    else:
                        st.warning("⚠️ No sheets selected. Select at least one sheet above.")
                else:
                    # Single sheet
                    st.write(f"**📄 Sheet:** {sheets[0]}")
                    df = st.session_state.multi_file_loader.get_sheet_data(file_name, sheets[0])
                    if df is not None:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", f"{len(df):,}")
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                        
                        st.dataframe(df.head(10), use_container_width=True)
        
        # Show loaded files summary
        st.markdown("---")
        loaded_files = st.session_state.multi_file_loader.get_loaded_files()
        
        # Count total selected sheets
        total_sheets_selected = sum(len(st.session_state.sheet_selection_state.get(f, [])) 
                                   for f in loaded_files)
        
        st.success(f"✅ **{len(loaded_files)} file(s) loaded** | **{total_sheets_selected} sheet(s) selected**")
        
        st.info("💡 **Tip:** You can select different sheets from each file. Only selected sheets will be processed in the next steps.")
    else:
        st.info("📤 Upload Excel files to get started")

elif page == "� Quick Excel Analysis":
    render_quick_excel_analysis()

elif page == "�🔗 Schema Analysis":
    st.title("🔗 Schema Analysis & Relationships")
    st.markdown("Discover relationships between tables (Data Modeling View)")
    
    # Check if we have datasets
    if not st.session_state.multi_file_loader or len(st.session_state.multi_file_loader.get_loaded_files()) == 0:
        st.warning("⚠️ Please load data first from 'Multi-File Loading' page")
    else:
        # Get all sheets as separate datasets (uses cleaned versions if available)
        all_datasets = get_all_datasets()
        
        if len(all_datasets) < 2:
            st.warning("⚠️ Please load at least 2 datasets/sheets to analyze relationships")
        else:
            st.success(f"📊 Found {len(all_datasets)} datasets/sheets")
            
            # ========== ANALYZE RELATIONSHIPS ==========
            st.markdown("---")
            st.markdown("### 🔍 Analyze Relationships")
            
            if st.button("🔍 Analyze Relationships", use_container_width=True, key="analyze_rel_main"):
                with st.spinner("Analyzing schema relationships..."):
                    from models.schema_analyzer import SchemaAnalyzer
                    
                    analyzer = SchemaAnalyzer()
                    for name, df in all_datasets.items():
                        analyzer.add_dataset(name, df)
                    
                    relationships = analyzer.analyze_relationships()
                    
                    st.session_state.schema_analyzer = analyzer
                    st.session_state.schema_analysis_complete = True
                    
                    st.rerun()
            
            # Show analysis results if complete
            if st.session_state.schema_analysis_complete and st.session_state.schema_analyzer:
                st.markdown("---")
                
                analyzer = st.session_state.schema_analyzer
                results = analyzer.relationships
                stats = analyzer.get_combined_statistics()
                
                # Initialize deleted relationships
                if 'deleted_relationships' not in st.session_state:
                    st.session_state.deleted_relationships = set()
                
                # Filter deleted relationships
                active_results = [r for i, r in enumerate(results) 
                                  if f"{r.get('file1')}|{r.get('column1')}|{r.get('file2')}|{r.get('column2')}" 
                                  not in st.session_state.deleted_relationships]
                
                # Summary metrics
                st.markdown("**📊 Model Summary:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📋 Tables", stats['total_datasets'])
                with col2:
                    st.metric("🔗 Relationships", len(active_results))
                with col3:
                    st.metric("📊 Total Rows", f"{stats['total_rows']:,}")
                with col4:
                    st.metric("📝 Total Columns", stats['total_columns'])
                
                # ========== ERD DIAGRAM VISUALIZATION ==========
                st.markdown("---")
                st.markdown("### 📐 Entity Relationship Diagram (Power BI Style)")
                st.info("🎨 **Visual data model showing table relationships** - Hover over connections for details")
                
                # Filter relationships by confidence (>= 50%)
                quality_results = [r for r in active_results if r.get('match_percentage', 0) >= 50]
                
                if quality_results:
                    # Create visualization using Plotly instead
                    import plotly.graph_objects as go
                    
                    # Get unique tables
                    tables = set()
                    for rel in quality_results:
                        tables.add(rel['file1'])
                        tables.add(rel['file2'])
                    tables = sorted(list(tables))
                    
                    # Create nodes and edges data for plotly
                    node_x = []
                    node_y = []
                    node_text = []
                    node_labels = []
                    
                    # Position nodes in a circle
                    import math
                    num_tables = len(tables)
                    radius = 350
                    for idx, table in enumerate(tables):
                        angle = 2 * math.pi * idx / max(1, num_tables)
                        x = radius * math.cos(angle)
                        y = radius * math.sin(angle)
                        node_x.append(x)
                        node_y.append(y)
                        
                        df = all_datasets.get(table)
                        row_count = len(df) if df is not None else 0
                        col_count = len(df.columns) if df is not None else 0
                        
                        # Create rich hover text with table info
                        hover_text = f"<b style='font-size:14px'>{table}</b><br>"
                        hover_text += f"<b style='color:#666'>📊 Rows:</b> {row_count:,}<br>"
                        hover_text += f"<b style='color:#666'>📋 Columns:</b> {col_count}<br>"
                        if df is not None and len(df) > 0:
                            hover_text += f"<b style='color:#666'>💾 Size:</b> ~{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
                        node_text.append(hover_text)
                        node_labels.append(table)
                    
                    # Create edge traces
                    edge_traces = []
                    edge_labels = []
                    
                    # Track which edges we've drawn
                    drawn_edges = set()
                    
                    for rel in quality_results:
                        table1 = rel['file1']
                        table2 = rel['file2']
                        
                        edge_key = tuple(sorted([table1, table2]))
                        if edge_key in drawn_edges:
                            continue
                        drawn_edges.add(edge_key)
                        
                        idx1 = tables.index(table1)
                        idx2 = tables.index(table2)
                        
                        x_edge = [node_x[idx1], node_x[idx2], None]
                        y_edge = [node_y[idx1], node_y[idx2], None]
                        
                        # Determine color based on relationship type
                        rel_type = rel.get('relationship_type', 'N/A').upper()
                        match_pct = rel.get('match_percentage', 0)
                        # Width based on match percentage
                        line_width = 2 + (match_pct / 100) * 2
                        line_dash = 'solid' if rel_type in ['1:1', 'ONE_TO_ONE', '1:M', 'ONE_TO_MANY'] else 'dash'
                        
                        if rel_type in ['1:1', 'ONE_TO_ONE']:
                            color = '#2ecc71'  # Green
                            card_label = "🟢 1:1"
                        elif rel_type in ['1:M', 'ONE_TO_MANY']:
                            color = '#3498db'  # Blue
                            card_label = "🔵 1:M"
                        else:
                            color = '#e74c3c'  # Red
                            card_label = "🔴 M:M"
                        
                        edge_trace = go.Scatter(
                            x=x_edge, y=y_edge,
                            mode='lines',
                            line=dict(width=line_width, color=color, dash=line_dash),
                            hoverinfo='none',
                            showlegend=False
                        )
                        edge_traces.append(edge_trace)
                        
                        # Add edge label
                        mid_x = (node_x[idx1] + node_x[idx2]) / 2
                        mid_y = (node_y[idx1] + node_y[idx2]) / 2
                        label_text = f"<b>{card_label}</b><br><i>{match_pct}%</i>"
                        edge_label_trace = go.Scatter(
                            x=[mid_x],
                            y=[mid_y],
                            mode='text',
                            text=[label_text],
                            textfont=dict(size=11, color=color, family='Arial Black'),
                            hoverinfo='text',
                            hovertext=f"{rel['column1']} → {rel['column2']} ({rel.get('match_percentage', 0)}%)",
                            showlegend=False
                        )
                        edge_traces.append(edge_label_trace)
                    
                    # Create node trace
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_labels,
                        textposition="middle center",
                        textfont=dict(size=10, color='black'),
                        hoverinfo='text',
                        hovertext=node_text,
                        marker=dict(
                            size=50,
                            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(node_labels)],
                            line=dict(color='white', width=3),
                            symbol='circle',
                            opacity=0.95
                        ),
                        showlegend=False
                    )
                    
                    # Build figure
                    fig = go.Figure(data=edge_traces + [node_trace])
                    
                    fig.update_layout(
                        title={
                            'text': '<b>📊 Entity Relationship Diagram</b><br><sub>Showing relationships between data tables</sub>',
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'size': 18, 'color': '#2c3e50'}
                        },
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=30, l=30, r=30, t=100),
                        plot_bgcolor='#fafafa',
                        paper_bgcolor='white',
                        height=750,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        font=dict(family="Arial, sans-serif", size=12, color='#2c3e50')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    # Add legend
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("🟢 **1:1** - One-to-One")
                    with col2:
                        st.markdown("🔵 **1:M** - One-to-Many")
                    with col3:
                        st.markdown("🔴 **M:M** - Many-to-Many")
                else:
                    st.info("📊 No high-quality relationships found (Match % ≥ 50%)")
                
                # Show filtered out relationships
                low_quality = [r for r in active_results if r.get('match_percentage', 0) < 50]
                if low_quality:
                    with st.expander(f"⚠️ Low Quality Relationships ({len(low_quality)} filtered out)"):
                        st.caption(f"These relationships have Match % < 50% and are hidden by default")
                        for rel in low_quality:
                            st.write(f"- {rel['file1']}.{rel['column1']} → {rel['file2']}.{rel['column2']} ({rel.get('match_percentage', 0)}% match)")
                
                # ========== RELATIONSHIP DETAILS TABLE ==========
                st.markdown("---")
                st.markdown("### 🔗 Relationship Details (Match % ≥ 50%)")
                
                if quality_results:
                    # Create a detailed relationships table
                    rel_data = []
                    for idx, rel in enumerate(quality_results):
                        conf = rel.get('confidence', 'medium').lower()
                        conf_icon = "🟢" if conf == "high" else ("🟡" if conf == "medium" else "🔴")
                        
                        rel_data.append({
                            "Trust": f"{conf_icon} {conf.upper()}",
                            "Table 1": rel['file1'],
                            "Column 1": rel['column1'],
                            "Type": rel.get('relationship_type', 'N/A'),
                            "Table 2": rel['file2'],
                            "Column 2": rel['column2'],
                            "Match %": f"{rel.get('match_percentage', 0)}%",
                        })
                    
                    rel_df = pd.DataFrame(rel_data)
                    st.dataframe(rel_df, use_container_width=True, hide_index=True)
                    
                    # Delete/Restore relationships
                    with st.expander("⚙️ Manage Relationships"):
                        st.markdown("**Delete specific relationships:**")
                        st.caption("Trust level indicator: 🟢=High | 🟡=Medium | 🔴=Low")
                        
                        st.markdown("---")
                        
                        for idx, rel in enumerate(quality_results):
                            rel_key = f"{rel.get('file1')}|{rel.get('column1')}|{rel.get('file2')}|{rel.get('column2')}"
                            conf = rel.get('confidence', 'medium').lower()
                            conf_icon = "🟢" if conf == "high" else ("🟡" if conf == "medium" else "🔴")
                            
                            rel_display = f"{rel['file1']}.{rel['column1']} → {rel['file2']}.{rel['column2']}"
                            match_pct = rel.get('match_percentage', 0)
                            
                            # Show trust level and match percentage
                            st.markdown(f"**{conf_icon} [{conf.upper()}] {rel_display} ({match_pct}% match)**")
                            
                            if st.checkbox(f"🗑️ Delete this relationship", key=f"del_rel_check_{idx}"):
                                st.session_state.deleted_relationships.add(rel_key)
                                st.rerun()
                            
                            st.markdown("---")
                        
                        # Restore option
                        if st.session_state.deleted_relationships:
                            st.warning(f"⚠️ {len(st.session_state.deleted_relationships)} relationship(s) marked for deletion")
                            if st.button("🔄 Restore All", use_container_width=True):
                                st.session_state.deleted_relationships = set()
                                st.rerun()
                else:
                    st.info("No relationships to display (all filtered or deleted).")
            
            st.markdown("---")
            
            st.markdown("---")

elif page == "⚠️ Data Quality":
    st.title("⚠️ Data Quality Assessment")
    st.markdown("Comprehensive data quality analysis")
    
    # Dataset selector with Select All button
    render_dataset_selector("quality", allow_multi=True, show_select_all=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first from 'Multi-File Loading' page")
    else:
        # Handle multiple datasets
        selected_datasets = st.session_state.get('selected_datasets', {})
        
        if len(selected_datasets) > 1:
            st.info(f"📊 **Analyzing {len(selected_datasets)} datasets**")
            
            # Create tabs for each dataset
            dataset_tabs = st.tabs(list(selected_datasets.keys()))
            
            for tab, (ds_name, ds_df) in zip(dataset_tabs, selected_datasets.items()):
                with tab:
                    assessor = DataQualityAssessor(ds_df)
                    with st.spinner(f"Assessing {ds_name}..."):
                        assessment = assessor.assess_all()
                    
                    # Quality Score
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        score = assessment['quality_score']
                        if score >= 90:
                            color = "🟢"
                        elif score >= 75:
                            color = "🟡"
                        else:
                            color = "🔴"
                        st.metric("Quality Score", f"{color} {score:.1f}/100")
                    with col2:
                        st.metric("Total Issues", assessment['total_issues'])
                    with col3:
                        st.metric("Critical Issues", len(assessment['by_severity']['critical']))
                    
                    st.write(f"**Summary:** {assessment['summary']}")
                    
                    # Priority Actions
                    st.subheader("🎯 Priority Actions")
                    for action in assessment['priority_actions']:
                        st.write(action)
                    
                    # Issues by Severity (simplified for multi-dataset view)
                    with st.expander("📋 View All Issues"):
                        for severity in ['critical', 'high', 'medium', 'low']:
                            if assessment['by_severity'][severity]:
                                st.write(f"**{severity.upper()}:**")
                                for issue in assessment['by_severity'][severity]:
                                    column_info = issue.get('column', 'N/A')
                                    st.write(f"- {issue['type'].replace('_', ' ').title()} in `{column_info}`: {issue['explanation']}")
        else:
            # Single dataset - original behavior
            assessor = DataQualityAssessor(st.session_state.df)
            
            with st.spinner("Assessing data quality..."):
                assessment = assessor.assess_all()
            
            # Quality Score
            st.subheader("📊 Overall Quality Score")
            col1, col2, col3 = st.columns(3)
            with col1:
                score = assessment['quality_score']
                if score >= 90:
                    color = "🟢"
                elif score >= 75:
                    color = "🟡"
                else:
                    color = "🔴"
                st.metric("Quality Score", f"{color} {score:.1f}/100")
            with col2:
                st.metric("Total Issues", assessment['total_issues'])
            with col3:
                st.metric("Critical Issues", len(assessment['by_severity']['critical']))
            
            st.write(f"**Summary:** {assessment['summary']}")
            
            st.markdown("---")
            
            # Priority Actions
            st.subheader("🎯 Priority Actions")
            for action in assessment['priority_actions']:
                st.write(action)
            
            st.markdown("---")
            
            # Issues by Severity
            st.subheader("📋 Issues by Severity")
            
            tabs = st.tabs(["🔴 Critical", "🟠 High", "🟡 Medium", "🟢 Low"])
            
            with tabs[0]:
                if assessment['by_severity']['critical']:
                    for issue in assessment['by_severity']['critical']:
                        column_info = issue.get('column', 'Multiple Columns' if 'col1' in issue else 'N/A')
                        with st.expander(f"❌ {issue['type'].replace('_', ' ').title()} - {column_info}"):
                            st.write(f"**Explanation:** {issue['explanation']}")
                            st.write(f"**Count:** {issue.get('count', 'N/A')}")
                            st.write(f"**Percentage:** {issue.get('percentage', 'N/A'):.1f}%")
                            st.write(f"**Recommendation:** {issue['recommendation']}")
                else:
                    st.success("✅ No critical issues found!")
            
            with tabs[1]:
                if assessment['by_severity']['high']:
                    for issue in assessment['by_severity']['high']:
                        column_info = issue.get('column', 'Multiple Columns' if 'col1' in issue else 'N/A')
                        with st.expander(f"⚠️ {issue['type'].replace('_', ' ').title()} - {column_info}"):
                            st.write(f"**Explanation:** {issue['explanation']}")
                            st.write(f"**Count:** {issue.get('count', 'N/A')}")
                            st.write(f"**Percentage:** {issue.get('percentage', 'N/A'):.1f}%")
                            st.write(f"**Recommendation:** {issue['recommendation']}")
                else:
                    st.success("✅ No high-priority issues found!")
            
            with tabs[2]:
                if assessment['by_severity']['medium']:
                    for issue in assessment['by_severity']['medium']:
                        column_info = issue.get('column', 'Multiple Columns' if 'col1' in issue else 'N/A')
                        with st.expander(f"ℹ️ {issue['type'].replace('_', ' ').title()} - {column_info}"):
                            st.write(f"**Explanation:** {issue['explanation']}")
                            st.write(f"**Recommendation:** {issue['recommendation']}")
                else:
                    st.success("✅ No medium-priority issues found!")
            
            with tabs[3]:
                if assessment['by_severity']['low']:
                    for issue in assessment['by_severity']['low']:
                        column_info = issue.get('column', 'Multiple Columns' if 'col1' in issue else 'N/A')
                        with st.expander(f"💡 {issue['type'].replace('_', ' ').title()} - {column_info}"):
                            st.write(f"**Explanation:** {issue['explanation']}")
                            st.write(f"**Recommendation:** {issue['recommendation']}")
                else:
                    st.success("✅ No low-priority issues found!")
        
        # Final message about Data Cleaning page
        st.markdown("---")
        st.info("💡 **Want to fix these issues?** If you want to clean all issues please move to the 🧹 **Data Cleaning** page, where you can apply intelligent fixes to your data.")

elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning")
    st.markdown("### Sequential Pipeline Logic: Data Cleaning → Merge → Append → Custom Columns")
    st.markdown("#### 🎯 Golden Rule: The next stage receives ONLY what you see in the final preview")
    st.info("💡 All operations work on the latest dataset. Once a transformation is applied, it becomes your new active dataset.")
    
    # Check if data is loaded
    if not st.session_state.multi_file_loader or len(st.session_state.multi_file_loader.get_loaded_files()) == 0:
        st.warning("⚠️ Please load data first from 'Multi-File Loading' page")
    else:
        # Initialize pipeline state management
        if 'data_cleaning_pipeline' not in st.session_state:
            st.session_state.data_cleaning_pipeline = {
                'active_dataset_name': None,          # Name of currently active dataset
                'active_dataset': None,               # The actual dataframe
                'operation_history': [],              # Log of all operations
                'used_in_merge': set(),               # Files used in merge operations
                'used_in_append': set(),              # Files used in append operations
                'final_dataset_preview': None,        # Final dataset ready to load
                'pipeline_state': 'file_cleaning'     # Current stage: file_cleaning → merge → append → custom_columns → final_preview
            }
        
        pipeline = st.session_state.data_cleaning_pipeline
        
        # Initialize cleaned datasets storage if not exists
        if 'cleaned_datasets' not in st.session_state:
            st.session_state.cleaned_datasets = {}
        
        # Get all loaded files
        loaded_files = st.session_state.multi_file_loader.get_loaded_files()
        
        st.success(f"📁 **{len(loaded_files)} file(s) loaded**")
        
        # ============ PIPELINE STATE MANAGEMENT FUNCTIONS ============
        
        def get_available_datasets_for_stage(stage_name):
            """Get datasets available for the current pipeline stage"""
            pipeline = st.session_state.data_cleaning_pipeline
            available = {}
            
            # For file cleaning: all original files
            if stage_name == 'file_cleaning':
                for file_name in loaded_files:
                    sheets = st.session_state.multi_file_loader.get_file_sheets(file_name)
                    for sheet in sheets:
                        dataset_key = f"{file_name} → {sheet}" if len(sheets) > 1 else file_name
                        if dataset_key in st.session_state.cleaned_datasets:
                            available[dataset_key] = st.session_state.cleaned_datasets[dataset_key]
                        else:
                            df = st.session_state.multi_file_loader.get_sheet_data(file_name, sheet)
                            if df is not None:
                                available[dataset_key] = df
            
            # For merge: all cleaned files except those already used
            elif stage_name == 'merge':
                for file_name in loaded_files:
                    sheets = st.session_state.multi_file_loader.get_file_sheets(file_name)
                    for sheet in sheets:
                        dataset_key = f"{file_name} → {sheet}" if len(sheets) > 1 else file_name
                        if dataset_key not in pipeline['used_in_merge']:  # Exclude already-merged files
                            if dataset_key in st.session_state.cleaned_datasets:
                                available[dataset_key] = st.session_state.cleaned_datasets[dataset_key]
                            else:
                                df = st.session_state.multi_file_loader.get_sheet_data(file_name, sheet)
                                if df is not None:
                                    available[dataset_key] = df
            
            # For append: all datasets except those already used
            elif stage_name == 'append':
                for file_name in loaded_files:
                    sheets = st.session_state.multi_file_loader.get_file_sheets(file_name)
                    for sheet in sheets:
                        dataset_key = f"{file_name} → {sheet}" if len(sheets) > 1 else file_name
                        if (dataset_key not in pipeline['used_in_append'] and 
                            dataset_key not in pipeline['used_in_merge']):  # Exclude used in merge too
                            if dataset_key in st.session_state.cleaned_datasets:
                                available[dataset_key] = st.session_state.cleaned_datasets[dataset_key]
                            else:
                                df = st.session_state.multi_file_loader.get_sheet_data(file_name, sheet)
                                if df is not None:
                                    available[dataset_key] = df
            
            # For custom columns: only active dataset or all available
            elif stage_name == 'custom_columns':
                if pipeline['active_dataset'] is not None and pipeline['active_dataset_name']:
                    available[pipeline['active_dataset_name']] = pipeline['active_dataset']
                else:
                    # Fall back to showing cleaned datasets
                    available = st.session_state.cleaned_datasets.copy()
            
            return available
        
        def get_data_source_badge(dataset_key, mark_merged=False, mark_appended=False):
            """Return a badge showing the data source of a dataset"""
            pipeline = st.session_state.data_cleaning_pipeline
            
            if mark_merged and dataset_key in pipeline['used_in_merge']:
                return "🔗 **Used in Merge** (excluded from next step)"
            if mark_appended and dataset_key in pipeline['used_in_append']:
                return "📊 **Used in Append** (excluded from next step)"
            
            if dataset_key in st.session_state.cleaned_datasets:
                if "🔗" in dataset_key:
                    return "🔗 **Merged Data**"
                elif "📊" in dataset_key:
                    return "📊 **Appended Data**"
                elif "➕" in dataset_key:
                    return "➕ **With Custom Columns**"
                else:
                    return "✅ **Cleaned Data**"
            else:
                return "📥 **Original Data**"
        
        def update_active_dataset(name, df, operation_desc):
            """Update the active dataset and log the operation"""
            pipeline = st.session_state.data_cleaning_pipeline
            pipeline['active_dataset_name'] = name
            pipeline['active_dataset'] = df.copy()
            pipeline['operation_history'].append(f"✓ {operation_desc}")
        
        # Create main tabs for different operations
        main_tabs = st.tabs(["📋 File Cleaning", "🔗 Merge", "📊 Append", "➕ Columns", "✅ Final Preview & Load"])
        
        # ==================== TAB 1: FILE-BY-FILE CLEANING ====================
        with main_tabs[0]:
            st.markdown("### 📋 Clean Data File by File")
            st.markdown("""
            > **Sequential Pipeline Rules:**
            > - ✅ Clean ONE file at a time
            > - ✅ Each cleaned file becomes available for merge/append
            > - ✅ Multiple files can be cleaned independently
            > - ✅ After cleaning, use **Merge** tab to combine files or **Append** to stack them
            """)
            
            # Create tabs for each file
            file_tabs = st.tabs([f"📄 {fname}" for fname in loaded_files])
            
            for file_idx, (file_name, file_tab) in enumerate(zip(loaded_files, file_tabs)):
                with file_tab:
                    sheets = st.session_state.multi_file_loader.get_file_sheets(file_name)
                    
                    # If multiple sheets, create sub-tabs
                    if len(sheets) > 1:
                        sheet_tabs = st.tabs([f"📋 {sheet}" for sheet in sheets])
                        sheet_list = list(zip(sheets, sheet_tabs))
                    else:
                        sheet_list = [(sheets[0], None)]
                    
                    for sheet_name, sheet_tab in sheet_list:
                        # Use sheet tab context if exists
                        container = sheet_tab if sheet_tab else st
                        
                        with container if sheet_tab else st.container():
                            dataset_key = f"{file_name} → {sheet_name}" if len(sheets) > 1 else file_name
                            
                            # Get the data (use cleaned version if exists, otherwise original)
                            if dataset_key in st.session_state.cleaned_datasets:
                                working_df = st.session_state.cleaned_datasets[dataset_key].copy()
                                st.success(f"✅ Using cleaned version of `{dataset_key}`")
                            else:
                                original_df = st.session_state.multi_file_loader.get_sheet_data(file_name, sheet_name)
                                if original_df is None:
                                    st.error(f"❌ Could not load data for {dataset_key}")
                                    continue
                                working_df = original_df.copy()
                            
                            # Initialize cleaner for this dataset
                            cleaner_key = f"cleaner_{dataset_key}"
                            if cleaner_key not in st.session_state:
                                st.session_state[cleaner_key] = DataCleaner(working_df)
                            
                            cleaner = st.session_state[cleaner_key]
                            
                            # Get current data and assess quality
                            current_df = cleaner.get_cleaned_df()
                            assessor = DataQualityAssessor(current_df)
                            assessment = assessor.assess_all()
                            
                            # Show Column Role Analysis
                            with st.expander("🔍 **Intelligent Column Analysis** (Click to expand)", expanded=False):
                                st.markdown("The system automatically detects column roles to apply appropriate cleaning rules:")
                                
                                role_summary = assessor.get_column_role_summary()
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if role_summary['primary_keys'] or role_summary['foreign_keys'] or role_summary['identifiers']:
                                        st.markdown("**🔑 Key/ID Columns** (excluded from numeric analysis):")
                                        all_keys = role_summary['primary_keys'] + role_summary['foreign_keys'] + role_summary['identifiers']
                                        for col in all_keys:
                                            st.write(f"  • `{col}`")
                                with col2:
                                    if role_summary['measures']:
                                        st.markdown("**📊 Measure Columns** (numeric analysis applies):")
                                        for col in role_summary['measures']:
                                            st.write(f"  • `{col}`")
                                with col3:
                                    if role_summary['categories']:
                                        st.markdown("**📁 Category Columns** (duplicates are normal):")
                                        for col in role_summary['categories']:
                                            st.write(f"  • `{col}`")
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Rows", len(current_df))
                            with col2:
                                st.metric("Columns", len(current_df.columns))
                            with col3:
                                total_issues = assessment['total_issues']
                                st.metric("Issues Found", total_issues)
                            with col4:
                                score = assessment['quality_score']
                                color = "🟢" if score >= 90 else ("🟡" if score >= 75 else "🔴")
                                st.metric("Quality Score", f"{color} {score:.0f}/100")
                            
                            st.markdown("---")
                            
                            # Quick Actions Row
                            st.markdown("### ⚡ Quick Actions")
                            quick_col1, quick_col2, quick_col3 = st.columns(3)
                            
                            with quick_col1:
                                # Remove full row duplicates
                                dup_count = current_df.duplicated().sum()
                                if dup_count > 0:
                                    if st.button(f"🗑️ Remove {dup_count} Duplicate Rows", key=f"quick_dup_{dataset_key}", use_container_width=True):
                                        cleaner.remove_duplicates()
                                        st.success(f"✅ Removed {dup_count} fully duplicate rows")
                                        st.rerun()
                                else:
                                    st.success("✅ No duplicate rows found")
                            
                            with quick_col2:
                                # Show key columns info
                                key_cols = cleaner.key_columns
                                if key_cols:
                                    st.info(f"🔑 {len(key_cols)} key column(s) detected (excluded from outlier analysis)")
                                else:
                                    st.info("ℹ️ No key columns detected")
                            
                            with quick_col3:
                                measure_cols = cleaner.measure_columns
                                st.info(f"📊 {len(measure_cols)} measure column(s) available for analysis")
                            
                            st.markdown("---")
                            
                            # Collect all issues
                            all_issues = []
                            for severity in ['critical', 'high', 'medium', 'low']:
                                for issue in assessment['by_severity'].get(severity, []):
                                    issue['severity'] = severity
                                    all_issues.append(issue)
                            
                            if not all_issues:
                                st.success("✨ **No issues found!** This dataset is clean.")
                            else:
                                st.markdown(f"### 🔍 Detected Issues ({len(all_issues)})")
                                
                                # Issue tracking in session state
                                fixed_key = f"fixed_issues_{dataset_key}"
                                skipped_key = f"skipped_issues_{dataset_key}"
                                if fixed_key not in st.session_state:
                                    st.session_state[fixed_key] = set()
                                if skipped_key not in st.session_state:
                                    st.session_state[skipped_key] = set()
                                
                                for issue_idx, issue in enumerate(all_issues):
                                    issue_id = f"{issue['type']}_{issue.get('column', 'general')}_{issue_idx}"
                                    
                                    # Skip if already fixed or skipped
                                    if issue_id in st.session_state[fixed_key]:
                                        continue
                                    if issue_id in st.session_state[skipped_key]:
                                        continue
                                    
                                    # Severity colors
                                    sev_colors = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                                    sev_color = sev_colors.get(issue['severity'], '⚪')
                                    
                                    column_name = issue.get('column', 'Multiple Columns')
                                    
                                    with st.expander(f"{sev_color} **{issue['type'].replace('_', ' ').title()}** in `{column_name}` ({issue['severity'].upper()})"):
                                        st.write(f"**📝 Description:** {issue['explanation']}")
                                        st.write(f"**💡 Recommendation:** {issue['recommendation']}")
                                        
                                        st.markdown("---")
                                        st.markdown("**🛠️ Choose Action:**")
                                        
                                        col_action, col_skip = st.columns([3, 1])
                                        
                                        with col_action:
                                            # Show fix options based on issue type
                                            if issue['type'] == 'missing_values':
                                                col_info = issue.get('column')
                                                if col_info:
                                                    missing_info = cleaner.get_missing_values_info(col_info)
                                                    missing_count = missing_info['total_count']
                                                    missing_pct = missing_info['percentage']
                                                    st.write(f"📊 **Missing:** {missing_count} values ({missing_pct:.1f}%)")
                                                    
                                                    # New feature: Selective fixing
                                                    fix_mode = st.radio(
                                                        "How many missing values to fix?",
                                                        options=["All", "Select specific rows"],
                                                        key=f"missing_mode_{dataset_key}_{issue_idx}",
                                                        horizontal=True
                                                    )
                                                    
                                                    rows_to_fix = missing_info['missing_indices']
                                                    
                                                    if fix_mode == "Select specific rows":
                                                        st.markdown("**Select which rows with missing values to fix:**")
                                                        # Show a preview of rows with missing values
                                                        missing_rows_df = current_df.loc[missing_info['missing_indices']].copy()
                                                        missing_rows_df['_row_id'] = missing_info['missing_indices']
                                                        
                                                        st.write(f"📋 Showing {min(10, len(missing_rows_df))} of {len(missing_rows_df)} rows with missing values:")
                                                        st.dataframe(missing_rows_df.head(10), use_container_width=True)
                                                        
                                                        # Let user select how many to fix
                                                        num_to_fix = st.slider(
                                                            f"Select how many of {missing_count} missing values to fix:",
                                                            min_value=0,
                                                            max_value=missing_count,
                                                            value=min(missing_count, 10),
                                                            key=f"missing_count_{dataset_key}_{issue_idx}"
                                                        )
                                                        
                                                        if num_to_fix > 0:
                                                            rows_to_fix = missing_info['missing_indices'][:num_to_fix]
                                                            st.info(f"✓ Will fix {len(rows_to_fix)} of {missing_count} missing values")
                                                        else:
                                                            st.warning("Select at least 1 value to fix")
                                                            rows_to_fix = []
                                                    
                                                    strategy = st.selectbox(
                                                        "Select fix strategy:",
                                                        ["mean", "median", "mode", "drop rows", "forward_fill", "backward_fill"],
                                                        key=f"strat_{dataset_key}_{issue_idx}",
                                                        help="mean/median for numeric, mode for categorical"
                                                    )
                                                    
                                                    # Show what fix will do
                                                    st.markdown("**📋 What this fix will do:**")
                                                    fix_label = f"{len(rows_to_fix)} missing values" if fix_mode == "Select specific rows" else f"{missing_count} missing values"
                                                    if strategy == "mean":
                                                        if col_info in current_df.select_dtypes(include=[np.number]).columns:
                                                            fill_val = current_df[col_info].mean()
                                                            st.info(f"→ Fill {fix_label} with mean: **{fill_val:.2f}**")
                                                        else:
                                                            st.warning("⚠️ Mean only works for numeric columns. Will use mode instead.")
                                                    elif strategy == "median":
                                                        if col_info in current_df.select_dtypes(include=[np.number]).columns:
                                                            fill_val = current_df[col_info].median()
                                                            st.info(f"→ Fill {fix_label} with median: **{fill_val:.2f}**")
                                                        else:
                                                            st.warning("⚠️ Median only works for numeric columns. Will use mode instead.")
                                                    elif strategy == "mode":
                                                        mode_val = current_df[col_info].mode()[0] if len(current_df[col_info].mode()) > 0 else "Unknown"
                                                        st.info(f"→ Fill {fix_label} with most frequent value: **{mode_val}**")
                                                    elif strategy == "drop rows":
                                                        st.info(f"→ Remove {fix_label} (drop rows containing missing values)")
                                                    elif strategy == "forward_fill":
                                                        st.info(f"→ Fill {fix_label} using the previous valid value")
                                                    elif strategy == "backward_fill":
                                                        st.info(f"→ Fill {fix_label} using the next valid value")
                                                    
                                                    if st.button("✅ Apply Fix", key=f"fix_{dataset_key}_{issue_idx}"):
                                                        if rows_to_fix:
                                                            if strategy == "drop rows":
                                                                cleaner.handle_missing_values_selective(col_info, rows_to_fix, strategy="drop")
                                                            else:
                                                                cleaner.handle_missing_values_selective(col_info, rows_to_fix, strategy=strategy.replace(" ", "_"))
                                                            st.session_state[fixed_key].add(issue_id)
                                                            st.success(f"✅ Fixed {len(rows_to_fix)} missing values in `{col_info}`")
                                                            st.rerun()
                                                        else:
                                                            st.warning("Please select at least one row to fix")
                                            
                                            elif issue['type'] == 'outliers':
                                                col_info = issue.get('column')
                                                if col_info and col_info in current_df.select_dtypes(include=[np.number]).columns:
                                                    # Check if it's a key column
                                                    if col_info in cleaner.key_columns:
                                                        st.warning(f"⚠️ Column `{col_info}` appears to be a key/identifier column. Outlier removal is not recommended for keys.")
                                                    else:
                                                        # Calculate outlier stats
                                                        Q1 = current_df[col_info].quantile(0.25)
                                                        Q3 = current_df[col_info].quantile(0.75)
                                                        IQR = Q3 - Q1
                                                        lower = Q1 - 1.5 * IQR
                                                        upper = Q3 + 1.5 * IQR
                                                        outlier_count = ((current_df[col_info] < lower) | (current_df[col_info] > upper)).sum()
                                                        
                                                        st.write(f"📊 **Outliers detected:** {outlier_count} values outside range [{lower:.2f}, {upper:.2f}]")
                                                        
                                                        method = st.selectbox(
                                                            "Select detection method:",
                                                            ["iqr", "zscore"],
                                                            key=f"outlier_meth_{dataset_key}_{issue_idx}"
                                                        )
                                                        
                                                        # Get outlier information
                                                        outlier_info = cleaner.get_outliers_in_column(col_info, method=method)
                                                        total_outliers = outlier_info['total_count']
                                                        
                                                        # New feature: Selective fixing
                                                        fix_mode = st.radio(
                                                            "How many outliers to remove?",
                                                            options=["All", "Select specific rows"],
                                                            key=f"outlier_mode_{dataset_key}_{issue_idx}",
                                                            horizontal=True
                                                        )
                                                        
                                                        rows_to_fix = outlier_info['outlier_indices']
                                                        
                                                        if fix_mode == "Select specific rows":
                                                            st.markdown("**Select which rows with outliers to remove:**")
                                                            # Show a preview of outlier rows
                                                            outlier_rows_df = current_df.loc[outlier_info['outlier_indices']].copy()
                                                            outlier_rows_df['_outlier_value'] = outlier_info['outlier_values']
                                                            outlier_rows_df['_row_id'] = outlier_info['outlier_indices']
                                                            
                                                            st.write(f"📋 Showing {min(10, len(outlier_rows_df))} of {total_outliers} rows with outliers:")
                                                            st.dataframe(outlier_rows_df[[col_info, '_outlier_value', '_row_id']].head(10), use_container_width=True)
                                                            
                                                            # Let user select how many to remove
                                                            num_to_remove = st.slider(
                                                                f"Select how many of {total_outliers} outliers to remove:",
                                                                min_value=0,
                                                                max_value=total_outliers,
                                                                value=min(total_outliers, 5),
                                                                key=f"outlier_count_{dataset_key}_{issue_idx}"
                                                            )
                                                            
                                                            if num_to_remove > 0:
                                                                rows_to_fix = outlier_info['outlier_indices'][:num_to_remove]
                                                                st.info(f"✓ Will remove {len(rows_to_fix)} of {total_outliers} outliers")
                                                            else:
                                                                st.warning("Select at least 1 outlier to remove")
                                                                rows_to_fix = []
                                                        
                                                        st.markdown("**📋 What this fix will do:**")
                                                        fix_label = f"{len(rows_to_fix)} rows" if fix_mode == "Select specific rows" else f"{total_outliers} rows"
                                                        if method == "iqr":
                                                            st.info(f"→ Remove {fix_label} with values outside IQR bounds [{lower:.2f}, {upper:.2f}]")
                                                        else:
                                                            st.info(f"→ Remove {fix_label} with Z-score ≥ 3")
                                                        
                                                        if st.button("✅ Remove Outliers", key=f"fix_{dataset_key}_{issue_idx}"):
                                                            if rows_to_fix:
                                                                cleaner.remove_outliers_selective(col_info, rows_to_fix, method=method)
                                                                st.session_state[fixed_key].add(issue_id)
                                                                st.success(f"✅ Removed {len(rows_to_fix)} outliers from `{col_info}`")
                                                                st.rerun()
                                                            else:
                                                                st.warning("Please select at least one outlier to remove")
                                            
                                            elif issue['type'] == 'duplicate_rows':
                                                dup_count = current_df.duplicated().sum()
                                                st.write(f"📊 **Fully duplicate rows:** {dup_count}")
                                                st.markdown("**ℹ️ Note:** This counts rows where ALL columns are identical.")
                                                
                                                st.markdown("**📋 What this fix will do:**")
                                                st.info(f"→ Remove {dup_count} fully duplicate rows, keeping first occurrence")
                                                
                                                if st.button("✅ Remove Duplicate Rows", key=f"fix_{dataset_key}_{issue_idx}"):
                                                    cleaner.remove_duplicates()
                                                    st.session_state[fixed_key].add(issue_id)
                                                    st.success("✅ Removed fully duplicate rows")
                                                    st.rerun()
                                            
                                            elif issue['type'] == 'high_cardinality':
                                                col_info = issue.get('column')
                                                if col_info:
                                                    unique_count = current_df[col_info].nunique()
                                                    st.write(f"📊 **Unique values:** {unique_count}")
                                                    st.markdown("**📋 Suggested action:**")
                                                    st.info("→ Consider grouping rare categories or using embedding. Manual review recommended.")
                                            
                                            else:
                                                st.info("ℹ️ This issue requires manual review or custom handling.")
                                        
                                        with col_skip:
                                            if st.button("⏭️ Skip", key=f"skip_{dataset_key}_{issue_idx}"):
                                                st.session_state[skipped_key].add(issue_id)
                                                st.rerun()
                                
                                # Show skipped issues count
                                skipped_count = len(st.session_state.get(skipped_key, set()))
                                fixed_count = len(st.session_state.get(fixed_key, set()))
                                if skipped_count > 0 or fixed_count > 0:
                                    st.markdown("---")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"✅ **Fixed issues:** {fixed_count}")
                                    with col2:
                                        st.write(f"⏭️ **Skipped issues:** {skipped_count}")
                            
                            # Cleaning Log and Actions
                            st.markdown("---")
                            col_log, col_actions = st.columns([1, 1])
                            
                            with col_log:
                                st.markdown("### 📝 Cleaning Log")
                                log = cleaner.get_cleaning_log()
                                if log:
                                    for entry in log:
                                        st.write(f"✓ {entry}")
                                else:
                                    st.write("No operations performed yet.")
                            
                            with col_actions:
                                st.markdown("### ⚙️ Actions")
                                
                                if st.button("💾 Save Cleaned Version", key=f"save_{dataset_key}", use_container_width=True):
                                    cleaned_df = cleaner.get_cleaned_df()
                                    st.session_state.cleaned_datasets[dataset_key] = cleaned_df
                                    st.success(f"✅ Saved cleaned version of `{dataset_key}`")
                            
                            if st.button("🔄 Reset to Original", key=f"reset_{dataset_key}", use_container_width=True):
                                cleaner.reset()
                                # Clear fixed/skipped tracking
                                st.session_state[fixed_key] = set()
                                st.session_state[skipped_key] = set()
                                if dataset_key in st.session_state.cleaned_datasets:
                                    del st.session_state.cleaned_datasets[dataset_key]
                                st.success("🔄 Reset to original data")
                                st.rerun()
                        
                        # Preview and Export
                        st.markdown("---")
                        st.markdown("### 👀 Data Preview")
                        st.dataframe(cleaner.get_cleaned_df().head(15), use_container_width=True)
                        
                        # Export options
                        st.markdown("### 📥 Export")
                        col_csv, col_excel = st.columns(2)
                        with col_csv:
                            csv = cleaner.get_cleaned_df().to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📄 Download CSV",
                                csv,
                                f"cleaned_{dataset_key.replace(' ', '_').replace('→', '_')}.csv",
                                "text/csv",
                                key=f'download-csv-{dataset_key}'
                            )
                        with col_excel:
                            # Excel export
                            import io
                            buffer = io.BytesIO()
                            cleaner.get_cleaned_df().to_excel(buffer, index=False, engine='openpyxl')
                            st.download_button(
                                "📊 Download Excel",
                                buffer.getvalue(),
                                f"cleaned_{dataset_key.replace(' ', '_').replace('→', '_')}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f'download-excel-{dataset_key}'
                            )
        
        # ==================== TAB 2: MERGE QUERIES ====================
        with main_tabs[1]:
            st.markdown("### 🔗 Merge Queries (VLOOKUP Style)")
            st.info("💡 Combine data from two tables based on matching columns. ⚠️ **After merge, original files used here will be excluded from next steps.**")
            
            # Get available datasets for merge (excludes already-merged files)
            all_datasets = get_available_datasets_for_stage('merge')
            
            if len(all_datasets) < 2:
                st.warning("⚠️ You need at least 2 available datasets to merge. Check earlier tabs or load more files.")
            else:
                dataset_names = list(all_datasets.keys())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📋 Main Table (Left)**")
                    left_table = st.selectbox("Select main table:", dataset_names, key="merge_left_table")
                    if left_table:
                        left_df = all_datasets[left_table]
                        st.caption(get_data_source_badge(left_table, mark_merged=True))
                        left_col = st.selectbox("Match on column:", left_df.columns.tolist(), key="merge_left_col")
                        st.write(f"📊 {len(left_df)} rows, {len(left_df.columns)} columns")
                
                with col2:
                    st.markdown("**🔍 Lookup Table (Right)**")
                    right_options = [d for d in dataset_names if d != left_table]
                    right_table = st.selectbox("Select lookup table:", right_options, key="merge_right_table")
                    if right_table:
                        right_df = all_datasets[right_table]
                        st.caption(get_data_source_badge(right_table, mark_merged=True))
                        right_col = st.selectbox("Match on column:", right_df.columns.tolist(), key="merge_right_col")
                        st.write(f"📊 {len(right_df)} rows, {len(right_df.columns)} columns")
                
                st.markdown("---")
                
                col_type, col_name = st.columns(2)
                with col_type:
                    merge_type = st.selectbox(
                        "Merge Type:",
                        ["left", "inner", "outer", "right"],
                        format_func=lambda x: {
                            "left": "Left (Keep all from main table - like VLOOKUP)",
                            "inner": "Inner (Only matching rows)",
                            "outer": "Outer (All rows from both tables)",
                            "right": "Right (Keep all from lookup table)"
                        }.get(x, x),
                        key="merge_type"
                    )
                
                with col_name:
                    merged_name = st.text_input("Name for merged result:", value="Merged_Data", key="merged_name")
                
                if st.button("👁️ Preview Merge", key="preview_merge"):
                    try:
                        merged_df = PowerQueryOperations.merge_queries(
                            left_df=all_datasets[left_table],
                            right_df=all_datasets[right_table],
                            left_on=left_col,
                            right_on=right_col,
                            how=merge_type
                        )
                        st.success(f"✅ Merge preview successful! Result: {len(merged_df)} rows, {len(merged_df.columns)} columns")
                        st.dataframe(merged_df.head(10), use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("### 📊 What Happens Next?")
                        st.warning(f"""
                        ⚠️ **After you confirm the merge:**
                        - ✅ Merged dataset: `{merged_name}` will be available for next operations
                        - ❌ These files will be EXCLUDED from append operations:
                          - {left_table}
                          - {right_table}
                        - 💡 You can only use the merged result going forward
                        """)
                        
                        if st.button("✅ Confirm & Save Merge", use_container_width=True, key="confirm_merge"):
                            # Save merged dataset
                            st.session_state.cleaned_datasets[merged_name] = merged_df.copy()
                            
                            # Mark original files as used in merge
                            pipeline['used_in_merge'].add(left_table)
                            pipeline['used_in_merge'].add(right_table)
                            
                            # Update active dataset
                            update_active_dataset(merged_name, merged_df, f"Merged {left_table} + {right_table}")
                            
                            st.success(f"✅ Merged and saved as `{merged_name}`!")
                            st.info("💡 Now available for Append or Custom Columns operations")
                            st.balloons()
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Merge failed: {str(e)}")
        
        # ==================== TAB 3: APPEND QUERIES ====================
        with main_tabs[2]:
            st.markdown("### 📊 Append Queries (Stack Tables)")
            st.info("💡 Stack multiple tables vertically. ⚠️ **After append, used files will be excluded from further operations.**")
            
            # Get available datasets for append
            all_datasets = get_available_datasets_for_stage('append')
            
            if len(all_datasets) < 2:
                st.warning("⚠️ You need at least 2 available datasets to append. Files used in Merge are excluded.")
            else:
                dataset_names = list(all_datasets.keys())
                
                st.markdown("**Select tables to append:**")
                selected_tables = st.multiselect(
                    "Choose 2 or more tables:",
                    dataset_names,
                    default=dataset_names[:2] if len(dataset_names) >= 2 else dataset_names,
                    key="append_tables"
                )
                
                if selected_tables:
                    st.markdown("**📊 Selected Tables:**")
                    cols = st.columns(min(len(selected_tables), 3))
                    for idx, table in enumerate(selected_tables):
                        with cols[idx % 3]:
                            st.write(f"**{table}**")
                            st.caption(get_data_source_badge(table, mark_appended=True))
                            st.write(f"{len(all_datasets[table])} rows")
                
                if len(selected_tables) >= 2:
                    # Show column comparison
                    st.markdown("**📋 Column Comparison:**")
                    all_cols = set()
                    for table in selected_tables:
                        all_cols.update(all_datasets[table].columns.tolist())
                    
                    col_status = []
                    for col in sorted(all_cols):
                        tables_with_col = [t for t in selected_tables if col in all_datasets[t].columns]
                        status = "✅ All tables" if len(tables_with_col) == len(selected_tables) else f"⚠️ {len(tables_with_col)}/{len(selected_tables)}"
                        col_status.append({"Column": col, "Status": status})
                    
                    st.dataframe(pd.DataFrame(col_status), use_container_width=True, hide_index=True)
                    
                    appended_name = st.text_input("Name for appended result:", value="Appended_Data", key="appended_name")
                    
                    if st.button("📊 Append Tables", key="execute_append"):
                        try:
                            dfs_to_append = [all_datasets[t] for t in selected_tables]
                            appended_df = PowerQueryOperations.append_queries(dfs_to_append)
                            
                            st.success(f"✅ Appended {len(selected_tables)} tables! Result: {len(appended_df)} rows, {len(appended_df.columns)} columns")
                            st.dataframe(appended_df.head(10), use_container_width=True)
                            
                            st.markdown("---")
                            st.markdown("### 📊 What Happens Next?")
                            st.warning(f"""
                            ⚠️ **After you confirm the append:**
                            - ✅ Appended dataset: `{appended_name}` will be available
                            - ❌ These files will be EXCLUDED from further operations:
                            {chr(10).join([f'  - {t}' for t in selected_tables])}
                            - 💡 You can only use the appended result going forward
                            """)
                            
                            if st.button("✅ Confirm & Save Append", use_container_width=True, key="confirm_append"):
                                st.session_state.cleaned_datasets[appended_name] = appended_df.copy()
                                
                                # Mark used files
                                for table in selected_tables:
                                    pipeline['used_in_append'].add(table)
                                
                                # Update active dataset
                                update_active_dataset(appended_name, appended_df, f"Appended {len(selected_tables)} tables")
                                
                                st.success(f"✅ Appended and saved as `{appended_name}`!")
                                st.balloons()
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Append failed: {str(e)}")
        

        # ==================== TAB 4: ADD CUSTOM COLUMNS ====================
        with main_tabs[3]:
            st.markdown("### ➕ Add Custom Columns")
            st.info("💡 Create calculated columns using expressions - like Custom Column in Power Query or formulas in Excel.")
            
            # Show notification if cleaned data is available
            if st.session_state.cleaned_datasets:
                with st.expander("✅ **Cleaned & Modified Datasets Available** (Click to see)", expanded=True):
                    cleaned_count = len(st.session_state.cleaned_datasets)
                    st.success(f"📊 **{cleaned_count} dataset(s)** have been cleaned, merged, or modified in previous tabs!")
                    st.write("These are ready to add columns to:")
                    for key in list(st.session_state.cleaned_datasets.keys())[:5]:
                        st.write(f"  ✅ {key}")
                    if cleaned_count > 5:
                        st.write(f"  ... and {cleaned_count - 5} more")
            
            # Get all available datasets
            all_datasets = {}
            for file_name in loaded_files:
                sheets = st.session_state.multi_file_loader.get_file_sheets(file_name)
                for sheet in sheets:
                    dataset_key = f"{file_name} → {sheet}" if len(sheets) > 1 else file_name
                    if dataset_key in st.session_state.cleaned_datasets:
                        all_datasets[dataset_key] = st.session_state.cleaned_datasets[dataset_key]
                    else:
                        df = st.session_state.multi_file_loader.get_sheet_data(file_name, sheet)
                        if df is not None:
                            all_datasets[dataset_key] = df
            
            if not all_datasets:
                st.warning("⚠️ No datasets loaded. Load files first.")
            else:
                dataset_names = list(all_datasets.keys())
                
                target_table = st.selectbox("Select table to add column to:", dataset_names, key="custom_col_table")
                
                if target_table:
                    target_df = all_datasets[target_table]
                    
                    # Show data source
                    st.caption(get_data_source_badge(target_table))
                    
                    st.markdown("**📋 Available Columns:**")
                    col_info = []
                    for col in target_df.columns:
                        col_info.append({
                            "Column": col,
                            "Type": str(target_df[col].dtype),
                            "Variable Name": re.sub(r'[^a-zA-Z0-9_]', '_', col)
                        })
                    st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    st.markdown("### 🔄 Data Type Converter")
                    st.info("💡 Convert columns to different data types - like changing text to numbers or dates.")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        convert_col = st.selectbox(
                            "Select column to convert:",
                            target_df.columns,
                            key="convert_column"
                        )
                    
                    # Determine suitable data types based on column content
                    def get_suitable_datatypes(col_series):
                        """Determine which data types are suitable for this column"""
                        suitable = []
                        current_dtype = str(col_series.dtype)
                        
                        # All columns can be converted to string
                        if "string" not in current_dtype:
                            suitable.append("string")
                        
                        # Try numeric conversions
                        try:
                            pd.to_numeric(col_series, errors='coerce')
                            if "int" not in current_dtype:
                                suitable.append("int64")
                            if "float" not in current_dtype:
                                suitable.append("float64")
                        except:
                            pass
                        
                        # Try datetime conversion
                        try:
                            pd.to_datetime(col_series, errors='coerce')
                            if "datetime" not in current_dtype:
                                suitable.append("datetime64[ns]")
                        except:
                            pass
                        
                        # Boolean conversion (works with any column)
                        if "bool" not in current_dtype:
                            suitable.append("bool")
                        
                        # Category conversion (works with any column)
                        if "category" not in current_dtype:
                            suitable.append("category")
                        
                        return suitable if suitable else ["string"]
                    
                    with col2:
                        suitable_types = get_suitable_datatypes(target_df[convert_col])
                        target_dtype = st.selectbox(
                            "Convert to:",
                            suitable_types,
                            key="target_dtype"
                        )
                    
                    with col3:
                        st.write("")
                        st.write("")
                        show_preview = st.checkbox("Preview", value=True, key="dtype_preview")
                    
                    if convert_col:
                        current_dtype = str(target_df[convert_col].dtype)
                        st.write(f"**Current type:** {current_dtype}")
                        st.write(f"**Target type:** {target_dtype}")
                        
                        # Show preview
                        if show_preview:
                            st.write("**Preview (first 5 rows):**")
                            try:
                                if target_dtype == "int64":
                                    preview_values = target_df[convert_col].astype(int)
                                elif target_dtype == "float64":
                                    preview_values = target_df[convert_col].astype(float)
                                elif target_dtype == "string":
                                    preview_values = target_df[convert_col].astype(str)
                                elif target_dtype == "datetime64[ns]":
                                    preview_values = pd.to_datetime(target_df[convert_col])
                                elif target_dtype == "bool":
                                    preview_values = target_df[convert_col].astype(bool)
                                elif target_dtype == "category":
                                    preview_values = target_df[convert_col].astype("category")
                                else:
                                    preview_values = target_df[convert_col]
                                
                                preview_df = pd.DataFrame({
                                    "Original": target_df[convert_col].head(),
                                    "Converted": preview_values.head()
                                })
                                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                                st.success(f"✅ Preview successful - conversion will work!")
                            except Exception as e:
                                st.error(f"⚠️ Conversion preview failed: {str(e)}")
                        
                        col_convert_btn, col_cancel_btn = st.columns(2)
                        
                        with col_convert_btn:
                            if st.button("✅ Apply Data Type Conversion", use_container_width=True, key="apply_dtype_convert"):
                                try:
                                    # Apply conversion
                                    if target_dtype == "int64":
                                        target_df[convert_col] = target_df[convert_col].astype(int)
                                    elif target_dtype == "float64":
                                        target_df[convert_col] = target_df[convert_col].astype(float)
                                    elif target_dtype == "string":
                                        target_df[convert_col] = target_df[convert_col].astype(str)
                                    elif target_dtype == "datetime64[ns]":
                                        target_df[convert_col] = pd.to_datetime(target_df[convert_col])
                                    elif target_dtype == "bool":
                                        target_df[convert_col] = target_df[convert_col].astype(bool)
                                    elif target_dtype == "category":
                                        target_df[convert_col] = target_df[convert_col].astype("category")
                                    
                                    # Update the cleaned datasets
                                    st.session_state.cleaned_datasets[target_table] = target_df.copy()
                                    
                                    # Show success with balloons
                                    st.balloons()
                                    st.success(f"✅ Column '{convert_col}' converted successfully!")
                                    st.info(f"📊 Changed from {current_dtype} to {target_dtype}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Conversion failed: {str(e)}")
                    
                    st.markdown("---")
                    st.markdown("### 🤖 AI Column Creator")
                    st.info("💡 Describe what you want in a new column and AI will create it for you!")
                    
                    col_desc = st.text_area(
                        "📝 Describe the new column (e.g., 'Multiply Price by Quantity', 'Extract year from date', 'Concatenate first and last name'):",
                        placeholder="e.g., Calculate total revenue by multiplying unit price with quantity sold",
                        key="column_description"
                    )
                    
                    if col_desc and st.button("🤖 Generate Column with AI", key="ai_generate_column"):
                        with st.spinner("🔄 AI is analyzing your description and creating the column..."):
                            try:
                                import ollama
                                
                                available_cols = ", ".join([f"`{col}` ({target_df[col].dtype})" for col in target_df.columns[:10]])
                                
                                ai_prompt = f"""You are a data transformation expert. Based on the user's description, create a Python expression for a new column.

Available columns in the table:
{available_cols}

User's request: {col_desc}

Provide ONLY the Python expression that can be used in pandas. Use standard pandas syntax.
For example:
- df['ColumnA'] * df['ColumnB']
- df['DateColumn'].dt.year
- df['FirstName'] + ' ' + df['LastName']

Do not include any explanation, just the expression."""
                                
                                response = ollama.chat(
                                    model=st.session_state.get('chatbot_model', 'qwen2.5:7b'),
                                    messages=[{"role": "user", "content": ai_prompt}]
                                )
                                
                                generated_expression = response['message']['content'].strip()
                                
                                # Clean up expression - remove assignment syntax if present
                                # Handle cases like "df['col_name'] = expression" -> "expression"
                                if '=' in generated_expression and not any(op in generated_expression for op in ['==', '!=', '<=', '>=']):
                                    # Extract the part after the = sign
                                    parts = generated_expression.split('=')
                                    if len(parts) == 2:
                                        generated_expression = parts[1].strip()
                                
                                ai_name_prompt = f"""Based on: '{col_desc}', provide a SHORT column name (max 15 chars, no spaces, underscores only).
Just the name, nothing else."""
                                
                                name_response = ollama.chat(
                                    model=st.session_state.get('chatbot_model', 'qwen2.5:7b'),
                                    messages=[{"role": "user", "content": ai_name_prompt}]
                                )
                                
                                generated_name = name_response['message']['content'].strip().replace(' ', '_').replace('-', '_').lower()
                                
                                st.session_state.ai_generated_column = {
                                    'name': generated_name,
                                    'expression': generated_expression,
                                    'description': col_desc,
                                    'table': target_table
                                }
                                
                            except Exception as e:
                                st.error(f"❌ AI generation failed: {str(e)}")
                                st.info("💡 Create manually below instead")
                    
                    if 'ai_generated_column' in st.session_state and st.session_state.ai_generated_column:
                        generated = st.session_state.ai_generated_column
                        
                        st.markdown("---")
                        st.markdown("### ✅ AI Generated Column")
                        
                        st.markdown(f"**Column Name:** `{generated['name']}`")
                        st.markdown(f"**Expression:** `{generated['expression']}`")
                        st.markdown(f"**Description:** {generated['description']}")
                        
                        try:
                            preview_df = target_df.copy()
                            df = preview_df
                            # Safely evaluate the expression with built-in functions available
                            safe_builtins = {
                                'int': int, 'float': float, 'str': str, 'bool': bool,
                                'len': len, 'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum
                            }
                            new_col_values = eval(generated['expression'], {"__builtins__": safe_builtins}, {'df': preview_df})
                            preview_df[generated['name']] = new_col_values
                            
                            st.markdown("**✅ Preview (first 5 rows):**")
                            preview_display = preview_df[[generated['name']]].head(5).copy()
                            preview_display['Data Type'] = str(preview_df[generated['name']].dtype)
                            st.dataframe(preview_display, use_container_width=True)
                            
                            st.success(f"✅ Expression is valid! Column will have {str(preview_df[generated['name']].dtype)} datatype")
                        except SyntaxError as e:
                            st.error(f"❌ Syntax error in expression: {str(e)}")
                            st.info("💡 The expression might have invalid Python syntax. Try rejecting and using manual creation.")
                        except Exception as e:
                            st.warning(f"⚠️ Could not evaluate expression: {str(e)}")
                            st.info("💡 The expression references columns or operations that may not exist. Check column names and try again.")
                        
                        col_yes, col_no = st.columns(2)
                        
                        with col_yes:
                            if st.button("✅ Add It!", key="confirm_ai_column", use_container_width=True):
                                try:
                                    result_df = PowerQueryOperations.add_custom_column(
                                        df=target_df,
                                        new_column_name=generated['name'],
                                        expression=generated['expression']
                                    )
                                    st.session_state.cleaned_datasets[target_table] = result_df
                                    
                                    # Show success with balloons
                                    st.balloons()
                                    st.success(f"✅ Successfully added column '{generated['name']}'!")
                                    st.info(f"📊 New column has {len(result_df):,} rows with datatype: {result_df[generated['name']].dtype}")
                                    
                                    del st.session_state.ai_generated_column
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to add column: {str(e)}")
                        
                        with col_no:
                            if st.button("❌ Reject", key="reject_ai_column", use_container_width=True):
                                st.info("You can create it manually below or try a different description")
                                del st.session_state.ai_generated_column
                                st.rerun()
                    
                    st.markdown("---")
                    st.markdown("### 📝 Manual Column Creation")
                    
                    new_col_name = st.text_input("New column name:", key="new_col_name")
                    expression = st.text_area(
                        "Expression (use column variable names from table above):",
                        placeholder="Example: Price * Quantity\nOr: df['Price'] * df['Quantity']",
                        key="custom_expression"
                    )
                    
                    if new_col_name and expression:
                        if st.button("➕ Add Column Manually", key="add_custom_col"):
                            try:
                                result_df = PowerQueryOperations.add_custom_column(
                                    df=target_df,
                                    new_column_name=new_col_name,
                                    expression=expression
                                )
                                
                                st.session_state.cleaned_datasets[target_table] = result_df
                                
                                # Show success with balloons
                                st.balloons()
                                st.success(f"✅ Added column '{new_col_name}' successfully!")
                                st.info(f"📊 New column has {len(result_df):,} rows with datatype: {result_df[new_col_name].dtype}")
                            except Exception as e:
                                st.error(f"❌ Failed to add column: {str(e)}")
                    
                    st.markdown("---")
                    st.markdown("### 🔗 Add Column from Another Table (VLOOKUP-style)")
                    
                    # Get other tables for lookup
                    other_tables = [t for t in dataset_names if t != target_table]
                    
                    if other_tables:
                        lookup_table = st.selectbox("Lookup from table:", other_tables, key="lookup_table")
                        
                        if lookup_table:
                            lookup_df = all_datasets[lookup_table]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                main_key = st.selectbox(f"Key in {target_table}:", target_df.columns.tolist(), key="main_lookup_key")
                            with col2:
                                lookup_key = st.selectbox(f"Key in {lookup_table}:", lookup_df.columns.tolist(), key="lookup_key")
                            
                            value_col = st.selectbox("Column to bring over:", lookup_df.columns.tolist(), key="value_col")
                            new_col_name_lookup = st.text_input("Name for new column:", value=value_col, key="new_col_name_lookup")
                            
                            if st.button("🔗 Add Lookup Column", key="add_lookup_col"):
                                try:
                                    result_df = PowerQueryOperations.add_column_from_lookup(
                                        main_df=target_df,
                                        lookup_df=lookup_df,
                                        main_key=main_key,
                                        lookup_key=lookup_key,
                                        value_column=value_col,
                                        new_column_name=new_col_name_lookup
                                    )
                                    
                                    st.success(f"✅ Added lookup column '{new_col_name_lookup}'!")
                                    st.dataframe(result_df[[main_key, new_col_name_lookup]].head(10), use_container_width=True)
                                    
                                    # Save
                                    st.session_state.cleaned_datasets[target_table] = result_df
                                    st.success(f"✅ Updated '{target_table}' with lookup column!")
                                except Exception as e:
                                    st.error(f"❌ Failed to add lookup column: {str(e)}")
        
        # ==================== TAB 5: FINAL PREVIEW & LOAD ====================
        with main_tabs[4]:
            st.markdown("### ✅ Final Preview & Load to Next Stage")
            st.markdown("""
            #### 🎯 Golden Rule: **The next stage receives ONLY what you see here**
            
            This preview shows the exact dataset that will be passed to all other pages.
            """)
            
            pipeline = st.session_state.data_cleaning_pipeline
            
            # Determine the final dataset
            final_dataset = None
            final_dataset_name = None
            
            # Priority: Active dataset (from merge/append) > Cleaned datasets > Original files
            if pipeline['active_dataset'] is not None:
                final_dataset = pipeline['active_dataset']
                final_dataset_name = pipeline['active_dataset_name']
            elif st.session_state.cleaned_datasets:
                # Use the most recent cleaned dataset
                final_dataset_name = list(st.session_state.cleaned_datasets.keys())[-1]
                final_dataset = st.session_state.cleaned_datasets[final_dataset_name]
            
            if final_dataset is None:
                st.info("💡 **No transformations have been applied yet.** Your data will be loaded as-is from the original files.")
                
                # Show available original files as options
                all_original = {}
                for file_name in loaded_files:
                    sheets = st.session_state.multi_file_loader.get_file_sheets(file_name)
                    for sheet in sheets:
                        dataset_key = f"{file_name} → {sheet}" if len(sheets) > 1 else file_name
                        df = st.session_state.multi_file_loader.get_sheet_data(file_name, sheet)
                        if df is not None:
                            all_original[dataset_key] = df
                
                if all_original:
                    st.markdown("**Available Original Files:**")
                    selected_final = st.selectbox(
                        "Select which file to load:",
                        list(all_original.keys()),
                        key="select_original_final"
                    )
                    final_dataset = all_original[selected_final]
                    final_dataset_name = selected_final
            else:
                st.success(f"✅ **Active Dataset:** `{final_dataset_name}`")
            
            if final_dataset is not None:
                # Show dataset summary
                st.markdown("---")
                st.markdown("### 📊 Final Dataset Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📈 Rows", f"{len(final_dataset):,}")
                with col2:
                    st.metric("📋 Columns", len(final_dataset.columns))
                with col3:
                    st.metric("💾 Memory", f"{final_dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                with col4:
                    missing_pct = (final_dataset.isna().sum().sum() / (len(final_dataset) * len(final_dataset.columns)) * 100)
                    st.metric("📌 Missing Data", f"{missing_pct:.1f}%")
                
                st.markdown("---")
                st.markdown("### 🔍 Data Preview (First 10 Rows)")
                st.dataframe(final_dataset.head(10), use_container_width=True, height=300)
                
                st.markdown("---")
                st.markdown("### 📋 Column Information")
                
                col_info = []
                for col in final_dataset.columns:
                    col_info.append({
                        "Column": col,
                        "Type": str(final_dataset[col].dtype),
                        "Non-Null": f"{final_dataset[col].notna().sum():,}",
                        "Null": final_dataset[col].isna().sum(),
                        "Unique": final_dataset[col].nunique()
                    })
                
                st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("### 📝 Pipeline Operations Applied")
                
                if pipeline['operation_history']:
                    for i, operation in enumerate(pipeline['operation_history'], 1):
                        st.write(f"{i}. {operation}")
                else:
                    st.write("No cleaning operations applied - using original data")
                
                st.markdown("---")
                st.markdown("### ⚠️ Excluded Files (Will NOT be passed to next stage)")
                
                excluded_files = pipeline['used_in_merge'].union(pipeline['used_in_append'])
                if excluded_files:
                    st.warning(f"""
                    The following files were used in merge/append operations and are **EXCLUDED** from the next stage:
                    {chr(10).join([f'- {f}' for f in sorted(excluded_files)])}
                    
                    ✅ **Only this will be loaded:** `{final_dataset_name}`
                    """)
                else:
                    st.info("✅ No files excluded - clean data only")
                
                st.markdown("---")
                st.markdown("### ✅ Confirm & Load to Next Stage")
                st.markdown(f"""
                **📤 Loading: `{final_dataset_name}`**
                
                This dataset contains:
                - **{len(final_dataset):,}** rows
                - **{len(final_dataset.columns)}** columns
                - **{final_dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB** of data
                
                🎯 **Golden Rule:** Next stage will receive EXACTLY this dataset, nothing more, nothing less.
                """)
                
                if st.button("🚀 Load & Proceed to Next Stage", key="load_to_next_stage", use_container_width=True):
                    # Save the final dataset as THE cleaned dataset
                    st.session_state.cleaned_datasets.clear()
                    st.session_state.cleaned_datasets['_final_dataset'] = final_dataset.copy()
                    st.session_state.pipeline_final_dataset = final_dataset.copy()
                    st.session_state.pipeline_final_dataset_name = final_dataset_name
                    
                    st.success("✅ Dataset loaded successfully!")
                    st.info(f"💡 Your cleaned data is now ready for analysis across all pages. All other pages will use this dataset.")
                    st.balloons()
                    
                    # Reset pipeline for next use
                    st.session_state.data_cleaning_pipeline = {
                        'active_dataset_name': None,
                        'active_dataset': None,
                        'operation_history': [],
                        'used_in_merge': set(),
                        'used_in_append': set(),
                        'final_dataset_preview': None,
                        'pipeline_state': 'file_cleaning'
                    }

elif page == "🎯 Business Goals":
    st.title("🎯 Business Strategic Goals")
    st.markdown("### Define your business problem to guide AI analysis")
    
    with st.form("business_goals_form"):
        problem = st.text_area("What is the core business problem you're trying to solve?", 
                              value=st.session_state.business_goals.get("problem", ""),
                              placeholder="e.g., Sales have declined by 15% in the last quarter...")
        
        objective = st.text_area("What is your primary objective for this analysis?", 
                                value=st.session_state.business_goals.get("objective", ""),
                                placeholder="e.g., Identify the main drivers of sales decline and suggest recovery strategies.")
        
        target = st.text_input("Who is the target audience for this report?", 
                              value=st.session_state.business_goals.get("target", ""),
                              placeholder="e.g., Regional Sales Managers and Executive Board.")
        
        submitted = st.form_submit_button("💾 Save Goals")
        if submitted:
            st.session_state.business_goals = {
                "problem": problem,
                "objective": objective,
                "target": target,
                "completed": True if problem and objective else False
            }
            st.success("✅ Business goals saved! AI analysis will now be more strategic.")
    
    # Full Data Analysis Section
    st.markdown("---")
    st.markdown("## 📊 Run Full Strategic Data Analysis")
    st.info("After saving your business goals, run a comprehensive analysis that addresses your business problem using all loaded data.")
    
    # Check if data is loaded and goals are defined
    has_data = st.session_state.multi_file_loader and len(st.session_state.multi_file_loader.get_loaded_files()) > 0
    has_goals = st.session_state.business_goals.get("completed", False)
    
    if not has_data:
        st.warning("⚠️ Please load data first from 'Multi-File Loading' page")
    elif not has_goals:
        st.warning("⚠️ Please define and save your business goals above first")
    else:
        # Show current goals summary
        with st.expander("📋 Current Business Goals", expanded=True):
            st.markdown(f"**Business Problem:** {st.session_state.business_goals.get('problem', 'Not defined')}")
            st.markdown(f"**Objective:** {st.session_state.business_goals.get('objective', 'Not defined')}")
            st.markdown(f"**Target Audience:** {st.session_state.business_goals.get('target', 'Not defined')}")
        
        # Run Analysis Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_analysis = st.button("🚀 Run Full Strategic Analysis", key="run_full_analysis", use_container_width=True)
        
        if run_analysis or st.session_state.get('strategic_analysis_result'):
            if run_analysis:
                with st.spinner("🔄 Running comprehensive strategic analysis... This may take a few minutes."):
                    # Gather all data summaries (uses cleaned versions if available)
                    all_datasets = get_all_datasets()
                    data_summary = ""
                    
                    for dataset_name, df in all_datasets.items():
                        # Create data summary for LLM
                        data_summary += f"\n\n=== Dataset: {dataset_name} ===\n"
                        data_summary += f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
                        data_summary += f"Columns: {', '.join(df.columns.tolist())}\n"
                        
                        # Column details
                        for col in df.columns:
                            dtype = str(df[col].dtype)
                            unique = df[col].nunique()
                            nulls = df[col].isna().sum()
                            sample = df[col].dropna().head(3).tolist()
                            data_summary += f"  - {col} ({dtype}): {unique} unique, {nulls} nulls, samples: {sample}\n"
                        
                        # Basic statistics for numeric columns
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            data_summary += "\nNumeric Statistics:\n"
                            stats = df[numeric_cols].describe().to_string()
                            data_summary += stats + "\n"
                    
                    # Get schema relationships if available
                    relationship_summary = ""
                    if st.session_state.get('schema_analyzer') and st.session_state.schema_analyzer.relationships:
                        relationship_summary = "\n\n=== Table Relationships ===\n"
                        for rel in st.session_state.schema_analyzer.relationships:
                            relationship_summary += f"- {rel['file1']}.{rel['column1']} → {rel['file2']}.{rel['column2']} ({rel['relationship_type']}, {rel['confidence']} confidence)\n"
                    
                    # Prepare the comprehensive analysis prompt
                    analysis_prompt = f"""You are a senior data analyst and business strategy expert.

=== BUSINESS CONTEXT ===
Business Problem: {st.session_state.business_goals.get('problem', 'Not specified')}
Primary Objective: {st.session_state.business_goals.get('objective', 'Not specified')}
Target Audience: {st.session_state.business_goals.get('target', 'Not specified')}

=== DATABASE SCHEMA AND DATA ===
{data_summary}
{relationship_summary}

=== YOUR TASK ===
Perform a full, end-to-end data analysis that directly addresses the user's business problem.

--------------------------------
STEP 1: Understand the Business Context
--------------------------------
- Clearly interpret the business problem in plain language.
- Translate the business objective into measurable analytical goals.
- Adjust the depth and style of analysis based on the target audience.

--------------------------------
STEP 2: Derive Key Analytical Questions
--------------------------------
- Convert the business problem into specific, data-driven questions.
- Focus only on questions that can be answered using the available data.
- Avoid assumptions or external knowledge.

--------------------------------
STEP 3: Automatically Select Analysis Approaches
--------------------------------
Based on the business problem and objective, choose the most relevant analysis techniques:
- Exploratory Data Analysis (EDA)
- Trend and time-based analysis
- Comparative analysis (period-over-period, group comparisons)
- Correlation and driver analysis
- Segmentation or cohort analysis
- Anomaly or outlier detection
- Forecasting (only if data supports it)

Explain why each selected approach is relevant to the business goal.

--------------------------------
STEP 4: Execute Full Data Analysis
--------------------------------
- Perform the selected analyses using only the uploaded data.
- Use table relationships and data distributions where applicable.
- Identify patterns, trends, drivers, and anomalies.
- Validate insights with supporting data evidence.

--------------------------------
STEP 5: Generate Insights and Findings
--------------------------------
- Summarize key findings clearly and concisely.
- Link each insight directly to the business problem.
- Include confidence levels or data limitations where relevant.

--------------------------------
STEP 6: Provide Actionable Business Recommendations
--------------------------------
- Propose practical, data-backed recommendations.
- Prioritize actions based on impact and feasibility.
- Tailor recommendations to the target audience.

--------------------------------
STEP 7: Highlight Risks, Gaps, and Next Steps
--------------------------------
- Identify missing data or analytical limitations.
- Suggest additional data or analyses that could improve accuracy.
- Clearly state what cannot be concluded from the current data.

--------------------------------
OUTPUT FORMAT:
--------------------------------
# Executive Summary
(Brief overview for the target audience)

# 1. Business Problem Interpretation
(Your understanding of the problem)

# 2. Key Analytical Questions
(List of data-driven questions)

# 3. Selected Analysis Approaches
(Methods chosen and why)

# 4. Data Analysis Summary
(Detailed findings from each analysis)

# 5. Key Insights
(Main discoveries with evidence)

# 6. Business Recommendations
(Actionable suggestions prioritized by impact)

# 7. Risks, Limitations, and Next Steps
(What's missing and what to do next)

Rules:
- Use ONLY the uploaded data and schema provided above.
- Do NOT assume unavailable fields or metrics.
- Do NOT use external benchmarks or knowledge.
- Ensure every insight is traceable to the data.
- Format the output in clean Markdown for readability."""

                    try:
                        import ollama
                        response = ollama.chat(
                            model="qwen2.5:7b",
                            messages=[{"role": "user", "content": analysis_prompt}]
                        )
                        
                        analysis_result = response['message']['content']
                        st.session_state.strategic_analysis_result = analysis_result
                        st.session_state.strategic_analysis_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                    except Exception as e:
                        st.error(f"❌ Error running analysis: {str(e)}")
                        st.session_state.strategic_analysis_result = None
            
            # Display results
            if st.session_state.get('strategic_analysis_result'):
                st.markdown("---")
                st.markdown("## 📈 Strategic Analysis Report")
                
                # Report metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"📅 Generated: {st.session_state.get('strategic_analysis_timestamp', 'N/A')}")
                with col2:
                    st.caption(f"🎯 Objective: {st.session_state.business_goals.get('objective', 'N/A')[:50]}...")
                
                # Display the analysis
                st.markdown(st.session_state.strategic_analysis_result)
                
                # Download options
                st.markdown("---")
                st.markdown("### 📥 Download Report")
                
                col1, col2, col3 = st.columns(3)
                
                # Prepare report content
                report_content = f"""# Strategic Business Analysis Report

Generated: {st.session_state.get('strategic_analysis_timestamp', 'N/A')}

## Business Context
- **Problem:** {st.session_state.business_goals.get('problem', 'Not specified')}
- **Objective:** {st.session_state.business_goals.get('objective', 'Not specified')}
- **Target Audience:** {st.session_state.business_goals.get('target', 'Not specified')}

---

{st.session_state.strategic_analysis_result}

---
*Report generated by Strategic AI Analyst*
"""
                
                with col1:
                    # Download as Markdown
                    st.download_button(
                        label="📄 Download as Markdown",
                        data=report_content,
                        file_name=f"strategic_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with col2:
                    # Download as HTML (can be printed to PDF)
                    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Strategic Business Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; line-height: 1.6; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4a90d9; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        h3 {{ color: #0f3460; }}
        .metadata {{ background: #f0f4f8; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .metadata p {{ margin: 5px 0; }}
        ul, ol {{ margin-left: 20px; }}
        li {{ margin-bottom: 8px; }}
        blockquote {{ border-left: 4px solid #4a90d9; padding-left: 15px; color: #555; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 8px; overflow-x: auto; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #777; font-size: 0.9em; }}
        @media print {{
            body {{ padding: 20px; }}
            h1 {{ page-break-before: avoid; }}
            h2 {{ page-break-after: avoid; }}
        }}
    </style>
</head>
<body>
    <h1>📊 Strategic Business Analysis Report</h1>
    
    <div class="metadata">
        <p><strong>📅 Generated:</strong> {st.session_state.get('strategic_analysis_timestamp', 'N/A')}</p>
        <p><strong>🎯 Business Problem:</strong> {st.session_state.business_goals.get('problem', 'Not specified')}</p>
        <p><strong>📌 Objective:</strong> {st.session_state.business_goals.get('objective', 'Not specified')}</p>
        <p><strong>👥 Target Audience:</strong> {st.session_state.business_goals.get('target', 'Not specified')}</p>
    </div>
    
    <div class="content">
        {st.session_state.strategic_analysis_result.replace(chr(10), '<br>').replace('# ', '<h2>').replace('## ', '<h3>').replace('### ', '<h4>')}
    </div>
    
    <div class="footer">
        <p>Report generated by Strategic AI Analyst | Print this page (Ctrl+P) to save as PDF</p>
    </div>
</body>
</html>"""
                    
                    st.download_button(
                        label="🌐 Download as HTML",
                        data=html_content,
                        file_name=f"strategic_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                
                with col3:
                    # Download as Text
                    st.download_button(
                        label="📝 Download as Text",
                        data=report_content,
                        file_name=f"strategic_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                st.info("💡 **Tip:** Download the HTML file and open it in your browser, then use **Print → Save as PDF** for a professional PDF report.")
                
                # Re-run analysis button
                if st.button("🔄 Re-run Analysis", key="rerun_analysis"):
                    st.session_state.strategic_analysis_result = None
                    st.rerun()

elif page == "🤖 Strategic AI Analyst":
    render_strategic_analysis_page()

elif page == "📊 KPIs Dashboard":
    render_kpi_dashboard()

elif page == "📊 Custom Dashboard":
    render_custom_dashboard()

elif page == "📄 Monthly Report":
    render_monthly_report_page()

elif page == "📈 Visualization":
    st.title("📈 Visualization Studio")
    st.markdown("### Create powerful visualizations across all your data")
    
    # Check if data is loaded
    if not st.session_state.multi_file_loader or len(st.session_state.multi_file_loader.get_loaded_files()) == 0:
        st.warning("⚠️ Please load data first from 'Multi-File Loading' page")
    else:
        # Get all datasets
        all_datasets = get_all_datasets()
        total_files = len(st.session_state.multi_file_loader.get_loaded_files())
        cleaned_datasets = st.session_state.get('cleaned_datasets', {})
        cleaned_count = len([k for k in all_datasets.keys() if k in cleaned_datasets])
        not_cleaned_count = len(all_datasets) - cleaned_count
        
        # Data Overview Section
        st.markdown("---")
        st.subheader("📁 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📂 Total Files", total_files)
        with col2:
            st.metric("📊 Total Datasets", len(all_datasets))
        with col3:
            st.metric("✨ Cleaned", cleaned_count)
        with col4:
            st.metric("📄 Not Cleaned", not_cleaned_count)
        
        # Calculate totals
        total_rows = sum(df.shape[0] for df in all_datasets.values())
        total_cols = sum(df.shape[1] for df in all_datasets.values())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📈 Total Rows", f"{total_rows:,}")
        with col2:
            st.metric("📋 Total Columns", total_cols)
        
        # Generate All Visualizations Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_all = st.button("🚀 Generate Smart Visualizations for All Data", key="viz_generate_all", use_container_width=True)
        
        # Initialize visualization cache if needed
        if 'viz_cache_key' not in st.session_state:
            st.session_state.viz_cache_key = None
        if 'viz_all_generated' not in st.session_state:
            st.session_state.viz_all_generated = False
        
        # Only regenerate if explicitly requested or cache is empty
        cache_key = f"{len(all_datasets)}_{sum(df.shape[0] for df in all_datasets.values())}"
        
        # ============= CRITICAL ANTI-REGENERATION LOGIC =============
        # GOLDEN RULE: Do NOT re-run analysis when:
        # 1. A chart is pinned on the visualization page
        # 2. A column or file is selected inside Custom Chart Builder
        # 3. Only dataset itself changes - then regenerate
        
        # Check if user explicitly clicked "Generate Smart Visualizations" button
        if generate_all:
            st.session_state.viz_all_generated = True
            st.session_state.viz_cache_key = cache_key
            st.session_state.viz_analysis_in_progress = True
        
        # CRITICAL FIX: Show cached results unless dataset changed or first time
        should_show_viz = (st.session_state.get('viz_all_generated') and 
                           st.session_state.get('viz_cache_key') == cache_key)
        
        # Check if this is the first time showing visualizations or if data changed
        should_run_llm_analysis = False
        if should_show_viz and st.session_state.viz_analysis_in_progress:
            # First time generating - run LLM analysis
            should_run_llm_analysis = True
            # Mark that we're no longer in the initial "generating" state
            # This ensures next reruns (from pinning/selecting) will use cached results
            st.session_state.viz_analysis_in_progress = False
        
        # Show visualizations only if generated and cache is still valid
        if should_show_viz:
            
            # ============= AUTO-GENERATED VISUALIZATIONS =============
            st.markdown("---")
            st.markdown("## 🎨 Auto-Generated Smart Visualizations")
            st.info("📌 **Important**: These visualizations use cached analysis. They won't regenerate when you pin charts or select columns. Click the button above to force a new analysis.")
            
            # Only run LLM analysis if this is the first generation
            if should_run_llm_analysis:
                with st.spinner("🤖 Analyzing data to identify business-critical visualizations..."):
                    # Prepare data summary for LLM
                    data_summary = []
                    for dataset_name, df in all_datasets.items():
                        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:5]
                        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
                        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()][:3]
                        
                        data_summary.append(f"""
**Dataset: {dataset_name}**
- Rows: {df.shape[0]:,}, Columns: {df.shape[1]}
- Categorical columns: {', '.join(cat_cols) if cat_cols else 'None'}
- Numeric columns: {', '.join(num_cols) if num_cols else 'None'}
- Date columns: {', '.join(date_cols) if date_cols else 'None'}
""")
                    
                    llm_prompt = f"""
You are a business intelligence expert. Analyze this data and recommend ONLY the most business-critical visualizations.

{chr(10).join(data_summary)}

Business Context:
- Problem: {st.session_state.business_goals.get('problem', 'General business analysis')}
- Objective: {st.session_state.business_goals.get('objective', 'Understand key metrics')}

IMPORTANT: Recommend ONLY 4-6 visualizations that will provide REAL business value. For each visualization, specify:

1. Chart type (bar/line/histogram/pie)
2. Dataset name (exactly as shown above)
3. Columns to use (X and Y if applicable)
4. Why this visualization is business-critical (one sentence)

Format your response EXACTLY like this:

VIZ_1:
Type: bar
Dataset: [exact dataset name]
X_Column: [column name]
Y_Column: [column name]
Reason: [Why this is critical for the business]

VIZ_2:
...

Do NOT recommend generic visualizations. Focus on insights that drive business decisions.
"""
                    
                    try:
                        import ollama
                        response = ollama.chat(model=st.session_state.get('ollama_model', 'qwen2.5:7b'), messages=[
                            {'role': 'user', 'content': llm_prompt}
                        ])
                        
                        llm_recommendations = response['message']['content']
                        st.session_state.viz_recommendations = llm_recommendations
                    except Exception as e:
                        st.error(f"❌ LLM analysis failed: {str(e)}")
                        llm_recommendations = st.session_state.get('viz_recommendations', None)
            else:
                # Use cached recommendations instead of running LLM again
                # This is what happens when pinning charts or selecting columns
                llm_recommendations = st.session_state.get('viz_recommendations', None)
            
            viz_tabs = st.tabs(["🎯 Business-Critical Visualizations", "📊 Top Performers", "📈 Trends & Time", "📉 Distributions"])
            
            # Tab 0: LLM-Recommended Business-Critical Visualizations
            with viz_tabs[0]:
                st.subheader("🎯 AI-Recommended Business-Critical Visualizations")
                st.info("💡 These visualizations have been specifically selected by AI as the most valuable for your business objectives")
                
                if llm_recommendations:
                    # Parse LLM recommendations
                    viz_pattern = r'VIZ_\d+:\s*Type:\s*(\w+)\s*Dataset:\s*([^\n]+)\s*X_Column:\s*([^\n]+)\s*(?:Y_Column:\s*([^\n]+)\s*)?Reason:\s*([^\n]+)'
                    matches = re.findall(viz_pattern, llm_recommendations, re.IGNORECASE)
                    
                    if matches:
                        for idx, match in enumerate(matches):
                            viz_type = match[0].lower().strip()
                            dataset_name = match[1].strip()
                            x_col = match[2].strip()
                            y_col = match[3].strip() if match[3] else None
                            reason = match[4].strip()
                            
                            # Find matching dataset
                            target_df = None
                            for ds_name, df in all_datasets.items():
                                if dataset_name.lower() in ds_name.lower():
                                    target_df = df
                                    break
                            
                            if target_df is not None:
                                try:
                                    st.markdown(f"**💼 Business Insight {idx+1}:** {reason}")
                                    
                                    if viz_type == 'bar' and y_col:
                                        # Check if columns exist
                                        if x_col in target_df.columns and y_col in target_df.columns:
                                            agg_df = target_df.groupby(x_col)[y_col].sum().nlargest(10).reset_index()
                                            fig = px.bar(agg_df, x=x_col, y=y_col,
                                                       title=f"{y_col} by {x_col}",
                                                       color=y_col, color_continuous_scale=get_palette_color_scale())
                                            st.plotly_chart(fig, use_container_width=True)
                                            render_pin_button(f"{y_col} by {x_col}", fig, f"viz_critical_{idx}")
                                    
                                    elif viz_type == 'line' and y_col:
                                        if x_col in target_df.columns and y_col in target_df.columns:
                                            plot_df = target_df.sort_values(x_col).head(200)
                                            fig = px.line(plot_df, x=x_col, y=y_col,
                                                        title=f"{y_col} Trend")
                                            st.plotly_chart(fig, use_container_width=True)
                                            render_pin_button(f"{y_col} Trend", fig, f"viz_critical_{idx}")
                                    
                                    elif viz_type == 'histogram' and x_col in target_df.columns:
                                        fig = px.histogram(target_df, x=x_col, nbins=30,
                                                         title=f"Distribution of {x_col}")
                                        st.plotly_chart(fig, use_container_width=True)
                                        render_pin_button(f"Dist: {x_col}", fig, f"viz_critical_{idx}")
                                    
                                    elif viz_type == 'pie' and x_col in target_df.columns:
                                        value_counts = target_df[x_col].value_counts().head(10)
                                        fig = px.pie(values=value_counts.values, names=value_counts.index,
                                                   title=f"Composition: {x_col}")
                                        st.plotly_chart(fig, use_container_width=True)
                                        render_pin_button(f"Pie: {x_col}", fig, f"viz_critical_{idx}")
                                    
                                    st.markdown("---")
                                except Exception as e:
                                    st.warning(f"⚠️ Could not generate visualization: {str(e)}")
                        
                        if not matches:
                            st.warning("⚠️ Could not parse LLM recommendations. Showing default visualizations.")
                    else:
                        st.info("💡 LLM provided general analysis. Review other tabs for available visualizations.")
                else:
                    st.warning("⚠️ LLM analysis unavailable. Please check your Ollama connection.")
                
                # Tab 1: Top Performers (Bar Charts)
                with viz_tabs[1]:
                    st.subheader("📊 Top Performers Analysis")
                    
                    chart_count = 0
                    for dataset_name, df in all_datasets.items():
                        cat_cols = df.select_dtypes(include=['object', 'category']).columns
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        
                        if len(cat_cols) > 0 and len(numeric_cols) > 0:
                            # Best categorical-numeric combinations
                            for cat_col in cat_cols[:2]:
                                for num_col in numeric_cols[:2]:
                                    if chart_count >= 6:
                                        break
                                    
                                    try:
                                        agg_df = df.groupby(cat_col)[num_col].sum().nlargest(10).reset_index()
                                        if len(agg_df) > 1:
                                            col1, col2 = st.columns(2) if chart_count % 2 == 0 else [st.columns(2)[1], st.columns(2)[0]]
                                            
                                            fig = px.bar(agg_df, x=cat_col, y=num_col,
                                                        title=f"Top {cat_col} by {num_col}",
                                                        color=num_col, color_continuous_scale=get_palette_color_scale())
                                            st.plotly_chart(fig, use_container_width=True)
                                            render_pin_button(f"Top {cat_col} by {num_col}", fig, f"viz_bar_{dataset_name}_{cat_col}_{num_col}")
                                            chart_count += 1
                                    except:
                                        pass
                
                # Tab 2: Trends & Time Series
                with viz_tabs[2]:
                    st.subheader("📈 Trends & Time Series")
                    
                    found_time = False
                    for dataset_name, df in all_datasets.items():
                        # Find date columns
                        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                        
                        # Try to detect date-like columns
                        if len(date_cols) == 0:
                            for col in df.columns:
                                if any(x in col.lower() for x in ['date', 'time', 'year', 'month', 'day']):
                                    try:
                                        df[col] = pd.to_datetime(df[col])
                                        date_cols.append(col)
                                    except:
                                        pass
                        
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        
                        if len(date_cols) > 0 and len(numeric_cols) > 0:
                            found_time = True
                            for date_col in date_cols[:1]:
                                for num_col in numeric_cols[:2]:
                                    try:
                                        plot_df = df.sort_values(date_col).head(200)
                                        fig = px.line(plot_df, x=date_col, y=num_col,
                                                     title=f"{num_col} Over Time - {dataset_name}")
                                        st.plotly_chart(fig, use_container_width=True)
                                        render_pin_button(f"Trend: {num_col}", fig, f"viz_trend_{dataset_name}_{num_col}")
                                    except:
                                        pass
                    
                    if not found_time:
                        st.info("💡 No date/time columns detected. Upload data with date columns for trend analysis.")
                
                # Tab 3: Distributions
                with viz_tabs[3]:
                    st.subheader("📉 Data Distributions")
                    
                    chart_count = 0
                    for dataset_name, df in all_datasets.items():
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        
                        for col in numeric_cols[:3]:
                            if chart_count >= 6:
                                break
                            
                            try:
                                fig = px.histogram(df, x=col, nbins=30,
                                                  title=f"Distribution: {col} ({dataset_name})",
                                                  color_discrete_sequence=['#4a90d9'])
                                st.plotly_chart(fig, use_container_width=True)
                                render_pin_button(f"Dist: {col}", fig, f"viz_dist_{dataset_name}_{col}")
                                chart_count += 1
                            except:
                                pass
                
                # Tab 4: Compositions (Pie Charts) - Removed
                # Compositions are now handled in Business-Critical tab only if LLM recommends them
        
        # ============= MANUAL CHART BUILDER =============
        st.markdown("---")
        st.markdown("## 🛠️ Custom Chart Builder")
        st.info("💡 Create charts using columns from different datasets (cross-file visualization)")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("⚙️ Configuration")
            
            chart_type = st.selectbox("Chart Type", ["Bar", "Scatter", "Line", "Box", "Histogram", "Pie", "Heatmap"], key="custom_chart_type")
            
            # Enable cross-dataset mode
            use_cross_dataset = st.checkbox("🔗 Use columns from different datasets", value=False, key="cross_dataset_mode")
            
            if not use_cross_dataset:
                # Single dataset mode (original)
                selected_dataset = st.selectbox("Select Dataset", list(all_datasets.keys()), key="viz_dataset")
                
                if selected_dataset:
                    df = all_datasets[selected_dataset]
                    
                    x_col = st.selectbox("X Axis", df.columns, key="viz_x")
                    
                    if chart_type not in ["Histogram", "Pie"]:
                        y_col = st.selectbox("Y Axis", df.columns, key="viz_y")
                    else:
                        y_col = None
                    
                    color_col = st.selectbox("Color By (Optional)", [None] + list(df.columns), key="viz_color")
                    
                    if chart_type == "Bar":
                        agg_func = st.selectbox("Aggregation", ["Sum", "Mean", "Count", "Max", "Min"], key="viz_agg")
                    
                    # CRITICAL: Track column/file selection to prevent unnecessary regeneration
                    # When user selects a column or file in this builder, DON'T regenerate auto-visualizations
                    st.session_state.custom_chart_builder_selection = {
                        "chart_type": chart_type,
                        "use_cross_dataset": use_cross_dataset,
                        "selected_dataset": selected_dataset,
                        "x_col": x_col,
                        "y_col": y_col if chart_type not in ["Histogram", "Pie"] else None
                    }
            else:
                # Cross-dataset mode - new functionality
                st.markdown("### 📊 Dataset 1 (X Axis)")
                dataset1 = st.selectbox("Dataset 1", list(all_datasets.keys()), key="cross_dataset1")
                df1 = all_datasets[dataset1]
                x_col = st.selectbox("X Column", df1.columns, key="cross_x")
                
                st.markdown("### 📈 Dataset 2 (Y Axis)")
                dataset2 = st.selectbox("Dataset 2", list(all_datasets.keys()), key="cross_dataset2")
                df2 = all_datasets[dataset2]
                
                if chart_type not in ["Histogram", "Pie"]:
                    y_col = st.selectbox("Y Column", df2.columns, key="cross_y")
                else:
                    y_col = None
                
                # Join column selection
                st.markdown("### 🔗 Join Settings")
                join_col1 = st.selectbox("Join Column from Dataset 1", df1.columns, key="join_col1")
                join_col2 = st.selectbox("Join Column from Dataset 2", df2.columns, key="join_col2")
                join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"], key="join_type")
                
                if chart_type == "Bar":
                    agg_func = st.selectbox("Aggregation", ["Sum", "Mean", "Count", "Max", "Min"], key="cross_agg")
                
                # CRITICAL: Track cross-dataset selection to prevent unnecessary regeneration
                # When user selects columns/files in cross-dataset mode, DON'T regenerate auto-visualizations
                st.session_state.custom_chart_builder_selection = {
                    "chart_type": chart_type,
                    "use_cross_dataset": use_cross_dataset,
                    "dataset1": dataset1,
                    "dataset2": dataset2,
                    "x_col": x_col,
                    "y_col": y_col if chart_type not in ["Histogram", "Pie"] else None
                }
                
                color_col = None
        
        with col2:
            st.subheader("📊 Preview")
            
            fig = None
            
            try:
                if not use_cross_dataset:
                    # Single dataset mode
                    if selected_dataset:
                        df = all_datasets[selected_dataset]
                        
                        if chart_type == "Scatter":
                            fig = px.scatter(df.head(1000), x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
                        elif chart_type == "Line":
                            fig = px.line(df.sort_values(x_col).head(500), x=x_col, y=y_col, color=color_col, title=f"{y_col} Trend")
                        elif chart_type == "Bar":
                            if y_col and df[y_col].dtype.kind in 'iuf':
                                agg_df = df.groupby(x_col)[y_col].agg(agg_func.lower()).nlargest(15).reset_index()
                                fig = px.bar(agg_df, x=x_col, y=y_col, color=color_col, title=f"{agg_func} of {y_col} by {x_col}")
                            else:
                                fig = px.bar(df[x_col].value_counts().head(15).reset_index(), x='index', y=x_col, title=f"Count of {x_col}")
                        elif chart_type == "Box":
                            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} Distribution by {x_col}")
                        elif chart_type == "Histogram":
                            fig = px.histogram(df, x=x_col, color=color_col, nbins=30, title=f"Distribution of {x_col}")
                        elif chart_type == "Pie":
                            counts = df[x_col].value_counts().head(10)
                            fig = px.pie(values=counts.values, names=counts.index, title=f"Composition of {x_col}")
                        elif chart_type == "Heatmap":
                            numeric_df = df.select_dtypes(include=[np.number])
                            if len(numeric_df.columns) > 1:
                                corr = numeric_df.corr()
                                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                else:
                    # Cross-dataset mode - merge datasets
                    merged_df = pd.merge(df1, df2, left_on=join_col1, right_on=join_col2, how=join_type, suffixes=('_ds1', '_ds2'))
                    
                    if len(merged_df) == 0:
                        st.warning("⚠️ No matching records found with current join settings. Try different join columns or join type.")
                    else:
                        st.success(f"✅ Successfully joined {len(merged_df):,} records from both datasets")
                        
                        # Get the actual column names after merge
                        x_col_merged = x_col if x_col in merged_df.columns else f"{x_col}_ds1"
                        y_col_merged = y_col if y_col in merged_df.columns else f"{y_col}_ds2" if y_col else None
                        
                        if chart_type == "Scatter" and y_col_merged:
                            fig = px.scatter(merged_df.head(1000), x=x_col_merged, y=y_col_merged, 
                                           title=f"Cross-Dataset: {x_col} ({dataset1}) vs {y_col} ({dataset2})")
                        elif chart_type == "Line" and y_col_merged:
                            fig = px.line(merged_df.sort_values(x_col_merged).head(500), x=x_col_merged, y=y_col_merged,
                                        title=f"Cross-Dataset Trend: {y_col} by {x_col}")
                        elif chart_type == "Bar" and y_col_merged:
                            if merged_df[y_col_merged].dtype.kind in 'iuf':
                                agg_df = merged_df.groupby(x_col_merged)[y_col_merged].agg(agg_func.lower()).nlargest(15).reset_index()
                                fig = px.bar(agg_df, x=x_col_merged, y=y_col_merged,
                                           title=f"Cross-Dataset: {agg_func} of {y_col} by {x_col}",
                                           color=y_col_merged, color_continuous_scale=get_palette_color_scale())
                            else:
                                fig = px.bar(merged_df[x_col_merged].value_counts().head(15).reset_index(), 
                                           x='index', y=x_col_merged, title=f"Count of {x_col}")
                        elif chart_type == "Box" and y_col_merged:
                            fig = px.box(merged_df, x=x_col_merged, y=y_col_merged, 
                                       title=f"Cross-Dataset Distribution: {y_col} by {x_col}")
                        elif chart_type == "Histogram":
                            fig = px.histogram(merged_df, x=x_col_merged, nbins=30, 
                                             title=f"Cross-Dataset Distribution: {x_col}")
                        elif chart_type == "Pie":
                            counts = merged_df[x_col_merged].value_counts().head(10)
                            fig = px.pie(values=counts.values, names=counts.index, 
                                       title=f"Cross-Dataset Composition: {x_col}")
                        elif chart_type == "Heatmap":
                            numeric_df = merged_df.select_dtypes(include=[np.number])
                            if len(numeric_df.columns) > 1:
                                corr = numeric_df.corr()
                                fig = px.imshow(corr, text_auto=True, title="Cross-Dataset Correlation Heatmap")
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    chart_title = f"Custom: {chart_type} - {x_col}" if not use_cross_dataset else f"Cross: {dataset1}.{x_col} × {dataset2}.{y_col if y_col else x_col}"
                    render_pin_button(chart_title, fig, f"custom_viz_{chart_type}_{x_col}")
                else:
                    st.info("Configure chart settings to see preview")
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
                st.info("💡 Try adjusting join columns or selecting compatible data types")

elif page == "💬 Enhanced Chatbot":
    st.title("💬 AI Data Analyst Chatbot")
    st.markdown("### 🤖 Ask questions about your data - Get answers with powerful visualizations")
    
    # Check if multi-file data is loaded
    all_datasets = get_all_datasets()
    
    if not all_datasets:
        st.warning("⚠️ Please load data first from 'Multi-File Loading' page")
    else:
        # Display data overview
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Datasets Loaded", len(all_datasets))
        with col2:
            total_rows = sum(df.shape[0] for df in all_datasets.values())
            st.metric("📈 Total Rows", f"{total_rows:,}")
        with col3:
            total_cols = sum(df.shape[1] for df in all_datasets.values())
            st.metric("📋 Total Columns", total_cols)
        with col4:
            cleaned_count = len([k for k in all_datasets.keys() if k in st.session_state.get('cleaned_datasets', {})])
            st.metric("✨ Cleaned Datasets", cleaned_count)
        
        # Configuration Section
        st.markdown("---")
        st.subheader("⚙️ Chatbot Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Model selection
            available_models = ["qwen2.5:7b", "llama3.2", "mistral", "phi3", "gemma2"]
            selected_model = st.selectbox(
                "🤖 Select LLM Model",
                available_models,
                index=0,
                key="chatbot_model",
                help="Choose the AI model for chatbot responses"
            )
        
        with col2:
            # Auto-visualization toggle
            auto_viz = st.checkbox(
                "📊 Auto-Generate Visualizations",
                value=True,
                key="chatbot_auto_viz",
                help="Automatically create charts when relevant to your question"
            )
        
        with col3:
            # Temperature control
            temperature = st.slider(
                "🌡️ Response Creativity",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                key="chatbot_temperature",
                help="0=Precise, 1=Creative"
            )
        
        # Initialize or reinitialize chatbot if model changed
        if (st.session_state.hybrid_chatbot is None or 
            st.session_state.get('chatbot_current_model') != selected_model):
            
            with st.spinner(f"🔄 Initializing AI Analyst with {selected_model}..."):
                from models.hybrid_chatbot import HybridChatbot
                st.session_state.hybrid_chatbot = HybridChatbot(all_datasets, model=selected_model)
                st.session_state.chatbot_current_model = selected_model
                st.success(f"✅ Chatbot ready with {len(all_datasets)} datasets!")
        
        chatbot = st.session_state.hybrid_chatbot
        
        # Control buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                chatbot.clear_history()
                st.success("✅ Chat history cleared!")
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset Chatbot", use_container_width=True):
                st.session_state.hybrid_chatbot = None
                st.session_state.chat_history = []
                st.success("✅ Chatbot reset!")
                st.rerun()
        
        # Example questions
        with st.expander("💡 Example Questions You Can Ask"):
            st.markdown("""
            **📊 Analysis Questions:**
            - What are the top 10 products by sales?
            - Show me the sales trend over time
            - Which region has the highest revenue?
            - Compare sales across different categories
            
            **📈 Visualization Requests:**
            - Plot the distribution of prices
            - Show correlation between quantity and amount
            - Create a chart showing monthly trends
            - Visualize the breakdown by territory
            
            **💼 Business Questions:**
            - What is the average order value?
            - How many unique customers do we have?
            - Which products are underperforming?
            - Calculate total revenue by quarter
            """)
        
        # Chat interface
        st.markdown("---")
        st.subheader("💬 Chat Conversation")
        
        # Display chat history
        for msg_idx, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display visualizations if present
                if "graphs" in message and message["graphs"]:
                    st.markdown(f"**📊 {len(message['graphs'])} Visualization(s) Generated:**")
                    for graph_idx, graph in enumerate(message["graphs"]):
                        with st.expander(f"📈 {graph['title']}", expanded=True):
                            st.plotly_chart(graph["figure"], use_container_width=True, key=f"chat_hist_{msg_idx}_{graph_idx}")
                            
                            # Add pin button for each graph
                            col1, col2 = st.columns([3, 1])
                            with col2:
                                render_pin_button(
                                    graph['title'], 
                                    graph['figure'], 
                                    f"chat_{msg_idx}_{graph_idx}"
                                )
        
        # Chat input
        if prompt := st.chat_input("💭 Ask me anything about your data..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing data and generating response..."):
                    try:
                        # Include business context if available
                        context = ""
                        if st.session_state.get('business_goals'):
                            goals = st.session_state.business_goals
                            if goals.get('problem') or goals.get('objective'):
                                context = f"\n\nBusiness Context:\n- Problem: {goals.get('problem', 'N/A')}\n- Objective: {goals.get('objective', 'N/A')}"
                        
                        full_prompt = prompt + context
                        
                        # Get response from chatbot
                        response = chatbot.chat(full_prompt, temperature=temperature)
                        
                        # Handle errors
                        if response.get("error"):
                            st.error(f"❌ Error: {response['error']}")
                            answer = response["answer"]
                        else:
                            answer = response["answer"]
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Prepare message for history
                        msg = {"role": "assistant", "content": answer}
                        
                        # Display and store visualizations
                        if auto_viz and response.get("graphs") and len(response["graphs"]) > 0:
                            graphs = response["graphs"]
                            msg["graphs"] = graphs
                            
                            st.markdown(f"**📊 {len(graphs)} Visualization(s) Generated:**")
                            
                            for graph_idx, graph in enumerate(graphs):
                                with st.expander(f"📈 {graph['title']}", expanded=True):
                                    st.plotly_chart(graph["figure"], use_container_width=True, key=f"chat_new_{len(st.session_state.chat_history)}_{graph_idx}")
                                    
                                    # Add pin button
                                    col1, col2 = st.columns([3, 1])
                                    with col2:
                                        render_pin_button(
                                            graph['title'],
                                            graph['figure'],
                                            f"chat_new_{len(st.session_state.chat_history)}_{graph_idx}"
                                        )
                            
                            st.success(f"✅ Generated {len(graphs)} powerful visualization(s) to support the answer!")
                        
                        # Add to history
                        st.session_state.chat_history.append(msg)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        error_msg = {"role": "assistant", "content": f"⚠️ I encountered an error: {str(e)}"}
                        st.session_state.chat_history.append(error_msg)
                    
                    st.rerun()

if __name__ == "__main__":
    pass
