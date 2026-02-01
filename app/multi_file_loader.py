"""
Multi-File Data Loader Module
Handles loading and combining multiple Excel files into a single dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import streamlit as st
from app.data_loader import ExcelLoader


class MultiFileLoader:
    """Load and combine multiple Excel files"""
    
    def __init__(self):
        self.files = {}  # {file_name: {sheets: {sheet_name: df}}}
        self.combined_df = None
        self.file_metadata = {}
        self.load_history = []
        
    def is_file_loaded(self, file_name: str) -> bool:
        """Check if a file is already loaded"""
        return file_name in self.files

    
    def add_file(self, file_path: str, file_name: str = None) -> bool:
        """
        Add an Excel file to the loader
        
        Args:
            file_path: Path to Excel file
            file_name: Optional custom name for the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_name is None:
                file_name = Path(file_path).stem
            
            # Validate file exists
            if not Path(file_path).exists():
                st.error(f"❌ File not found: {file_path}")
                return False
            
            # Validate file size
            file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
            if file_size > 1024:  # 1GB
                st.error(f"❌ File too large: {file_size:.1f}MB (max 1GB)")
                return False
            
            # Try to load file
            try:
                loader = ExcelLoader(file_path)
            except Exception as load_err:
                st.error(f"❌ Cannot open Excel file {file_name}: {str(load_err)}")
                st.caption("Possible issue: File may be corrupted or not a valid Excel format")
                return False
            
            # Get sheets
            try:
                sheets = loader.get_sheets()
            except Exception as sheet_err:
                st.error(f"❌ Cannot read sheets from {file_name}: {str(sheet_err)}")
                return False
            
            if not sheets:
                st.error(f"❌ No sheets found in {file_name}")
                return False
            
            # Store file info
            self.files[file_name] = {
                'path': file_path,
                'sheets': {},
                'loader': loader
            }
            
            # Load all sheets with error handling
            loaded_count = 0
            for sheet in sheets:
                try:
                    df = loader.load_sheet(sheet)
                    if df is not None and not df.empty:
                        self.files[file_name]['sheets'][sheet] = df
                        loaded_count += 1
                except Exception as sheet_load_err:
                    st.warning(f"⚠️ Cannot load sheet '{sheet}' from {file_name}: {str(sheet_load_err)}")
                    continue
            
            if loaded_count == 0:
                st.error(f"❌ No valid sheets could be loaded from {file_name}")
                return False
            
            # Store metadata
            self.file_metadata[file_name] = {
                'path': file_path,
                'sheets': sheets,
                'total_sheets': len(sheets),
                'loaded_sheets': len(self.files[file_name]['sheets'])
            }
            
            self.load_history.append(f"✅ Loaded {file_name} ({loaded_count}/{len(sheets)} sheets)")
            return True
        
        except Exception as e:
            st.error(f"❌ Unexpected error loading file {file_name}: {str(e)}")
            import traceback
            st.caption(f"Technical details: {traceback.format_exc()}")
            return False
            
    def load_file(self, uploaded_file) -> bool:
        """Load a file from Streamlit's file uploader"""
        try:
            import os
            # Ensure data directory exists
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            # Save file to disk
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            return self.add_file(file_path, uploaded_file.name)
        except Exception as e:
            st.error(f"❌ Error saving uploaded file: {str(e)}")
            return False
    
    def get_loaded_files(self) -> List[str]:
        """Get list of loaded files"""
        return list(self.files.keys())
    
    def get_file_sheets(self, file_name: str) -> List[str]:
        """Get sheets from a specific file"""
        if file_name in self.files:
            return list(self.files[file_name]['sheets'].keys())
        return []
    
    def get_sheet_data(self, file_name: str, sheet_name: str) -> Optional[pd.DataFrame]:
        """Get data from specific sheet"""
        if file_name in self.files and sheet_name in self.files[file_name]['sheets']:
            return self.files[file_name]['sheets'][sheet_name]
        return None
    
    def combine_sheets(self, file_sheet_pairs: List[Tuple[str, str]], 
                      how: str = 'concat', on: str = None) -> Optional[pd.DataFrame]:
        """
        Combine multiple sheets from different files
        
        Args:
            file_sheet_pairs: List of (file_name, sheet_name) tuples
            how: 'concat' (stack rows) or 'merge' (join columns)
            on: Column name to merge on (for merge operation)
            
        Returns:
            Combined DataFrame or None
        """
        try:
            dfs = []
            for file_name, sheet_name in file_sheet_pairs:
                df = self.get_sheet_data(file_name, sheet_name)
                if df is not None:
                    dfs.append(df)
            
            if not dfs:
                st.error("No valid sheets selected")
                return None
            
            if how == 'concat':
                # Stack rows (vertical concatenation)
                combined = pd.concat(dfs, ignore_index=True)
            elif how == 'merge':
                # Join columns (horizontal merge)
                if on is None:
                    st.error("Please specify 'on' column for merge operation")
                    return None
                combined = dfs[0]
                for df in dfs[1:]:
                    combined = combined.merge(df, on=on, how='outer')
            else:
                st.error(f"Unknown combine method: {how}")
                return None
            
            self.combined_df = combined
            self.load_history.append(f"✅ Combined {len(dfs)} sheets using '{how}' method")
            return combined
        
        except Exception as e:
            st.error(f"Error combining sheets: {str(e)}")
            return None
    
    def combine_all_files(self, how: str = 'concat', on: str = None) -> Optional[pd.DataFrame]:
        """
        Combine all loaded files (first sheet from each)
        
        Args:
            how: 'concat' or 'merge'
            on: Column to merge on (for merge)
            
        Returns:
            Combined DataFrame
        """
        try:
            file_sheet_pairs = []
            for file_name in self.files:
                sheets = self.get_file_sheets(file_name)
                if sheets:
                    file_sheet_pairs.append((file_name, sheets[0]))
            
            if not file_sheet_pairs:
                st.error("No files to combine")
                return None
            
            return self.combine_sheets(file_sheet_pairs, how=how, on=on)
        
        except Exception as e:
            st.error(f"Error combining all files: {str(e)}")
            return None
    
    def combine_by_sheet_name(self, sheet_name: str, how: str = 'concat', 
                             on: str = None) -> Optional[pd.DataFrame]:
        """
        Combine same-named sheets from all files
        
        Args:
            sheet_name: Name of sheet to combine from all files
            how: 'concat' or 'merge'
            on: Column to merge on (for merge)
            
        Returns:
            Combined DataFrame
        """
        try:
            file_sheet_pairs = []
            for file_name in self.files:
                if sheet_name in self.files[file_name]['sheets']:
                    file_sheet_pairs.append((file_name, sheet_name))
            
            if not file_sheet_pairs:
                st.error(f"Sheet '{sheet_name}' not found in any file")
                return None
            
            return self.combine_sheets(file_sheet_pairs, how=how, on=on)
        
        except Exception as e:
            st.error(f"Error combining sheets: {str(e)}")
            return None
    
    def get_combined_df(self) -> Optional[pd.DataFrame]:
        """Get the combined DataFrame"""
        return self.combined_df
    
    def get_combined_shape(self) -> Tuple[int, int]:
        """Get shape of combined DataFrame"""
        if self.combined_df is not None:
            return self.combined_df.shape
        return (0, 0)
    
    def get_combined_columns(self) -> List[str]:
        """Get columns of combined DataFrame"""
        if self.combined_df is not None:
            return self.combined_df.columns.tolist()
        return []
    
    def get_combined_preview(self, n_rows: int = 5) -> Optional[pd.DataFrame]:
        """Get preview of combined DataFrame"""
        if self.combined_df is not None:
            return self.combined_df.head(n_rows)
        return None
    
    def get_file_summary(self) -> Dict:
        """Get summary of all loaded files"""
        summary = {
            'total_files': len(self.files),
            'files': {}
        }
        
        for file_name, file_info in self.files.items():
            sheets_info = {}
            for sheet_name, df in file_info['sheets'].items():
                sheets_info[sheet_name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
                }
            
            summary['files'][file_name] = {
                'sheets': sheets_info,
                'total_sheets': len(sheets_info)
            }
        
        return summary
    
    def get_load_history(self) -> List[str]:
        """Get loading history"""
        return self.load_history
    
    def clear(self):
        """Clear all loaded data"""
        self.files = {}
        self.combined_df = None
        self.file_metadata = {}
        self.load_history = []
    
    def export_combined(self, output_path: str, format: str = 'excel') -> bool:
        """
        Export combined DataFrame
        
        Args:
            output_path: Path to save file
            format: 'excel', 'csv', or 'parquet'
            
        Returns:
            True if successful
        """
        try:
            if self.combined_df is None:
                st.error("No combined data to export")
                return False
            
            if format == 'excel':
                self.combined_df.to_excel(output_path, index=False)
            elif format == 'csv':
                self.combined_df.to_csv(output_path, index=False)
            elif format == 'parquet':
                self.combined_df.to_parquet(output_path, index=False)
            else:
                st.error(f"Unknown format: {format}")
                return False
            
            st.success(f"✅ Exported to {output_path}")
            return True
        
        except Exception as e:
            st.error(f"Error exporting: {str(e)}")
            return False
    
    def get_column_mapping(self) -> Dict[str, List[str]]:
        """
        Get mapping of columns across all files
        Shows which files have which columns
        """
        column_mapping = {}
        
        for file_name in self.files:
            for sheet_name, df in self.files[file_name]['sheets'].items():
                for col in df.columns:
                    if col not in column_mapping:
                        column_mapping[col] = []
                    column_mapping[col].append(f"{file_name}/{sheet_name}")
        
        return column_mapping
    
    def validate_merge_column(self, column_name: str) -> bool:
        """Check if column exists in all files for merging"""
        for file_name in self.files:
            for sheet_name, df in self.files[file_name]['sheets'].items():
                if column_name not in df.columns:
                    return False
        return True
    
    def get_common_columns(self) -> List[str]:
        """Get columns that exist in all files"""
        if not self.files:
            return []
        
        all_columns = None
        for file_name in self.files:
            for sheet_name, df in self.files[file_name]['sheets'].items():
                file_columns = set(df.columns)
                if all_columns is None:
                    all_columns = file_columns
                else:
                    all_columns = all_columns.intersection(file_columns)
        
        return list(all_columns) if all_columns else []
    
    def get_unique_columns(self) -> Dict[str, List[str]]:
        """Get columns unique to each file"""
        common = set(self.get_common_columns())
        unique_cols = {}
        
        for file_name in self.files:
            unique_cols[file_name] = []
            for sheet_name, df in self.files[file_name]['sheets'].items():
                file_cols = set(df.columns)
                unique = file_cols - common
                unique_cols[file_name].extend(list(unique))
        
        return unique_cols
