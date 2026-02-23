"""
Convert structured data to natural language text for LLM training
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class DataToText:
    """Convert dataframe to natural language text"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def row_to_sentence(self, row: pd.Series) -> str:
        """Convert a single row to natural language sentence"""
        parts = []
        
        for col, value in row.items():
            if pd.isna(value):
                continue
            
            # Format value based on type
            if isinstance(value, (int, float)):
                if isinstance(value, float) and value == int(value):
                    value = int(value)
                formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            # Create natural language phrase
            parts.append(f"{col} is {formatted_value}")
        
        return ". ".join(parts) + "."
    
    def rows_to_sentences(self, limit: Optional[int] = None) -> List[str]:
        """Convert all rows to sentences"""
        sentences = []
        df_to_process = self.df.head(limit) if limit else self.df
        
        for idx, row in df_to_process.iterrows():
            sentence = self.row_to_sentence(row)
            if sentence.strip():
                sentences.append(sentence)
        
        return sentences
    
    def get_column_descriptions(self) -> str:
        """Generate descriptions of all columns"""
        descriptions = []
        descriptions.append("Dataset Overview:")
        descriptions.append(f"Total records: {len(self.df)}")
        descriptions.append(f"Total columns: {len(self.df.columns)}")
        descriptions.append("")
        
        descriptions.append("Column Descriptions:")
        for col in self.df.columns:
            col_type = "numeric" if col in self.numeric_cols else "categorical"
            unique_count = self.df[col].nunique()
            non_null = self.df[col].notna().sum()
            
            if col in self.numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                mean_val = self.df[col].mean()
                descriptions.append(
                    f"- {col} ({col_type}): ranges from {min_val} to {max_val}, "
                    f"average is {mean_val:.2f}, {non_null} non-null values"
                )
            else:
                top_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else "N/A"
                descriptions.append(
                    f"- {col} ({col_type}): {unique_count} unique values, "
                    f"most common is '{top_value}', {non_null} non-null values"
                )
        
        return "\n".join(descriptions)
    
    def get_summary_statistics_text(self) -> str:
        """Generate summary statistics as text"""
        text = []
        text.append("Summary Statistics:")
        text.append("")
        
        for col in self.numeric_cols:
            text.append(f"{col}:")
            text.append(f"  - Count: {self.df[col].count()}")
            text.append(f"  - Mean: {self.df[col].mean():.2f}")
            text.append(f"  - Median: {self.df[col].median():.2f}")
            text.append(f"  - Std Dev: {self.df[col].std():.2f}")
            text.append(f"  - Min: {self.df[col].min():.2f}")
            text.append(f"  - Max: {self.df[col].max():.2f}")
            text.append("")
        
        return "\n".join(text)
    
    def get_categorical_summary_text(self) -> str:
        """Generate categorical summary as text"""
        text = []
        text.append("Categorical Summary:")
        text.append("")
        
        for col in self.categorical_cols:
            text.append(f"{col}:")
            value_counts = self.df[col].value_counts()
            for value, count in value_counts.head(5).items():
                percentage = (count / len(self.df)) * 100
                text.append(f"  - {value}: {count} ({percentage:.1f}%)")
            text.append("")
        
        return "\n".join(text)
    
    def get_insights_text(self) -> str:
        """Generate key insights as text"""
        text = []
        text.append("Key Insights:")
        text.append("")
        
        # Missing values
        missing = self.df.isna().sum()
        if missing.sum() > 0:
            text.append("Missing Values:")
            for col, count in missing[missing > 0].items():
                pct = (count / len(self.df)) * 100
                text.append(f"  - {col}: {count} ({pct:.1f}%)")
            text.append("")
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            text.append(f"Duplicate rows: {duplicates}")
            text.append("")
        
        # Correlations
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            "col1": corr_matrix.columns[i],
                            "col2": corr_matrix.columns[j],
                            "corr": corr_matrix.iloc[i, j]
                        })
            
            if high_corr:
                text.append("High Correlations:")
                for item in high_corr:
                    text.append(f"  - {item['col1']} and {item['col2']}: {item['corr']:.2f}")
                text.append("")
        
        return "\n".join(text)
    
    def get_full_knowledge_base(self) -> str:
        """Generate complete knowledge base text"""
        sections = [
            self.get_column_descriptions(),
            "",
            self.get_summary_statistics_text(),
            "",
            self.get_categorical_summary_text(),
            "",
            self.get_insights_text(),
        ]
        
        return "\n".join(sections)
    
    def get_sample_records_text(self, n_samples: int = 10) -> str:
        """Get sample records as text"""
        text = []
        text.append(f"Sample Records (first {n_samples}):")
        text.append("")
        
        for idx, row in self.df.head(n_samples).iterrows():
            text.append(f"Record {idx + 1}:")
            text.append(self.row_to_sentence(row))
            text.append("")
        
        return "\n".join(text)
