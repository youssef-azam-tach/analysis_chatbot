"""
Schema Analyzer Module
Automatically understands dataset schema and detects relationships between files using LLM
Enhanced with data-based relationship detection similar to ERD
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaAnalyzer:
    """
    Analyze multiple datasets and understand their structure and relationships
    Uses LLM to intelligently detect connections between files based on actual data values
    """
    
    def __init__(self, model: str = "qwen2.5:7b"):
        """
        Initialize schema analyzer
        
        Args:
            model: LLM model to use for analysis
        """
        self.model = model
        self.datasets = {}
        self.schemas = {}
        self.relationships = []
        self.data_overlaps = {}  # Store computed data overlaps between columns
    
    def add_dataset(self, name: str, df: pd.DataFrame):
        """Add a dataset for analysis"""
        self.datasets[name] = df
        self.schemas[name] = self._extract_schema(name, df)
        logger.info(f"Added dataset: {name} with {len(df)} rows and {len(df.columns)} columns")
    
    def _extract_schema(self, name: str, df: pd.DataFrame) -> Dict:
        """Extract detailed schema information from dataframe"""
        schema = {
            "name": name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_details": []
        }
        
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(5).tolist()
            }
            
            # Determine column type
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["type"] = "numeric"
                col_info["min"] = float(df[col].min()) if df[col].notna().any() else None
                col_info["max"] = float(df[col].max()) if df[col].notna().any() else None
                col_info["mean"] = float(df[col].mean()) if df[col].notna().any() else None
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info["type"] = "datetime"
            else:
                col_info["type"] = "categorical"
                col_info["unique_ratio"] = df[col].nunique() / len(df) if len(df) > 0 else 0
            
            schema["column_details"].append(col_info)
        
        return schema
    
    def _compute_data_overlaps(self) -> Dict:
        """
        Compute actual data value overlaps between all column pairs across datasets.
        This is the core of data-based relationship detection.
        """
        overlaps = {}
        dataset_names = list(self.datasets.keys())
        
        for i, name1 in enumerate(dataset_names):
            df1 = self.datasets[name1]
            for name2 in dataset_names[i+1:]:
                df2 = self.datasets[name2]
                
                pair_key = f"{name1}|{name2}"
                overlaps[pair_key] = []
                
                # Compare each column in df1 with each column in df2
                for col1 in df1.columns:
                    set1 = set(df1[col1].dropna().astype(str).unique())
                    if len(set1) == 0:
                        continue
                        
                    for col2 in df2.columns:
                        set2 = set(df2[col2].dropna().astype(str).unique())
                        if len(set2) == 0:
                            continue
                        
                        # Calculate overlap
                        intersection = set1 & set2
                        if len(intersection) > 0:
                            # Calculate match percentages
                            match_pct_1 = len(intersection) / len(set1) * 100
                            match_pct_2 = len(intersection) / len(set2) * 100
                            
                            # Determine cardinality
                            unique_1 = df1[col1].nunique()
                            unique_2 = df2[col2].nunique()
                            total_1 = len(df1)
                            total_2 = len(df2)
                            
                            # Infer relationship type based on cardinality
                            if unique_1 == total_1 and unique_2 < total_2:
                                rel_type = "one-to-many"
                            elif unique_2 == total_2 and unique_1 < total_1:
                                rel_type = "many-to-one"
                            elif unique_1 == total_1 and unique_2 == total_2:
                                rel_type = "one-to-one"
                            else:
                                rel_type = "many-to-many"
                            
                            overlap_info = {
                                "table1": name1,
                                "table2": name2,
                                "column1": col1,
                                "column2": col2,
                                "overlap_count": len(intersection),
                                "unique_in_table1": len(set1),
                                "unique_in_table2": len(set2),
                                "match_pct_table1": round(match_pct_1, 2),
                                "match_pct_table2": round(match_pct_2, 2),
                                "sample_matching_values": list(intersection)[:5],
                                "inferred_relationship_type": rel_type,
                                "cardinality_1": f"{unique_1}/{total_1}",
                                "cardinality_2": f"{unique_2}/{total_2}"
                            }
                            overlaps[pair_key].append(overlap_info)
        
        self.data_overlaps = overlaps
        return overlaps
    
    def analyze_relationships(self) -> List[Dict]:
        """
        Use LLM to analyze relationships between datasets based on actual data values.
        Returns list of detected relationships similar to ERD.
        """
        if len(self.datasets) < 2:
            logger.warning("Need at least 2 datasets to analyze relationships")
            self.relationships = []
            return []
        
        # First compute actual data overlaps
        overlaps = self._compute_data_overlaps()
        
        # Prepare schema and data overlap description for LLM
        schema_description = self._prepare_schema_description()
        data_overlap_description = self._prepare_data_overlap_description(overlaps)
        
        # Enhanced ERD-focused prompt
        prompt = f"""You are a database schema analysis expert specializing in ERD (Entity-Relationship Diagram) design.

Your task is to analyze the provided database tables, columns, sample data, and ACTUAL DATA OVERLAPS to detect relationships.

=== DATASET SCHEMAS ===
{schema_description}

=== ACTUAL DATA VALUE OVERLAPS ===
The following shows actual matching values found between columns across different tables:
{data_overlap_description}

=== YOUR OBJECTIVES ===

1. **Detect relationships between tables similar to an ERD diagram**
   - Identify Primary Key columns (unique identifiers)
   - Identify Foreign Key columns (references to other tables)
   - Map the entity relationships

2. **Infer relationships based on ACTUAL DATA VALUES, not column names alone**
   - Use the data overlap analysis provided above
   - Look at match percentages and cardinality
   - Detect surrogate relationships even if column names differ completely

3. **Identify relationship types accurately:**
   - **One-to-One (1:1)**: Each record in Table A relates to exactly one record in Table B
   - **One-to-Many (1:N)**: One record in Table A can relate to many records in Table B
   - **Many-to-Many (M:N)**: Many records in Table A can relate to many records in Table B

4. **Validate and assess each relationship:**
   - Calculate confidence score based on data evidence
   - Flag relationships that are:
     * Not supported by data (low overlap)
     * Redundant (duplicate information)
     * Weak or statistically insignificant (< 10% match)

=== DATA-BASED RELATIONSHIP DETECTION RULES ===
- Match percentage > 80%: HIGH confidence
- Match percentage 50-80%: MEDIUM confidence  
- Match percentage 20-50%: LOW confidence
- Match percentage < 20%: Consider for deletion/ignore

- If column has unique values = total rows → likely Primary Key
- If column values match another table's PK → likely Foreign Key

=== OUTPUT FORMAT (JSON ONLY) ===
{{
  "relationships": [
    {{
      "file1": "exact_table_name",
      "file2": "exact_table_name",
      "column1": "exact_column_name (include PK/FK indicator)",
      "column2": "exact_column_name (include PK/FK indicator)",
      "relationship_type": "one-to-many|many-to-one|one-to-one|many-to-many",
      "confidence": "high|medium|low",
      "match_percentage": 85.5,
      "overlap_count": 150,
      "reasoning": "detailed explanation with data evidence",
      "is_primary_key": true/false,
      "is_foreign_key": true/false,
      "recommendation": "keep|review|delete"
    }}
  ],
  "suggested_deletions": [
    {{
      "file1": "table_name",
      "file2": "table_name", 
      "column1": "column_name",
      "column2": "column_name",
      "reason": "why this relationship should be deleted"
    }}
  ],
  "erd_summary": "comprehensive ERD-style summary of the data structure",
  "primary_keys_detected": ["table.column", ...],
  "foreign_keys_detected": ["table.column -> referenced_table.column", ...]
}}

Return ONLY valid JSON. Be thorough and analyze ALL potential relationships based on the actual data overlaps provided."""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse LLM response
            import json
            import re
            
            # Extract JSON from response
            content = response['message']['content']
            
            # Try to find JSON block - look for { ... } pattern
            # Use a more robust approach to handle malformed JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            
            if json_match:
                json_str = json_match.group()
                
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as json_err:
                    # Try to clean up common JSON issues
                    logger.warning(f"Initial JSON parse failed: {str(json_err)}")
                    
                    # Try removing trailing commas before ] or }
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    
                    try:
                        result = json.loads(json_str)
                        logger.info("Successfully parsed JSON after cleanup")
                    except json.JSONDecodeError as json_err2:
                        logger.error(f"Could not parse JSON even after cleanup: {str(json_err2)}")
                        logger.debug(f"JSON string: {json_str[:500]}...")
                        # Return empty relationships
                        return []
                
                self.relationships = result.get("relationships", [])
                self.suggested_deletions = result.get("suggested_deletions", [])
                self.erd_summary = result.get("erd_summary", "")
                self.primary_keys = result.get("primary_keys_detected", [])
                self.foreign_keys = result.get("foreign_keys_detected", [])
                
                logger.info(f"Detected {len(self.relationships)} relationships")
                return self.relationships
            else:
                logger.error("Could not find JSON block in LLM response")
                logger.debug(f"Response content: {content[:500]}...")
                return []
                
        except Exception as e:
            logger.error(f"Error analyzing relationships: {str(e)}")
            logger.debug(f"Full error: {repr(e)}")
            
            # Fallback: Generate relationships from data overlaps
            logger.info("Falling back to basic relationship detection from data overlaps")
            self.relationships = self._generate_relationships_from_overlaps(overlaps)
            return self.relationships
    
    def _generate_relationships_from_overlaps(self, overlaps: Dict) -> List[Dict]:
        """
        Generate relationships directly from data overlaps when LLM analysis fails.
        This is a fallback mechanism to ensure the system still works.
        """
        relationships = []
        
        for pair_key, overlap_list in overlaps.items():
            if not overlap_list:
                continue
            
            file1, file2 = pair_key.split("|")
            
            # Sort overlaps by match percentage
            sorted_overlaps = sorted(overlap_list, key=lambda x: x['match_pct_table1'], reverse=True)
            
            # Take top 3 overlaps per table pair
            for overlap in sorted_overlaps[:3]:
                match_pct = (overlap['match_pct_table1'] + overlap['match_pct_table2']) / 2
                
                # Determine confidence based on match percentage
                if match_pct >= 80:
                    confidence = "high"
                elif match_pct >= 50:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                # Infer relationship type from cardinality
                card1 = overlap.get('cardinality_1', '').split('/')
                card2 = overlap.get('cardinality_2', '').split('/')
                
                if len(card1) == 2 and len(card2) == 2:
                    try:
                        unique1, total1 = int(card1[0]), int(card1[1])
                        unique2, total2 = int(card2[0]), int(card2[1])
                        
                        if unique1 == total1 and unique2 == total2:
                            rel_type = "one-to-one"
                        elif unique1 == total1:
                            rel_type = "one-to-many"
                        elif unique2 == total2:
                            rel_type = "many-to-one"
                        else:
                            rel_type = "many-to-many"
                    except (ValueError, IndexError):
                        rel_type = "many-to-many"
                else:
                    rel_type = "many-to-many"
                
                relationship = {
                    "file1": file1,
                    "file2": file2,
                    "column1": overlap['column1'],
                    "column2": overlap['column2'],
                    "relationship_type": rel_type,
                    "confidence": confidence,
                    "match_percentage": round(match_pct, 2),
                    "overlap_count": overlap['overlap_count'],
                    "reasoning": f"Data overlap detected: {overlap['overlap_count']} matching values ({match_pct:.1f}% match)",
                    "is_primary_key": unique1 == total1 if len(card1) == 2 else False,
                    "is_foreign_key": unique2 == total2 if len(card2) == 2 else False,
                    "recommendation": "keep" if confidence in ["high", "medium"] else "review"
                }
                
                relationships.append(relationship)
        
        logger.info(f"Generated {len(relationships)} relationships from data overlaps (fallback mode)")
        return relationships
    
    def _prepare_data_overlap_description(self, overlaps: Dict) -> str:
        """Prepare human-readable description of data overlaps for LLM"""
        if not overlaps:
            return "No data overlaps computed yet."
        
        description = ""
        for pair_key, overlap_list in overlaps.items():
            if not overlap_list:
                continue
                
            tables = pair_key.split("|")
            description += f"\n=== {tables[0]} ↔ {tables[1]} ===\n"
            
            # Sort by overlap count (most significant first)
            sorted_overlaps = sorted(overlap_list, key=lambda x: x['overlap_count'], reverse=True)
            
            for overlap in sorted_overlaps[:10]:  # Top 10 overlaps per table pair
                description += f"\n  Column Pair: {overlap['column1']} ↔ {overlap['column2']}\n"
                description += f"    - Matching Values: {overlap['overlap_count']}\n"
                description += f"    - Match % in {tables[0]}: {overlap['match_pct_table1']}%\n"
                description += f"    - Match % in {tables[1]}: {overlap['match_pct_table2']}%\n"
                description += f"    - Cardinality: {overlap['cardinality_1']} vs {overlap['cardinality_2']}\n"
                description += f"    - Inferred Type: {overlap['inferred_relationship_type']}\n"
                description += f"    - Sample Matching Values: {overlap['sample_matching_values']}\n"
        
        return description if description else "No significant data overlaps found between tables."
    
    def _prepare_schema_description(self) -> str:
        """Prepare human-readable schema description for LLM"""
        description = "DATASET SCHEMAS:\n\n"
        
        for name, schema in self.schemas.items():
            description += f"=== {name} ===\n"
            description += f"Rows: {schema['rows']}, Columns: {schema['columns']}\n\n"
            description += "Columns:\n"
            
            for col in schema['column_details']:
                description += f"  - {col['name']} ({col['dtype']}, {col['type']})\n"
                description += f"    Non-null: {col['non_null']}, Unique: {col['unique_count']}\n"
                description += f"    Sample values: {col['sample_values']}\n"
            
            description += "\n"
        
        return description
    
    def get_unified_view(self) -> str:
        """
        Get LLM-generated unified view of all datasets
        Returns business context and how datasets relate
        """
        if not self.datasets:
            return "No datasets loaded"
        
        schema_description = self._prepare_schema_description()
        relationships_text = self._format_relationships()
        
        prompt = f"""You are a business intelligence expert. Analyze these datasets and provide a comprehensive overview.

{schema_description}

Detected Relationships:
{relationships_text}

Task: Provide a unified business view of this data:
1. What business domain does this data represent? (e.g., Sales, HR, Finance, Inventory)
2. What are the key entities? (e.g., Customers, Products, Orders)
3. How do the datasets connect to tell a business story?
4. What kind of analysis would be most valuable?
5. What insights can we derive from this data?

Provide a clear, structured analysis in markdown format."""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating unified view: {str(e)}")
            return f"Error generating unified view: {str(e)}"
    
    def _format_relationships(self) -> str:
        """Format relationships for display"""
        if not self.relationships:
            return "No relationships detected yet."
        
        text = ""
        for rel in self.relationships:
            text += f"- {rel['file1']}.{rel['column1']} → {rel['file2']}.{rel['column2']}\n"
            text += f"  Type: {rel['relationship_type']}, Confidence: {rel['confidence']}\n"
            text += f"  Reasoning: {rel['reasoning']}\n\n"
        
        return text
    
    def suggest_joins(self) -> List[Dict]:
        """
        Suggest possible join operations based on detected relationships
        """
        suggestions = []
        
        for rel in self.relationships:
            if rel['confidence'] in ['high', 'medium']:
                suggestion = {
                    "left_dataset": rel['file1'],
                    "right_dataset": rel['file2'],
                    "left_on": rel['column1'],
                    "right_on": rel['column2'],
                    "how": self._infer_join_type(rel['relationship_type']),
                    "confidence": rel['confidence'],
                    "reasoning": rel['reasoning']
                }
                suggestions.append(suggestion)
        
        return suggestions
    
    def _infer_join_type(self, relationship_type: str) -> str:
        """Infer pandas join type from relationship type"""
        mapping = {
            "one-to-many": "left",
            "many-to-one": "left",
            "one-to-one": "inner",
            "many-to-many": "outer"
        }
        return mapping.get(relationship_type, "inner")
    
    def get_combined_statistics(self) -> Dict:
        """Get combined statistics across all datasets"""
        stats = {
            "total_datasets": len(self.datasets),
            "total_rows": sum(len(df) for df in self.datasets.values()),
            "total_columns": sum(len(df.columns) for df in self.datasets.values()),
            "datasets": {}
        }
        
        for name, df in self.datasets.items():
            stats["datasets"][name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns)
            }
        
        return stats
