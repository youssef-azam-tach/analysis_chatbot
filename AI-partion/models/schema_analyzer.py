"""
Schema Analyzer Module
Automatically understands dataset schema and detects relationships between files using LLM
Enhanced with data-based relationship detection similar to ERD
"""

import math

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
import ollama
import re
from collections import Counter

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
        self.confirmed_relationships: List[Dict[str, Any]] = []
        self.rejected_relationships: List[Dict[str, Any]] = []
        self.suspicious_columns: List[Dict[str, Any]] = []

    # ── Performance: max rows used for value-set / overlap computations ──
    _SAMPLE_LIMIT = 20_000

    # ── Profiling engine weights & threshold ──
    _W_CARDINALITY  = 0.25
    _W_UNIQUENESS   = 0.25
    _W_SEQUENTIAL   = 0.15
    _W_AGGREGATION  = 0.10
    _W_DISTRIBUTION = 0.10
    _W_NAME_PRIOR   = 0.15
    _ID_THRESHOLD   = 55   # total_score >= this → is_id_likely

    _TEMPORAL_AUDIT_FIELDS = {
        "date", "startdate", "enddate", "duedate", "modifieddate",
        "createddate", "updateddate", "created_at", "updated_at",
        "modified_at", "timestamp", "audit", "rowguid",
        "modifiedby", "createdby", "updatedby", "lastmodifiedby",
    }
    _MEASURE_FIELDS = {
        "quantity", "amount", "rate", "taxrate", "price", "cost",
        "total", "subtotal", "discount", "revenue", "sales", "profit",
        "score", "percent", "percentage", "ratio",
    }
    _DESCRIPTIVE_FIELDS = {
        "name", "type", "description", "status", "title", "category", "label", "text"
    }
    
    def add_dataset(self, name: str, df: pd.DataFrame):
        """Add a dataset for analysis"""
        self.datasets[name] = df
        self.schemas[name] = self._extract_schema(name, df)
        logger.info(f"Added dataset: {name} with {len(df)} rows and {len(df.columns)} columns")
    
    def _extract_schema(self, name: str, df: pd.DataFrame) -> Dict:
        """Extract detailed schema information from dataframe, including ID profiling."""
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
            
            # Attach ID profiling signals
            col_info["id_profile"] = self._profile_column_id(df[col], col)
            
            schema["column_details"].append(col_info)
        
        return schema

    def _is_temporal_or_audit(self, column: str, dtype: str) -> bool:
        col = column.lower().strip().replace(" ", "")
        if col in self._TEMPORAL_AUDIT_FIELDS:
            return True
        if "date" in col:
            return True
        if "time" in col and "id" not in col:
            return True
        return "datetime" in dtype.lower()

    def _is_measure_field(self, column: str) -> bool:
        col = column.lower().strip().replace(" ", "")
        return col in self._MEASURE_FIELDS or any(token in col for token in ["amount", "rate", "quantity", "price", "taxrate", "revenue", "profit"]) 

    def _is_descriptive_field(self, column: str) -> bool:
        col = column.lower().strip().replace(" ", "")
        return col in self._DESCRIPTIVE_FIELDS or any(token in col for token in ["name", "type", "description", "status", "title"])

    def _is_disallowed_for_key(self, column: str, dtype: str) -> bool:
        return self._is_temporal_or_audit(column, dtype) or self._is_measure_field(column) or self._is_descriptive_field(column)

    def _is_identifier_like(self, column: str) -> bool:
        col = column.lower().strip().replace(" ", "")
        return (
            col == "id"
            or col.endswith("id")
            or col.endswith("_id")
            or col.endswith("key")
            or col.endswith("_key")
            or col.endswith("code")
            or col.endswith("_code")
            or col.endswith("number")
            or col.endswith("_number")
            or col.endswith("no")
            or col.endswith("_no")
        )

    def _normalized_tokens(self, text: str) -> Set[str]:
        value = str(text)
        value = value.replace("::", "_").replace("->", "_").replace(".", "_").replace("-", "_").replace(" ", "_")
        value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
        raw = [part.lower() for part in value.split("_") if part]
        stop = {
            "tbl", "table", "dim", "fact", "data", "info", "header", "detail", "history",
            "sales", "production", "person", "purchasing", "xlsx", "csv", "json", "txt", "file",
            "src", "stg", "raw", "cleaned", "dataset",
        }
        tokens = {t for t in raw if t and t not in stop and len(t) > 2 and ":" not in t and "/" not in t}
        return tokens

    def _identifier_entity_core(self, name: str) -> str:
        """Extract entity core from identifier-like name: TerritoryID -> territory."""
        value = str(name).strip()
        value = value.replace("::", "_").replace(".", "_").replace("-", "_").replace(" ", "_")
        value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower()
        value = re.sub(r"[^a-z0-9_]", "", value)
        value = re.sub(r"(_id|id|_key|key|_code|code|_number|number|_no|no)$", "", value)
        parts = [p for p in value.split("_") if p]
        if not parts:
            return ""
        stop = {
            "tbl", "table", "dim", "fact", "data", "info", "header", "detail", "history",
            "sales", "production", "person", "purchasing", "xlsx", "csv", "json", "txt", "file",
            "src", "stg", "raw", "cleaned", "dataset",
        }
        filtered = [p for p in parts if p not in stop]
        if not filtered:
            return ""
        core = "_".join(filtered)
        if core.endswith("ies") and len(core) > 4:
            core = core[:-3] + "y"
        elif core.endswith("s") and not core.endswith("ss") and len(core) > 3:
            core = core[:-1]
        return core

    def _identifier_entity_compatible(self, child_table: str, child_col: str, parent_table: str, parent_pk: str) -> bool:
        child_core = self._identifier_entity_core(child_col)
        parent_core = self._identifier_entity_core(parent_pk)

        if not child_core or not parent_core:
            return False
        if child_core == parent_core:
            return True

        child_table_tokens = self._normalized_tokens(child_table)
        parent_table_tokens = self._normalized_tokens(parent_table)

        child_core_tokens = set(child_core.split("_"))
        parent_core_tokens = set(parent_core.split("_"))

        child_matches_parent_table = len(child_core_tokens & parent_table_tokens) > 0
        parent_matches_child_table = len(parent_core_tokens & child_table_tokens) > 0

        # Strict rule: only accept non-exact core matches when BOTH sides have table-level semantic evidence.
        return child_matches_parent_table and parent_matches_child_table

    def _are_types_fk_compatible(self, child_series: pd.Series, parent_series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(child_series) or pd.api.types.is_datetime64_any_dtype(parent_series):
            return False

        child_num = pd.api.types.is_numeric_dtype(child_series)
        parent_num = pd.api.types.is_numeric_dtype(parent_series)
        if child_num != parent_num:
            return False

        if child_num and (pd.api.types.is_float_dtype(child_series) or pd.api.types.is_float_dtype(parent_series)):
            return False

        return True

    def _semantic_entity_alignment(self, child_table: str, child_col: str, parent_table: str, parent_pk: str) -> int:
        parent_tokens = self._normalized_tokens(parent_table) | self._normalized_tokens(parent_pk)
        child_tokens = self._normalized_tokens(child_table) | self._normalized_tokens(child_col)
        return len(parent_tokens & child_tokens)

    def _simulate_sql_fk_constraint(
        self,
        child_series: pd.Series,
        parent_series: pd.Series,
    ) -> Tuple[bool, str]:
        parent_non_null = int(parent_series.notna().sum())
        parent_unique_non_null = int(parent_series.dropna().nunique())
        parent_is_unique_non_null = parent_non_null > 0 and parent_non_null == len(parent_series) and parent_unique_non_null == parent_non_null
        if not parent_is_unique_non_null:
            return False, "Parent column is not unique+non-null, so it cannot be referenced as a PK/UNIQUE key."

        if not self._are_types_fk_compatible(child_series, parent_series):
            return False, "Child/parent data types are not FK-compatible under strict SQL typing rules."

        child_values = self._value_set(child_series)
        parent_values = self._value_set(parent_series)
        missing = child_values - parent_values
        if missing:
            return False, f"FK check failed: {len(missing)} child value(s) are missing in parent key domain."

        return True, "Constraint simulation passed."

    def _is_stable_deterministic_dtype(self, series: pd.Series) -> bool:
        # SQL-Server-like strictness: avoid floating measures/temporal for key enforcement simulation
        if pd.api.types.is_datetime64_any_dtype(series):
            return False
        if pd.api.types.is_float_dtype(series):
            return False
        return (
            pd.api.types.is_integer_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        )

    def _value_set(self, series: pd.Series, limit: Optional[int] = None) -> Set[str]:
        """Build a set of string-cast non-null values with optional sampling.

        When *limit* is None the class-level ``_SAMPLE_LIMIT`` is used.
        Pass ``limit=0`` to disable sampling and use all rows.
        """
        if series is None:
            return set()
        non_null = series.dropna()
        cap = limit if limit is not None else self._SAMPLE_LIMIT
        if cap and len(non_null) > cap:
            non_null = non_null.sample(n=cap, random_state=42)
        return {
            str(v).strip()
            for v in non_null.astype(str)
            if str(v).strip() != ""
        }

    # ═══════════════════════════════════════════════════════════
    #  DATA PROFILING – ID / KEY CONFIDENCE ENGINE
    # ═══════════════════════════════════════════════════════════
    # Each sub-scorer returns 0-100. The weighted total decides
    # whether a column is likely an identifier (threshold >= _ID_THRESHOLD).

    @staticmethod
    def _prof_cardinality_ratio(series: pd.Series) -> float:
        """nunique(non-null) / count(non-null), range 0.0–1.0."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return 0.0
        return non_null.nunique() / len(non_null)

    @staticmethod
    def _prof_uniqueness_score(cardinality: float) -> float:
        """Map cardinality ratio → 0-100 score."""
        if cardinality >= 0.99:
            return 100
        if cardinality >= 0.95:
            return 90
        if cardinality >= 0.90:
            return 75
        if cardinality >= 0.75:
            return 40
        if cardinality >= 0.50:
            return 15
        return 0

    @staticmethod
    def _prof_sequential_score(series: pd.Series) -> float:
        """Detect auto-increment patterns (diff==1 ratio), 0-100."""
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

    @staticmethod
    def _prof_aggregation_score(series: pd.Series) -> float:
        """High CV (std/mean) → meaningless aggregation → likely ID.  0-100."""
        if not pd.api.types.is_numeric_dtype(series):
            return 0
        non_null = series.dropna()
        if len(non_null) < 10:
            return 0
        mean = float(non_null.mean())
        if mean == 0:
            return 0
        cv = float(non_null.std()) / abs(mean)
        if cv > 1.5:
            return 80
        if cv > 1.0:
            return 60
        if cv > 0.5:
            return 30
        return 0

    @staticmethod
    def _prof_distribution_score(series: pd.Series) -> float:
        """Low skewness → uniform-like → ID.  Uses manual skewness (no scipy).  0-100."""
        if not pd.api.types.is_numeric_dtype(series):
            return 0
        non_null = series.dropna()
        if len(non_null) < 10:
            return 0
        # Manual Pearson skewness: E[((x-mu)/σ)^3]
        mean = float(non_null.mean())
        std = float(non_null.std())
        if std == 0:
            return 0
        skewness = abs(float(((non_null - mean) / std).pow(3).mean()))
        if skewness < 0.3:
            return 70
        if skewness < 0.7:
            return 40
        return 0

    @staticmethod
    def _prof_entropy_score(series: pd.Series) -> float:
        """High character-level entropy → random IDs / UUIDs.  0-100."""
        non_null = series.dropna().astype(str)
        if len(non_null) < 10:
            return 0
        sample_str = ''.join(non_null.head(200))
        if not sample_str:
            return 0
        freq = Counter(sample_str)
        total = len(sample_str)
        probs = [v / total for v in freq.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        if entropy > 4:
            return 80
        if entropy > 3:
            return 50
        return 0

    def _prof_name_prior(self, col_name: str) -> float:
        """Name-only prior: 0-100 (100 = definitely ID by name, 0 = no signal)."""
        name = str(col_name).lower().strip().replace(" ", "")
        # Exact ID names
        if name in {"id", "uuid", "guid", "rowid", "row_id", "index", "pk", "fk"}:
            return 100
        # Suffix patterns
        for suffix in ("_id", "id", "_key", "key", "_pk", "_fk",
                        "_guid", "_uuid", "_code", "code",
                        "_number", "number", "_no", "no"):
            if name.endswith(suffix) and len(name) > len(suffix):
                return 95
        # CamelCase ending in ID (BusinessEntityID)
        if re.search(r'[a-z]ID$', str(col_name)):
            return 95
        # Token check
        tokens = {t for t in re.split(r'[^a-z0-9]+', name) if t}
        if tokens & {"id", "key", "pk", "fk", "uuid", "guid", "rowguid", "hash", "token"}:
            return 90
        return 0

    def _profile_column_id(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Run the full ID profiling engine on a single column.

        Returns a dict with sub-scores, total_score (0-100), and is_id_likely bool.
        """
        cardinality = self._prof_cardinality_ratio(series)
        name_prior  = self._prof_name_prior(col_name)

        # Float columns are almost never IDs – penalty
        is_float = pd.api.types.is_float_dtype(series)
        card_raw = cardinality * 100
        uniq_raw = self._prof_uniqueness_score(cardinality)
        if is_float:
            card_raw *= 0.25
            uniq_raw *= 0.25

        seq  = self._prof_sequential_score(series) if not is_float else 0
        agg  = self._prof_aggregation_score(series)
        dist = self._prof_distribution_score(series)

        total = (
            card_raw    * self._W_CARDINALITY  +
            uniq_raw    * self._W_UNIQUENESS   +
            seq         * self._W_SEQUENTIAL   +
            agg         * self._W_AGGREGATION  +
            dist        * self._W_DISTRIBUTION +
            name_prior  * self._W_NAME_PRIOR
        )

        # Name-prior dominates when it is very high (explicit "ID" suffix)
        if name_prior >= 90:
            total = max(total, name_prior)

        profile = {
            "cardinality":       round(card_raw, 2),
            "uniqueness_score":  round(uniq_raw, 2),
            "sequential_score":  round(seq, 2),
            "aggregation_score": round(agg, 2),
            "distribution_score": round(dist, 2),
            "name_prior":        round(name_prior, 2),
            "total_score":       round(total, 2),
            "is_id_likely":      total >= self._ID_THRESHOLD,
        }
        return profile

    def _detect_primary_keys(self) -> Dict[str, Dict[str, List[str]]]:
        """Detect primary key candidates for each table.

        Returns ``{table: {"strict": [...], "probable": [...]}}``.

        * **strict**: 100% non-null, 100% unique, deterministic dtype,
          not disallowed (temporal/measure/descriptive).
        * **probable**: high ID confidence (total_score >= _ID_THRESHOLD),
          cardinality >= 0.97, null_rate <= 2%.  These are plausible PKs
          that may have minor data-quality issues.
        """
        pk_map: Dict[str, Dict[str, List[str]]] = {}
        for table_name, df in self.datasets.items():
            strict: List[str] = []
            probable: List[Tuple[str, float]] = []  # (col, total_score)
            total_rows = len(df)
            if total_rows == 0:
                pk_map[table_name] = {"strict": [], "probable": []}
                continue

            # Build a fast lookup for id_profiles from schema
            schema = self.schemas.get(table_name, {})
            id_profiles: Dict[str, Dict] = {}
            for cd in schema.get("column_details", []):
                id_profiles[cd["name"]] = cd.get("id_profile", {})

            for col in df.columns:
                col_series = df[col]
                non_null = int(col_series.notna().sum())
                unique_non_null = int(col_series.dropna().nunique())
                dtype = str(col_series.dtype)

                if self._is_disallowed_for_key(col, dtype):
                    continue
                if not self._is_stable_deterministic_dtype(col_series):
                    continue

                is_unique = unique_non_null == non_null and non_null > 0
                is_non_null = non_null == total_rows

                if is_unique and is_non_null:
                    strict.append(col)
                else:
                    # Check probable PK: high ID score + near-perfect cardinality + low nulls
                    profile = id_profiles.get(col, {})
                    total_score = profile.get("total_score", 0)
                    if total_rows > 0:
                        null_rate = (total_rows - non_null) / total_rows
                        cardinality = unique_non_null / non_null if non_null > 0 else 0
                    else:
                        null_rate = 1.0
                        cardinality = 0

                    if (
                        total_score >= self._ID_THRESHOLD
                        and cardinality >= 0.97
                        and null_rate <= 0.02
                    ):
                        probable.append((col, total_score))

            strict.sort(key=lambda c: (len(c), c.lower()))
            # Sort probable by total_score descending, then name
            probable.sort(key=lambda t: (-t[1], len(t[0]), t[0].lower()))
            pk_map[table_name] = {
                "strict": strict,
                "probable": [p[0] for p in probable],
            }

        return pk_map
    
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
    
    def _get_column_id_profile(self, table_name: str, col_name: str) -> Dict[str, Any]:
        """Retrieve the id_profile from pre-computed schema, or compute on the fly."""
        schema = self.schemas.get(table_name, {})
        for cd in schema.get("column_details", []):
            if cd["name"] == col_name:
                return cd.get("id_profile", {})
        # Fallback: compute if schema missing
        df = self.datasets.get(table_name)
        if df is not None and col_name in df.columns:
            return self._profile_column_id(df[col_name], col_name)
        return {}

    def _is_fk_candidate(self, table_name: str, col_name: str, dtype: str, series: pd.Series) -> bool:
        """Filter: only evaluate child columns that look like FK candidates.

        Criteria:
        - Not disallowed (temporal / measure / descriptive)
        - Stable deterministic dtype
        - id_profile total_score >= 45  (lower than PK threshold; we want to
          catch FKs that are clearly key-like but not necessarily unique)
        """
        if self._is_disallowed_for_key(col_name, dtype):
            return False
        if not self._is_stable_deterministic_dtype(series):
            return False
        profile = self._get_column_id_profile(table_name, col_name)
        return profile.get("total_score", 0) >= 45

    def analyze_relationships(self) -> List[Dict]:
        """Strict relational modeling analysis using PK/FK integrity rules.

        Uses the new strict/probable PK map and data-profiling FK candidate
        filtering.  Confirmed relationships require strict PK on parent side.
        Suspicious relationships may consider probable PKs on parent side.
        """
        if len(self.datasets) < 2:
            logger.warning("Need at least 2 datasets to analyze relationships")
            self.relationships = []
            self.confirmed_relationships = []
            self.rejected_relationships = []
            self.suspicious_columns = []
            return []

        pk_map = self._detect_primary_keys()

        confirmed: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        suspicious: List[Dict[str, Any]] = []
        seen_confirmed: Set[Tuple[str, str, str, str]] = set()
        seen_rejected: Set[Tuple[str, str, str, str, str]] = set()
        seen_suspicious: Set[Tuple[str, str, str]] = set()

        # Helper to build rejection entry
        def _reject(child_t, child_c, parent_t, parent_pk, child_vals, parent_vals, fk_cov, pk_cov, olap, reason):
            r_key = (child_t, child_c, parent_t, parent_pk, reason)
            if r_key not in seen_rejected:
                rejected.append({
                    "file1": child_t, "file2": parent_t,
                    "column1": child_c, "column2": parent_pk,
                    "overlap_count": olap,
                    "child_distinct": len(child_vals),
                    "parent_distinct": len(parent_vals),
                    "child_to_parent_coverage": round(fk_cov * 100, 2),
                    "parent_coverage": round(pk_cov * 100, 2),
                    "reason": reason,
                })
                seen_rejected.add(r_key)

        # ── Phase 1: Confirmed relationships (parent key MUST be strict PK) ──
        for parent_table, parent_df in self.datasets.items():
            strict_pks = pk_map.get(parent_table, {}).get("strict", [])
            if not strict_pks:
                continue

            for parent_pk in strict_pks:
                parent_values = self._value_set(parent_df[parent_pk])
                if not parent_values:
                    continue

                for child_table, child_df in self.datasets.items():
                    if child_table == parent_table:
                        continue

                    for child_col in child_df.columns:
                        child_dtype = str(child_df[child_col].dtype)

                        # FK candidate filtering via id_profile
                        if not self._is_fk_candidate(child_table, child_col, child_dtype, child_df[child_col]):
                            continue

                        child_values = self._value_set(child_df[child_col])
                        if not child_values:
                            continue

                        intersection = child_values & parent_values
                        overlap = len(intersection)
                        if overlap == 0:
                            continue

                        fk_coverage = overlap / len(child_values)
                        pk_coverage = overlap / len(parent_values)
                        child_non_null = int(child_df[child_col].notna().sum())
                        child_unique_non_null = int(child_df[child_col].dropna().nunique())
                        child_is_unique = child_non_null > 0 and child_unique_non_null == child_non_null
                        type_compatible = self._are_types_fk_compatible(child_df[child_col], parent_df[parent_pk])
                        semantic_alignment = self._semantic_entity_alignment(child_table, child_col, parent_table, parent_pk)
                        child_identifier_like = self._is_identifier_like(child_col)
                        parent_identifier_like = self._is_identifier_like(parent_pk)

                        if not child_identifier_like or not parent_identifier_like:
                            _reject(child_table, child_col, parent_table, parent_pk,
                                    child_values, parent_values, fk_coverage, pk_coverage, overlap,
                                    "Rejected as SQL FK: child/parent columns are not identifier-like keys (attribute-to-attribute mapping).")
                            continue

                        if semantic_alignment == 0:
                            _reject(child_table, child_col, parent_table, parent_pk,
                                    child_values, parent_values, fk_coverage, pk_coverage, overlap,
                                    "Rejected ID-to-ID relation across semantically unrelated entities (no entity alignment evidence).")
                            continue

                        if not self._identifier_entity_compatible(child_table, child_col, parent_table, parent_pk):
                            _reject(child_table, child_col, parent_table, parent_pk,
                                    child_values, parent_values, fk_coverage, pk_coverage, overlap,
                                    "Rejected ID-to-ID relation: identifier entity core mismatch (e.g., ShoppingCartItemID cannot reference TerritoryID).")
                            continue

                        if not type_compatible:
                            _reject(child_table, child_col, parent_table, parent_pk,
                                    child_values, parent_values, fk_coverage, pk_coverage, overlap,
                                    "Rejected as SQL FK: incompatible data types between child and parent key columns.")
                            continue

                        key = (child_table, child_col, parent_table, parent_pk)
                        can_apply_fk, fk_reason = self._simulate_sql_fk_constraint(child_df[child_col], parent_df[parent_pk])

                        if can_apply_fk:
                            rel_type = "one-to-one" if child_is_unique else "many-to-one"
                            if rel_type == "one-to-one":
                                parent_tokens = self._normalized_tokens(parent_table)
                                child_tokens = self._normalized_tokens(child_table)
                                if len(parent_tokens & child_tokens) == 0:
                                    _reject(child_table, child_col, parent_table, parent_pk,
                                            child_values, parent_values, fk_coverage, pk_coverage, overlap,
                                            "Rejected 1:1: uniqueness exists but no clear parent-child entity extension semantics.")
                                    continue

                            child_profile = self._get_column_id_profile(child_table, child_col)
                            reasoning = (
                                f"Engine-valid FK simulation passed: all {len(child_values)} distinct child values "
                                f"exist in parent key values ({overlap}/{len(child_values)}). "
                                f"Child ID confidence: {child_profile.get('total_score', 0):.0f}%."
                            )
                            if key not in seen_confirmed:
                                confirmed.append({
                                    "file1": child_table,
                                    "file2": parent_table,
                                    "column1": child_col,
                                    "column2": parent_pk,
                                    "relationship_type": rel_type,
                                    "confidence": "high",
                                    "match_percentage": round(fk_coverage * 100, 2),
                                    "overlap_count": overlap,
                                    "reasoning": reasoning,
                                    "is_primary_key": True,
                                    "is_foreign_key": True,
                                    "recommendation": "keep",
                                    "child_id_score": child_profile.get("total_score", 0),
                                })
                                seen_confirmed.add(key)
                        else:
                            _reject(child_table, child_col, parent_table, parent_pk,
                                    child_values, parent_values, fk_coverage, pk_coverage, overlap,
                                    fk_reason)

                            # Promote to suspicious if substantial coverage
                            if fk_coverage >= 0.4:
                                s_key = (child_table, child_col, f"{parent_table}.{parent_pk}")
                                if s_key not in seen_suspicious:
                                    child_profile = self._get_column_id_profile(child_table, child_col)
                                    suspicious.append({
                                        "table": child_table,
                                        "column": child_col,
                                        "possible_reference": f"{parent_table}.{parent_pk}",
                                        "overlap_count": overlap,
                                        "coverage": round(fk_coverage * 100, 2),
                                        "child_id_score": child_profile.get("total_score", 0),
                                        "parent_key_tier": "strict",
                                        "reason": f"Partial referential overlap detected (FK constraint failed: {fk_reason}).",
                                    })
                                    seen_suspicious.add(s_key)

        # ── Phase 2: Suspicious relationships against probable PKs ──────────
        # These are NOT confirmed – parent key is not strict PK.
        confirmed_keys = set(seen_confirmed)
        for parent_table, parent_df in self.datasets.items():
            probable_pks = pk_map.get(parent_table, {}).get("probable", [])
            if not probable_pks:
                continue

            for parent_pk in probable_pks:
                parent_values = self._value_set(parent_df[parent_pk])
                if not parent_values:
                    continue

                for child_table, child_df in self.datasets.items():
                    if child_table == parent_table:
                        continue

                    for child_col in child_df.columns:
                        child_dtype = str(child_df[child_col].dtype)
                        if not self._is_fk_candidate(child_table, child_col, child_dtype, child_df[child_col]):
                            continue

                        # Skip if already confirmed or already suspicious for this pair
                        key = (child_table, child_col, parent_table, parent_pk)
                        if key in confirmed_keys:
                            continue
                        s_key = (child_table, child_col, f"{parent_table}.{parent_pk}")
                        if s_key in seen_suspicious:
                            continue

                        if not self._is_identifier_like(child_col) or not self._is_identifier_like(parent_pk):
                            continue
                        if not self._are_types_fk_compatible(child_df[child_col], parent_df[parent_pk]):
                            continue
                        if not self._identifier_entity_compatible(child_table, child_col, parent_table, parent_pk):
                            continue

                        child_values = self._value_set(child_df[child_col])
                        if not child_values:
                            continue

                        intersection = child_values & parent_values
                        overlap = len(intersection)
                        if overlap == 0:
                            continue

                        fk_coverage = overlap / len(child_values)
                        if fk_coverage < 0.4:
                            continue

                        child_profile = self._get_column_id_profile(child_table, child_col)
                        parent_profile = self._get_column_id_profile(parent_table, parent_pk)
                        suspicious.append({
                            "table": child_table,
                            "column": child_col,
                            "possible_reference": f"{parent_table}.{parent_pk}",
                            "overlap_count": overlap,
                            "coverage": round(fk_coverage * 100, 2),
                            "child_id_score": child_profile.get("total_score", 0),
                            "parent_id_score": parent_profile.get("total_score", 0),
                            "parent_key_tier": "probable",
                            "reason": (
                                f"Parent key '{parent_pk}' is probable PK (not strict). "
                                f"Coverage={fk_coverage*100:.1f}%. Cannot confirm without strict parent uniqueness."
                            ),
                        })
                        seen_suspicious.add(s_key)

        # ── Phase 3: Flag disallowed columns as suspicious ──────────────────
        for table_name, schema in self.schemas.items():
            for col in schema["column_details"]:
                if self._is_disallowed_for_key(col["name"], col["dtype"]):
                    s_key = (table_name, col["name"], "disallowed")
                    if s_key not in seen_suspicious:
                        suspicious.append({
                            "table": table_name,
                            "column": col["name"],
                            "possible_reference": None,
                            "overlap_count": 0,
                            "coverage": 0.0,
                            "reason": "Excluded from FK/PK validation (temporal, audit, or measure field).",
                        })
                        seen_suspicious.add(s_key)

        # Sort suspicious by coverage desc for easy review
        suspicious.sort(key=lambda x: -x.get("coverage", 0))

        self.confirmed_relationships = confirmed
        self.rejected_relationships = rejected
        self.suspicious_columns = suspicious
        self.relationships = confirmed
        self._last_pk_map = pk_map  # cache for get_relationship_audit
        self.junction_tables = self._detect_junction_tables(confirmed, pk_map)
        self.final_erd_structure = self._build_final_erd_structure(confirmed)
        self.modeling_observations = self._build_modeling_observations(pk_map, confirmed, rejected, suspicious)

        n_susp_refs = sum(1 for s in suspicious if s.get("possible_reference"))
        if confirmed:
            self.erd_summary = (
                f"Enforceable ERD: {len(confirmed)} FK relationship(s) passed SQL-Server-like constraint checks. "
                f"{n_susp_refs} suspicious relationship(s) need review. "
                f"Only engine-valid constraints are included."
            )
        else:
            self.erd_summary = (
                f"No enforceable FK constraints detected under strict SQL-Server-like validation. "
                f"{n_susp_refs} suspicious relationship(s) found that may warrant data cleanup."
            )
        logger.info(
            "Schema analysis: %d confirmed, %d rejected, %d suspicious",
            len(confirmed), len(rejected), len(suspicious)
        )
        return self.relationships

    def get_relationship_audit(self) -> Dict[str, Any]:
        pk_map = getattr(self, "_last_pk_map", None)
        if pk_map is None:
            pk_map = self._detect_primary_keys() if self.datasets else {}

        # Build per-table id_profiles summary from schema
        id_profiles_summary: Dict[str, Dict[str, Dict]] = {}
        for table_name, schema in self.schemas.items():
            id_profiles_summary[table_name] = {}
            for cd in schema.get("column_details", []):
                prof = cd.get("id_profile")
                if prof:
                    id_profiles_summary[table_name][cd["name"]] = prof

        return {
            "confirmed_relationships": self.confirmed_relationships,
            "rejected_relationships": self.rejected_relationships,
            "suspicious_columns": self.suspicious_columns,
            "possible_junction_tables": getattr(self, "junction_tables", []),
            "final_erd_structure": getattr(self, "final_erd_structure", {}),
            "data_modeling_observations": getattr(self, "modeling_observations", []),
            "erd_summary": getattr(self, "erd_summary", ""),
            "pk_map": pk_map,
            "id_profiles": id_profiles_summary,
        }

    def _detect_junction_tables(self, confirmed: List[Dict[str, Any]], pk_map: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, Any]]:
        """Detect junction/associative tables that FK to multiple parents."""
        by_child: Dict[str, List[Dict[str, Any]]] = {}
        for rel in confirmed:
            by_child.setdefault(rel["file1"], []).append(rel)

        junctions: List[Dict[str, Any]] = []
        for child_table, rels in by_child.items():
            parent_tables = {r["file2"] for r in rels}
            if len(parent_tables) < 2:
                continue

            child_cols = set(self.datasets[child_table].columns)
            fk_cols = {r["column1"] for r in rels}
            # pk_map now has {"strict": [...], "probable": [...]}
            pk_entry = pk_map.get(child_table, {})
            pk_candidates = set(pk_entry.get("strict", []) + pk_entry.get("probable", []))

            if pk_candidates and (pk_candidates & fk_cols):
                extra_cols = list(child_cols - fk_cols)
                junctions.append({
                    "table": child_table,
                    "fk_count": len(fk_cols),
                    "parents": sorted(parent_tables),
                    "fk_columns": sorted(fk_cols),
                    "extra_columns": sorted(extra_cols),
                    "reason": "Table has enforceable FKs to multiple parents and behaves like an associative entity.",
                })

        return junctions

    def _build_final_erd_structure(self, confirmed: List[Dict[str, Any]]) -> Dict[str, Any]:
        entities = []
        for table_name, df in self.datasets.items():
            entities.append({
                "name": table_name,
                "rows": len(df),
                "columns": list(df.columns),
            })

        relationships = [
            {
                "child": rel["file1"],
                "parent": rel["file2"],
                "child_fk": rel["column1"],
                "parent_pk": rel["column2"],
                "cardinality": rel["relationship_type"],
                "engine_valid": True,
            }
            for rel in confirmed
        ]

        return {
            "entities": entities,
            "relationships": relationships,
        }

    def _build_modeling_observations(
        self,
        pk_map: Dict[str, Dict[str, List[str]]],
        confirmed: List[Dict[str, Any]],
        rejected: List[Dict[str, Any]],
        suspicious: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        observations: List[str] = []

        # Tables with no strict PK
        tables_without_strict = [
            t for t, entry in pk_map.items()
            if len(entry.get("strict", [])) == 0
        ]
        tables_with_probable_only = [
            t for t in tables_without_strict
            if len(pk_map.get(t, {}).get("probable", [])) > 0
        ]
        tables_no_pk_at_all = [
            t for t in tables_without_strict
            if len(pk_map.get(t, {}).get("probable", [])) == 0
        ]

        if tables_no_pk_at_all:
            observations.append(
                f"{len(tables_no_pk_at_all)} table(s) have no PK candidate (strict or probable): "
                f"{', '.join(sorted(tables_no_pk_at_all)[:8])}."
            )
        if tables_with_probable_only:
            observations.append(
                f"{len(tables_with_probable_only)} table(s) have probable PKs but no strict PK: "
                f"{', '.join(sorted(tables_with_probable_only)[:8])}. "
                "These may need null cleanup or dedup to become enforceable."
            )

        if rejected:
            observations.append(
                f"{len(rejected)} candidate relationship(s) were rejected because SQL FK enforcement would fail."
            )

        if suspicious:
            ref_suspicious = [s for s in suspicious if s.get("possible_reference")]
            if ref_suspicious:
                observations.append(
                    f"{len(ref_suspicious)} suspicious relationship(s) detected with partial referential overlap "
                    f"that may indicate data-quality issues or missing constraints."
                )

        if confirmed:
            one_to_one = sum(1 for r in confirmed if r.get("relationship_type") == "one-to-one")
            many_to_one = sum(1 for r in confirmed if r.get("relationship_type") == "many-to-one")
            observations.append(
                f"Enforceable relationships: {len(confirmed)} total ({one_to_one} one-to-one, {many_to_one} many-to-one)."
            )
            observations.append("Model behaves as normalized OLTP structure where child tables reference stable parent keys.")
        else:
            observations.append("No enforceable FK constraints detected; schema likely needs explicit key constraints or data cleanup.")

        return observations
    
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
        """Return only safe joins that won't cause artificial many-to-many expansion."""
        suggestions = []

        for rel in self.relationships:
            child_table = rel['file1']
            parent_table = rel['file2']
            child_col = rel['column1']
            parent_col = rel['column2']

            child_df = self.datasets.get(child_table)
            parent_df = self.datasets.get(parent_table)
            if child_df is None or parent_df is None:
                continue

            if child_col not in child_df.columns or parent_col not in parent_df.columns:
                continue

            child_non_null = int(child_df[child_col].notna().sum())
            parent_non_null = int(parent_df[parent_col].notna().sum())
            child_unique = int(child_df[child_col].dropna().nunique()) == child_non_null if child_non_null > 0 else False
            parent_unique = int(parent_df[parent_col].dropna().nunique()) == parent_non_null if parent_non_null > 0 else False

            # safe OLTP join requires parent key uniqueness and avoids N:N explosion
            if not parent_unique:
                continue
            if not self._are_types_fk_compatible(child_df[child_col], parent_df[parent_col]):
                continue

            # Child -> Parent orientation is the only safe default for FK joins
            how = 'left' if not child_unique else 'inner'
            suggestion = {
                "left_dataset": child_table,
                "right_dataset": parent_table,
                "left_on": child_col,
                "right_on": parent_col,
                "how": how,
                "confidence": "high",
                "reasoning": "Engine-valid FK join: parent key unique, FK-compatible types, and parent-child direction enforced.",
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
