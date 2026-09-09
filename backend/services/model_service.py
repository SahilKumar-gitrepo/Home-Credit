"""
Model Service — Singleton Registry Pattern.

Supports multiple trained models and model chaining:
- Individual Models: XGBoost, LightGBM, Logistic Scorecard
- Virtual Chained Pipelines:
  - Weighted Ensemble (0.45 XGBoost + 0.45 LightGBM + 0.10 Logistic)
  - Two-Stage Cascade Hurdle (Logistic Fast Triage -> XGBoost Deep Assessment)
- Dynamic Active Model Switching
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd

from backend.config import settings

logger = logging.getLogger(__name__)

# Add src/ to path so we can import from existing pipeline if needed
_src_path = str(settings.models_dir.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)


class ModelService:
    """
    Singleton registry managing multiple trained PD models,
    virtual chained pipelines, and feature lookups.
    """

    _instance: Optional["ModelService"] = None

    def __init__(self):
        self._models: dict[str, dict] = {}
        self._active_model_id: str = "xgboost_calibrated"
        self._features_df: Optional[pd.DataFrame] = None
        self._feature_names: list[str] = []
        self._categorical_columns: list[str] = []
        self._categorical_mappings: dict[str, dict[Any, int]] = {}
        self._benchmark_data: dict = {}
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        """Discover and load all models and the feature dataset ONCE at startup."""
        if self._loaded:
            return

        logger.info("Initializing Model Service Registry from %s", settings.models_dir)

        # 1. Discover all pd_*.joblib models in models_dir
        if settings.models_dir.exists():
            for p in sorted(settings.models_dir.glob("pd_*.joblib")):
                model_id = p.stem.replace("pd_", "")
                try:
                    artifact = joblib.load(p)
                    if isinstance(artifact, dict):
                        self._models[model_id] = artifact
                    else:
                        self._models[model_id] = {
                            "model": artifact,
                            "features": [],
                            "categorical_columns": [],
                        }
                    logger.info("Loaded model artifact: %s from %s", model_id, p.name)
                except Exception as e:
                    logger.error("Failed to load model %s: %s", p.name, e)

        # Fallback for primary model if not found via discovery
        if "xgboost_calibrated" not in self._models and settings.model_path.exists():
            artifact = joblib.load(settings.model_path)
            self._models["xgboost_calibrated"] = artifact

        if "xgboost_calibrated" not in self._models:
            if "xgboost" in self._models:
                self._active_model_id = "xgboost"
            elif self._models:
                self._active_model_id = next(iter(self._models.keys()))

        if not self._models:
            raise FileNotFoundError(
                f"No model artifacts found in {settings.models_dir}."
            )

        # Reference features from the primary model (or first available)
        primary_artifact = (
            self._models.get("xgboost_calibrated")
            or self._models.get("xgboost")
            or next(iter(self._models.values()))
        )
        self._feature_names = primary_artifact.get("features", [])
        self._categorical_columns = primary_artifact.get("categorical_columns", [])

        # 2. Load benchmark data if available
        if settings.benchmark_path.exists():
            try:
                with open(settings.benchmark_path, "r") as f:
                    self._benchmark_data = json.load(f)
                logger.info("Loaded benchmark metrics for %d models", len(self._benchmark_data))
            except Exception as e:
                logger.warning("Could not load benchmark metrics: %s", e)

        # 3. Load feature dataset
        logger.info("Loading feature dataset from %s", settings.feature_path)
        if not settings.feature_path.exists():
            raise FileNotFoundError(
                f"Feature dataset not found: {settings.feature_path}"
            )

        self._features_df = pd.read_parquet(settings.feature_path)
        logger.info(
            "Feature dataset loaded: %d applicants, %d columns",
            len(self._features_df),
            len(self._features_df.columns),
        )

        # 4. Build global categorical code mappings from the dataset
        self._categorical_mappings = {}
        for col in self._categorical_columns:
            if col in self._features_df.columns:
                cats = pd.Categorical(self._features_df[col]).categories
                self._categorical_mappings[col] = {cat: idx for idx, cat in enumerate(cats)}
        logger.info("Built categorical mappings for %d columns", len(self._categorical_mappings))

        self._loaded = True

    # -------------------------------------------------------------------------
    # Properties & Getters
    # -------------------------------------------------------------------------

    @property
    def active_model_id(self) -> str:
        return self._active_model_id

    def set_active_model(self, model_id: str):
        """Set the active default model or chained pipeline."""
        available = [m["id"] for m in self.get_available_models()]
        if model_id not in available:
            raise ValueError(
                f"Unknown model '{model_id}'. Available: {available}"
            )
        self._active_model_id = model_id
        logger.info("Active model switched to: %s", model_id)

    @property
    def model(self):
        """Primary active model classifier (backward compatibility)."""
        if not self._loaded:
            self.load()
        if self._active_model_id in self._models:
            return self._models[self._active_model_id]["model"]
        return self._models.get("xgboost", next(iter(self._models.values())))["model"]

    @property
    def features_df(self) -> pd.DataFrame:
        if not self._loaded:
            self.load()
        return self._features_df

    @property
    def feature_names(self) -> list[str]:
        if not self._loaded:
            self.load()
        return self._feature_names

    @property
    def categorical_columns(self) -> list[str]:
        if not self._loaded:
            self.load()
        return self._categorical_columns

    @property
    def total_applicants(self) -> int:
        return len(self.features_df)

    def get_applicant(self, applicant_id: int) -> pd.DataFrame:
        """Retrieve one applicant row. Raises ValueError if not found."""
        if not self._loaded:
            self.load()
        mask = self.features_df["SK_ID_CURR"] == applicant_id
        row = self.features_df[mask]
        if row.empty:
            raise ValueError(f"Applicant {applicant_id} not found in feature dataset.")
        return row.copy()

    # -------------------------------------------------------------------------
    # Model Registry Management
    # -------------------------------------------------------------------------

    def get_available_models(self) -> list[dict]:
        """List all discrete trained models and virtual chained pipelines."""
        if not self._loaded:
            self.load()

        model_catalog = [
            {
                "id": "xgboost_calibrated",
                "name": "XGBoost (Calibrated)",
                "type": "Gradient Boosted Trees (Isotonic Calibrated)",
                "description": "Production champion: Tree boosting with post-hoc Isotonic calibration matching empirical default rates.",
                "is_pipeline": False,
                "is_trained": "xgboost_calibrated" in self._models,
            },
            {
                "id": "xgboost",
                "name": "XGBoost Classifier (Raw)",
                "type": "Gradient Boosted Trees (Histogram)",
                "description": "High-capacity non-linear model capturing complex cross-feature interactions (uncalibrated probability scores).",
                "is_pipeline": False,
                "is_trained": "xgboost" in self._models,
            },
            {
                "id": "lightgbm_calibrated",
                "name": "LightGBM (Calibrated)",
                "type": "Gradient Boosted Trees (Isotonic Calibrated)",
                "description": "Fast leaf-wise tree boosting with post-hoc Isotonic calibration for reliable probability estimates.",
                "is_pipeline": False,
                "is_trained": "lightgbm_calibrated" in self._models,
            },
            {
                "id": "lightgbm",
                "name": "LightGBM Classifier (Raw)",
                "type": "Gradient Boosted Trees (Leaf-Wise)",
                "description": "High-speed tree boosting with superior leaf-wise convergence and missing value handling.",
                "is_pipeline": False,
                "is_trained": "lightgbm" in self._models,
            },
            {
                "id": "logistic",
                "name": "Logistic Regression (Scorecard)",
                "type": "Linear / Basel Standard Scorecard",
                "description": "Standard regulatory baseline model with normalized scaling and median imputation.",
                "is_pipeline": False,
                "is_trained": "logistic" in self._models,
            },
            {
                "id": "chained_ensemble",
                "name": "Chained Weighted Ensemble",
                "type": "Multi-Model Blend (Ensemble)",
                "description": "Chains predictions across calibrated XGBoost (45%), calibrated LightGBM (45%), and Logistic Scorecard (10%).",
                "is_pipeline": True,
                "is_trained": True,
            },
            {
                "id": "cascade_hurdle",
                "name": "Two-Stage Cascade Hurdle",
                "type": "Sequential Decision Pipeline",
                "description": "Stage 1 Logistic filter for instant clear-cut approvals/rejections; Stage 2 deep calibrated tree scoring for borderline risks.",
                "is_pipeline": True,
                "is_trained": True,
            },
        ]

        result = []
        for m in model_catalog:
            m_copy = dict(m)
            m_copy["is_active"] = (m["id"] == self._active_model_id)
            # Attach benchmark metrics if available
            bench = self._benchmark_data.get(m["id"], {})
            m_copy["roc_auc"] = bench.get("roc_auc")
            m_copy["pr_auc"] = bench.get("pr_auc")
            m_copy["brier_score"] = bench.get("brier_score")
            result.append(m_copy)

        return result

    def get_model_comparison(self) -> dict:
        """Return benchmark comparison metrics across all models."""
        if not self._loaded:
            self.load()
        return self._benchmark_data

    # -------------------------------------------------------------------------
    # Input Preparation
    # -------------------------------------------------------------------------

    def prepare_model_input(
        self, df: pd.DataFrame, model_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Prepare one applicant's features for model ingestion."""
        if not self._loaded:
            self.load()

        mid = model_id or self._active_model_id
        if mid in ("chained_ensemble", "cascade_hurdle") or mid not in self._models:
            mid = "xgboost_calibrated" if "xgboost_calibrated" in self._models else "xgboost"

        artifact = self._models.get(mid, self._models.get("xgboost", {}))
        features = artifact.get("features", self._feature_names)
        categoricals = artifact.get("categorical_columns", self._categorical_columns)

        X = df.copy()

        # Drop non-feature columns
        for col in ["TARGET", "SK_ID_CURR", "application_date"]:
            if col in X.columns:
                X = X.drop(columns=[col])

        # Align training features
        if features:
            for f in features:
                if f not in X.columns:
                    X[f] = np.nan
            X = X[features]

        # Encode categoricals using consistent global training categories
        for col in categoricals:
            if col in X.columns and col in self._categorical_mappings:
                X[col] = X[col].map(self._categorical_mappings[col]).fillna(-1).astype(int)

        for col in X.columns:
            if X[col].dtype == "object" or str(X[col].dtype) == "category":
                if col in self._categorical_mappings:
                    X[col] = X[col].map(self._categorical_mappings[col]).fillna(-1).astype(int)
                else:
                    X[col] = X[col].astype("category").cat.codes

        # Replace infinities
        X = X.replace([np.inf, -np.inf], np.nan)

        return X

    # -------------------------------------------------------------------------
    # Prediction & Chaining
    # -------------------------------------------------------------------------

    def predict_single_model(self, applicant: pd.DataFrame, model_id: str) -> float:
        """Predict PD using a discrete single model, applying calibration if present."""
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' is not loaded.")

        artifact = self._models[model_id]
        model = artifact["model"]
        X = self.prepare_model_input(applicant, model_id)
        prob = model.predict_proba(X)[:, 1][0]

        # Post-hoc calibration if bundled in artifact
        if "calibrator" in artifact and artifact["calibrator"] is not None:
            cal = artifact["calibrator"]
            prob = float(cal.predict(np.array([prob]))[0])
            prob = max(0.0001, min(0.9999, prob))

        return float(prob)

    def predict_pd_detailed(
        self, applicant: pd.DataFrame, model_id: Optional[str] = None
    ) -> tuple[float, dict, str]:
        """
        Calculates Probability of Default, execution details, and effective model name.
        Handles both individual models and chained pipelines.
        """
        if not self._loaded:
            self.load()

        mid = model_id or self._active_model_id

        # 1. Chained Ensemble (Weighted Blend)
        if mid == "chained_ensemble":
            scores = {}
            weights = {}
            xgb_key = "xgboost_calibrated" if "xgboost_calibrated" in self._models else "xgboost"
            lgb_key = "lightgbm_calibrated" if "lightgbm_calibrated" in self._models else "lightgbm"

            if xgb_key in self._models:
                scores[xgb_key] = self.predict_single_model(applicant, xgb_key)
                weights[xgb_key] = 0.45
            if lgb_key in self._models:
                scores[lgb_key] = self.predict_single_model(applicant, lgb_key)
                weights[lgb_key] = 0.45
            if "logistic" in self._models:
                scores["logistic"] = self.predict_single_model(applicant, "logistic")
                weights["logistic"] = 0.10

            # Normalize weights if any model was missing
            total_weight = sum(weights.values())
            final_pd = sum(scores[m] * (w / total_weight) for m, w in weights.items())
            details = {
                "pipeline": "chained_ensemble",
                "sub_scores": scores,
                "weights": {m: round(w / total_weight, 3) for m, w in weights.items()},
            }
            return float(final_pd), details, "Chained Weighted Ensemble"

        # 2. Two-Stage Cascade Hurdle
        if mid == "cascade_hurdle":
            # Stage 1: Logistic Scorecard fast check
            stage1_pd = (
                self.predict_single_model(applicant, "logistic")
                if "logistic" in self._models
                else 0.10
            )

            # Clear cut low risk (< 3% default rate) or very high risk (> 35%)
            if stage1_pd < 0.03:
                details = {
                    "pipeline": "cascade_hurdle",
                    "stage_reached": 1,
                    "stage1_pd": stage1_pd,
                    "triage_verdict": "Fast Approved — Unambiguous Low Risk",
                }
                return float(stage1_pd), details, "Cascade Stage 1 (Logistic Triage)"
            elif stage1_pd > 0.35:
                details = {
                    "pipeline": "cascade_hurdle",
                    "stage_reached": 1,
                    "stage1_pd": stage1_pd,
                    "triage_verdict": "Fast Screened — Unambiguous High Risk",
                }
                return float(stage1_pd), details, "Cascade Stage 1 (Logistic Triage)"

            # Stage 2: Deep specialist evaluation (Calibrated Tree)
            specialist = (
                "xgboost_calibrated" if "xgboost_calibrated" in self._models
                else ("lightgbm_calibrated" if "lightgbm_calibrated" in self._models
                else ("xgboost" if "xgboost" in self._models else "lightgbm"))
            )
            stage2_pd = self.predict_single_model(applicant, specialist)
            details = {
                "pipeline": "cascade_hurdle",
                "stage_reached": 2,
                "stage1_pd": stage1_pd,
                "stage2_pd": stage2_pd,
                "specialist_model": specialist,
                "triage_verdict": "Stage 2 Deep Non-linear Assessment for Borderline Profile",
            }
            return float(stage2_pd), details, f"Cascade Stage 2 ({specialist.upper()})"

        # 3. Discrete Single Model
        if mid not in self._models:
            mid = "xgboost_calibrated" if "xgboost_calibrated" in self._models else "xgboost"

        pd_val = self.predict_single_model(applicant, mid)
        model_name = {
            "xgboost_calibrated": "XGBoost (Isotonic Calibrated)",
            "xgboost": "XGBoost Classifier (Raw)",
            "lightgbm_calibrated": "LightGBM (Isotonic Calibrated)",
            "lightgbm": "LightGBM Classifier (Raw)",
            "logistic": "Logistic Regression (Scorecard)",
        }.get(mid, mid)

        return float(pd_val), {"model_id": mid}, model_name

    def predict_pd(
        self, applicant: pd.DataFrame, model_id: Optional[str] = None
    ) -> float:
        """Backward-compatible single PD value return."""
        pd_val, _, _ = self.predict_pd_detailed(applicant, model_id)
        return pd_val

    # -------------------------------------------------------------------------
    # Model Metadata & Top Features
    # -------------------------------------------------------------------------

    def get_model_info(self, model_id: Optional[str] = None) -> dict:
        """Return model metadata, metrics, and feature importances for a model."""
        if not self._loaded:
            self.load()

        mid = model_id or self._active_model_id

        # If virtual chained pipeline
        if mid in ("chained_ensemble", "cascade_hurdle"):
            bench = self._benchmark_data.get(mid, {})
            # Use XGBoost's top features for visual explanation representation
            xgb_info = self.get_model_info("xgboost")
            is_active = (mid == self._active_model_id)
            return {
                "model_id": mid,
                "model_type": bench.get("model_type", "Virtual Chained Pipeline"),
                "model_class": "ChainedPipeline",
                "task": "Probability of Default (Binary Classification)",
                "training_dataset": "Home Credit Default Risk",
                "total_dataset_applicants": self.total_applicants,
                "feature_count": len(self._feature_names),
                "model_version": f"{mid}_v1",
                "model_file": "virtual_pipeline",
                "model_size_bytes": None,
                "is_active": is_active,
                "is_pipeline": True,
                "metrics": {
                    "roc_auc": bench.get("roc_auc", 0.7777),
                    "pr_auc": bench.get("pr_auc", 0.2699),
                    "brier_score": bench.get("brier_score", 0.1637),
                    "validation_applicants": 61503,
                },
                "top_features": xgb_info.get("top_features", []),
                "formula": bench.get("formula", ""),
                "shap_note": (
                    "Chained models synthesize predictions across multiple architectures. "
                    "Feature contributions are derived from the primary tree component."
                ),
            }

        # Discrete model
        if mid not in self._models:
            mid = "xgboost"

        artifact = self._models.get(mid, {})
        model_obj = artifact.get("model")
        feature_names = artifact.get("features", self._feature_names)
        file_path = settings.models_dir / f"pd_{mid}.joblib"
        stat = file_path.stat() if file_path.exists() else None

        # Metrics from benchmark cache
        metrics = {}
        if mid in self._benchmark_data:
            bench = self._benchmark_data[mid]
            metrics = {
                "roc_auc": bench.get("roc_auc"),
                "pr_auc": bench.get("pr_auc"),
                "brier_score": bench.get("brier_score"),
                "validation_applicants": 61503,
            }

        # Feature importances
        top_features = []
        try:
            if hasattr(model_obj, "feature_importances_"):
                importances = model_obj.feature_importances_
                feat_imp = sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]
                top_features = [
                    {"feature": f, "importance": round(float(i), 6)}
                    for f, i in feat_imp
                ]
            elif hasattr(model_obj, "named_steps"):
                # Logistic regression pipeline
                classifier = model_obj.named_steps.get("classifier")
                if hasattr(classifier, "coef_"):
                    importances = np.abs(classifier.coef_[0])
                    feat_imp = sorted(
                        zip(feature_names, importances),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:20]
                    top_features = [
                        {"feature": f, "importance": round(float(i), 6)}
                        for f, i in feat_imp
                    ]
        except Exception as e:
            logger.warning("Feature importance extraction failed for %s: %s", mid, e)

        return {
            "model_id": mid,
            "model_type": {
                "xgboost_calibrated": "XGBoost (Isotonic Calibrated)",
                "xgboost": "XGBoost Classifier (Raw)",
                "lightgbm_calibrated": "LightGBM (Isotonic Calibrated)",
                "lightgbm": "LightGBM Classifier (Raw)",
                "logistic": "Logistic Regression Scorecard",
            }.get(mid, mid.upper()),
            "model_class": type(model_obj).__name__,
            "task": "Probability of Default (Binary Classification)",
            "training_dataset": "Home Credit Default Risk",
            "total_dataset_applicants": self.total_applicants,
            "feature_count": len(feature_names),
            "model_version": f"pd_{mid}_v1",
            "model_file": file_path.name if file_path.exists() else f"pd_{mid}.joblib",
            "model_size_bytes": stat.st_size if stat else None,
            "is_active": (mid == self._active_model_id),
            "is_pipeline": False,
            "metrics": metrics if metrics else None,
            "top_features": top_features,
            "shap_note": (
                "SHAP values indicate feature contribution to model output. "
                "They should NOT be interpreted as causal explanations."
            ),
        }


# Module-level singleton
model_service = ModelService.get_instance()
