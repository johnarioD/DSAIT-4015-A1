import re
import json
import logging
import numpy as np
import pandas as pd
import onnxruntime as ort
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



@dataclass
class TestResult:
    """Immutable data class to store the result of a single test case."""
    test_name: str
    test_type: str          
    feature: str
    passed: bool
    metric_value: float
    threshold: float
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': str(self.test_name),
            'test_type': str(self.test_type),
            'feature': str(self.feature),
            'passed': bool(self.passed),
            'metric_value': float(self.metric_value),
            'threshold': float(self.threshold),
            'message': str(self.message),
            'timestamp': self.timestamp
        }




class TestReport:
    """Aggregates results and generates summaries."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results: List[TestResult] = []

    def add_result(self, result: TestResult):
        self.results.append(result)

    def summary(self) -> Dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        sorted_results = sorted(self.results, key=lambda x: x.passed)

        return {
            "model_name": self.model_name,
            "pass_rate": round(passed / total, 2) if total > 0 else 0.0,
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "results": [r.to_dict() for r in sorted_results] 
        }
    
    def save_json(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.summary(), f, indent=4)









class ModelTester:
    def __init__(self, model_path: str, log_level: int = logging.INFO):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
             alt_path = Path("model") / model_path
             if alt_path.exists():
                 self.model_path = alt_path
             else:
                 raise FileNotFoundError(f"Model file not found at {self.model_path} or {alt_path}")

        self.logger = self._setup_logging(log_level)
        self.session = self._load_model()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.report = TestReport(self.model_path.name)

    def _setup_logging(self, level: int) -> logging.Logger:
        logger = logging.getLogger(f"Tester_{self.model_path.stem}")
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_model(self) -> ort.InferenceSession:
        return ort.InferenceSession(str(self.model_path))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        try:
            input_data = df.astype(np.float32).values
            res = self.session.run([self.output_name], {self.input_name: input_data})[0]
            if res.ndim > 1 and res.shape[1] > 1:
                return res[:, 1]
            return res.flatten()
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise







    def compute_adaptive_threshold(self, df: pd.DataFrame, y: pd.Series, features: List[str], method: str = "baseline_variance") -> float:
        preds = self.predict(df)
        
        if method == "baseline_variance":
            # Compute variance of PPR across random partitions as baseline noise
            n_bootstrap = 50
            ppr_samples = []
            n = len(df)
            
            for _ in range(n_bootstrap):
                idx = np.random.choice(n, size=n//2, replace=False)
                ppr_samples.append(preds[idx].mean())
            
            # Threshold = 2 * std of random partition PPR (95% confidence)
            baseline_std = np.std(ppr_samples)
            threshold = 5.0 * baseline_std
            self.logger.info(f"[Threshold] baseline_variance method: {threshold:.4f}")
            return max(threshold, 0.02)  # minimum floor
        
        elif method == "permutation_baseline":
            # Run permutation tests on non-sensitive features to establish baseline
            n_permutations = 20
            diffs = []
            
            non_sensitive = [f for f in features if 'wijk' not in f and 'buurt' not in f 
                           and 'geslacht' not in f and 'inburger' not in f][:5]
            
            for feat in non_sensitive:
                if feat not in df.columns:
                    continue
                for _ in range(n_permutations // len(non_sensitive)):
                    df_perm = df.copy()
                    df_perm[feat] = np.random.permutation(df[feat].values)
                    preds_perm = self.predict(df_perm)
                    diffs.append(np.abs(preds.mean() - preds_perm.mean()))
            
            if len(diffs) > 0:
                threshold = np.percentile(diffs, 95)
                self.logger.info(f"[Threshold] permutation_baseline method: {threshold:.4f}")
                return max(threshold, 0.02)
            return 0.05
        
        elif method == "percentile":
            y_arr = y.values if hasattr(y, 'values') else y        
            fpr_0 = preds[y_arr == 0].mean()
            fpr_1 = preds[y_arr == 1].mean()
            baseline_diff = abs(fpr_1 - fpr_0)
            threshold = 0.75 * baseline_diff
            self.logger.info(f"[Threshold] percentile method: {threshold:.4f}")
            return max(threshold, 0.02)
        
        return 0.05







    def n_wise_partition(self, feature: pd.Series, n_partitions: int = 2, 
                         thresholds: List[float] = None) -> List[pd.Index]:
        """
        Partition a feature into n groups based on value thresholds.
        Returns list of index arrays for each partition.
        """
        feature = feature.copy()
        partitions = []
        
        if thresholds is None:
            mn = feature.min()
            mx = feature.max()
            if mn == mx:
                return [feature.index]
            step = (mx - mn) / n_partitions
            thresholds = [mn + step * i for i in range(1, n_partitions)]
        else:
            assert n_partitions == len(thresholds) + 1
        
        remaining = feature.copy()
        for i in range(n_partitions - 1):
            partition_idx = remaining.index[remaining <= thresholds[i]]
            remaining = remaining.drop(index=partition_idx)
            partitions.append(partition_idx)
        partitions.append(remaining.index)
        
        return partitions

    def smart_partition(self, df: pd.DataFrame, feature: str) -> Tuple[List[pd.Index], str]:
        if feature not in df.columns:
            return [], "Feature not found"
        
        col = df[feature]
        unique_vals = col.nunique()
        min_val = col.min()
        max_val = col.max()
        data_range = max_val - min_val
        
        # Case 1: Binary feature (0/1)
        if unique_vals == 2 and set(col.unique()).issubset({0, 1, 0.0, 1.0}):
            idx_0 = col.index[col == 0]
            idx_1 = col.index[col == 1]
            return [idx_0, idx_1], "Binary (0 vs 1)"
        
        # Case 2: Small categorical (<=5 unique values)
        if unique_vals <= 5:
            partitions = [col.index[col == v] for v in sorted(col.unique())]
            return partitions, f"Categorical ({unique_vals} groups)"
        
        # Case 3: Age-like continuous (typically 18-100 range)
        if 'leeftijd' in feature.lower() or 'age' in feature.lower():
            bins = [0, 30, 45, 60, 120]
            labels = ['<=30', '31-45', '46-60', '>=61']
            age_bucket = pd.cut(col, bins=bins, labels=labels, include_lowest=True)
            partitions = [col.index[age_bucket == label] for label in labels]
            partitions = [p for p in partitions if len(p) >= 20]
            return partitions, f"Age buckets ({len(partitions)} groups)"
        
        # Case 4: Count-like feature (small integer range)
        if data_range <= 20 and unique_vals <= 10:
            median = col.median()
            idx_low = col.index[col <= median]
            idx_high = col.index[col > median]
            return [idx_low, idx_high], f"Binary split at median={median:.0f}"
        
        # Case 5: Continuous with moderate range - use quantile-based
        if data_range <= 100:
            try:
                col_binned = pd.qcut(col, q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
                partitions = [col.index[col_binned == label] for label in ['Low', 'Mid', 'High']]
                partitions = [p for p in partitions if len(p) >= 20]
                return partitions, "Tertile split (quantile-based)"
            except Exception:
                pass
        
        # Case 6: Large range continuous - use percentile-based
        try:
            p25, p75 = col.quantile([0.25, 0.75])
            idx_low = col.index[col <= p25]
            idx_mid = col.index[(col > p25) & (col <= p75)]
            idx_high = col.index[col > p75]
            partitions = [idx_low, idx_mid, idx_high]
            partitions = [p for p in partitions if len(p) >= 20]
            return partitions, f"Percentile split (25th/75th)"
        except Exception:
            pass
        
        # Fallback: simple median split
        median = col.median()
        idx_low = col.index[col <= median]
        idx_high = col.index[col > median]
        return [idx_low, idx_high], f"Fallback median split at {median:.2f}"









    def run_partitioning_tests(self, df: pd.DataFrame, y: pd.Series, 
                               features: List[str], threshold: float = None):
        """
        Combined partitioning tests:
        1. Demographic Parity: Compare PPR across partitions
        2. Equalized Odds: Compare TPR/FPR across partitions (using ground truth y)
        """
        self.logger.info(f"[Partitioning] Running tests on {len(features)} features...")
        
        # Compute adaptive threshold if not provided
        if threshold is None:
            threshold = self.compute_adaptive_threshold(df, y, features)
        
        preds = self.predict(df)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            partitions, partition_desc = self.smart_partition(df, feature)
            
            if len(partitions) < 2:
                self.report.add_result(TestResult(
                    test_name=f"Partition_{feature}",
                    test_type="Partitioning",
                    feature=feature,
                    passed=True,
                    metric_value=0.0,
                    threshold=threshold,
                    message="Skipped: insufficient partitions"
                ))
                continue
            
            valid_partitions = [p for p in partitions if len(p) >= 20]
            if len(valid_partitions) < 2:
                continue
            

            # PPR difference
            pprs = []
            for idx in valid_partitions:
                preds_part = preds[df.index.get_indexer(idx)]
                pprs.append(preds_part.mean())
            
            ppr_diff = max(pprs) - min(pprs)
            ppr_passed = ppr_diff < threshold
            
            self.report.add_result(TestResult(
                test_name=f"DemographicParity_{feature}",
                test_type="Partitioning_PPR",
                feature=feature,
                passed=ppr_passed,
                metric_value=float(ppr_diff),
                threshold=threshold,
                message=f"PPR diff: {ppr_diff:.4f} | {partition_desc}"
            ))
            

            # FPR
            fprs = []
            tprs = []
            for idx in valid_partitions:
                indexer = df.index.get_indexer(idx)
                y_part = y_arr[indexer]
                preds_part = preds[indexer]
                
                # FPR: P(pred=1 | y=0)
                neg_mask = (y_part == 0)
                if neg_mask.sum() >= 10:
                    fpr = preds_part[neg_mask].mean()
                    fprs.append(fpr)
                
                # TPR: P(pred=1 | y=1)
                pos_mask = (y_part == 1)
                if pos_mask.sum() >= 10:
                    tpr = preds_part[pos_mask].mean()
                    tprs.append(tpr)
            
            if len(fprs) >= 2:
                fpr_diff = max(fprs) - min(fprs)
                fpr_passed = fpr_diff < threshold
                
                self.report.add_result(TestResult(
                    test_name=f"EqualizedOdds_FPR_{feature}",
                    test_type="Partitioning_FPR",
                    feature=feature,
                    passed=fpr_passed,
                    metric_value=float(fpr_diff),
                    threshold=threshold,
                    message=f"FPR diff: {fpr_diff:.4f} | {partition_desc}"
                ))
            
            if len(tprs) >= 2:
                tpr_diff = max(tprs) - min(tprs)
                tpr_passed = tpr_diff < threshold
                
                self.report.add_result(TestResult(
                    test_name=f"EqualizedOdds_TPR_{feature}",
                    test_type="Partitioning_TPR",
                    feature=feature,
                    passed=tpr_passed,
                    metric_value=float(tpr_diff),
                    threshold=threshold,
                    message=f"TPR diff: {tpr_diff:.4f} | {partition_desc}"
                ))







    
    def run_metamorphic_tests(self, df: pd.DataFrame, features: List[str], threshold: float = 0.05):
        """
        Metamorphic tests: shuffling sensitive features should not significantly change predictions.
        """
        self.logger.info(f"[Metamorphic] Running tests on {len(features)} features...")
        
        original_preds = self.predict(df)
        
        for i, feature in enumerate(features):
            if feature not in df.columns:
                continue

            df_mutated = df.copy()
            df_mutated[feature] = np.random.permutation(df[feature].values)
            
            mutated_preds = self.predict(df_mutated)
            mean_diff = np.mean(np.abs(original_preds - mutated_preds))
            
            self.report.add_result(TestResult(
                test_name=f"Invariance_{feature}",
                test_type="Metamorphic",
                feature=feature,
                passed=mean_diff < threshold,
                metric_value=float(mean_diff),
                threshold=threshold,
                message=f"Mean prediction shift: {mean_diff:.4f}"
            ))








    def visualize_partition_results(self, df: pd.DataFrame, y: pd.Series, 
                                    sensitive_features: List[str], 
                                    save_path: str = None):
        """
        Visualize partitioning test results for sensitive features.
        """
        preds = self.predict(df)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        
        n_features = len(sensitive_features)
        fig, axes = plt.subplots(2, min(n_features, 3), figsize=(5*min(n_features, 3), 8))
        if n_features == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Fairness Analysis: {self.model_path.name}', fontsize=14, fontweight='bold')
        
        for i, feature in enumerate(sensitive_features[:3]):
            if feature not in df.columns:
                continue
            
            partitions, desc = self.smart_partition(df, feature)
            if len(partitions) < 2:
                continue
            
            # PPR comparison
            ax = axes[0, i] if n_features > 1 else axes[0, 0]
            pprs = []
            labels = []
            for j, idx in enumerate(partitions):
                if len(idx) < 20:
                    continue
                preds_part = preds[df.index.get_indexer(idx)]
                pprs.append(preds_part.mean())
                labels.append(f"Group {j}\n(n={len(idx)})")
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(pprs)]
            bars = ax.bar(labels, pprs, color=colors, alpha=0.8, edgecolor='black')
            ax.set_ylabel('Positive Prediction Rate')
            ax.set_title(f'{feature}\n(PPR by partition)')
            ax.set_ylim([0, max(pprs) * 1.3 if pprs else 1])
            
            for bar, val in zip(bars, pprs):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                       f'{val:.3f}', ha='center', fontsize=9)
            
            # FPR comparison
            ax = axes[1, i] if n_features > 1 else axes[1, 0]
            fprs = []
            labels = []
            for j, idx in enumerate(partitions):
                if len(idx) < 20:
                    continue
                indexer = df.index.get_indexer(idx)
                y_part = y_arr[indexer]
                preds_part = preds[indexer]
                
                neg_mask = (y_part == 0)
                if neg_mask.sum() >= 10:
                    fpr = preds_part[neg_mask].mean()
                    fprs.append(fpr)
                    labels.append(f"Group {j}")
            
            if fprs:
                bars = ax.bar(labels, fprs, color=colors[:len(fprs)], alpha=0.8, edgecolor='black')
                ax.set_ylabel('False Positive Rate')
                ax.set_title(f'{feature}\n(FPR by partition)')
                ax.set_ylim([0, max(fprs) * 1.3 if fprs else 1])
                
                for bar, val in zip(bars, fprs):
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                           f'{val:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
        return fig







if __name__ == "__main__":
    MODELS_TO_TEST = ["model/bad_model.onnx", "model/good_model.onnx"]
    DATA_PATH = "data/synth_data_for_training.csv"
    TARGET_COLUMN = 'checked'
    SENSITIVE_FEATURES = [
        "typering_hist_inburgeringsbehoeftig",
        "persoonlijke_eigenschappen_spreektaal_anders",
        "persoon_geslacht_vrouw",
        "adres_recentste_wijk_charlois",
        "adres_recentste_wijk_feijenoord",
        "adres_recentste_plaats_other",
        "persoon_leeftijd_bij_onderzoek",
    ]

    if Path(DATA_PATH).exists():
        print(f"Loading data from {DATA_PATH}...")
        df_raw = pd.read_csv(DATA_PATH)
        
        if TARGET_COLUMN in df_raw.columns:
            y = df_raw[TARGET_COLUMN].astype(int)
            df = df_raw.drop(columns=[TARGET_COLUMN])
        else:
            print(f"Warning: Target column '{TARGET_COLUMN}' not found.")
            y = pd.Series(np.zeros(len(df_raw)))
            df = df_raw
        
        df = df.select_dtypes(include=[np.number])
        all_features_to_test = df.columns.tolist()
        
        print(f"Data loaded for testing: {df.shape} samples.")
        print(f"Testing ALL {len(all_features_to_test)} numeric features.")
    else:
        print(f"Error: Data file '{DATA_PATH}' not found.")
        exit(1)


    for model_path in MODELS_TO_TEST:
        if not Path(model_path).exists():
            print(f"Warning: {model_path} not found. Skipping.")
            continue
            
        print("\n" + "="*60)
        print(f"STARTING FULL AUDIT: {model_path}")
        print("="*60)
        tester = ModelTester(model_path)

        # TODO refine the metamorphic thresold
        tester.run_partitioning_tests(df, y, features=all_features_to_test, threshold=None)
        tester.run_metamorphic_tests(df, features=all_features_to_test, threshold=0.05)

        sensitive_in_data = [f for f in SENSITIVE_FEATURES if f in df.columns]
        if sensitive_in_data:
            tester.visualize_partition_results(
                df, y, sensitive_in_data[:3],
                save_path=f"fairness_viz_{Path(model_path).stem}.png"
            )

        report_name = f"audit_report_{Path(model_path).stem}.json"
        tester.report.save_json(report_name)
        
        summary = tester.report.summary()
        print(f"\nFull Audit Complete for {model_path}.")
        print(f"Detailed report saved to {report_name}")
        print("-" * 30)
        print(f"Total Tests:  {summary['total_tests']}")
        print(f"Passed Tests: {summary['passed_tests']}")
        print(f"Failed Tests: {summary['failed_tests']}")
        print(f"Pass Rate:    {summary['pass_rate']:.1%}")
        print("-" * 30)