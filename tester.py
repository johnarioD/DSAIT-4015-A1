import re
import json
import logging
import numpy as np
import pandas as pd
import onnxruntime as ort
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime



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




    def run_partitioning_tests(self, df: pd.DataFrame, features: List[str], threshold: float = 0.15):
        self.logger.info(f"[Partitioning] Running tests on {len(features)} features...")
        
        preds = self.predict(df)
        temp_df = df.copy()
        temp_df['_risk'] = preds
        
        for feature in features:
            if feature not in df.columns:
                continue

            if temp_df[feature].nunique() <= 5:
                means = temp_df.groupby(feature)['_risk'].mean()
                diff = means.max() - means.min()
                msg = f"Max diff between categories: {diff:.4f}"
            else:
                median = temp_df[feature].median()
                g1 = temp_df[temp_df[feature] <= median]['_risk'].mean()
                g2 = temp_df[temp_df[feature] > median]['_risk'].mean()
                diff = abs(g1 - g2)
                msg = f"Diff (Low vs High split at {median}): {diff:.4f}"

            self.report.add_result(TestResult(
                test_name=f"Partition_{feature}",
                test_type="Partitioning",
                feature=feature,
                passed=diff < threshold,
                metric_value=float(diff),
                threshold=threshold,
                message=msg
            ))

    def run_metamorphic_tests(self, df: pd.DataFrame, features: List[str], threshold: float = 0.05):
        self.logger.info(f"[Metamorphic] Running tests on {len(features)} features...")
        
        original_preds = self.predict(df)
        
        for i, feature in enumerate(features):
            if feature not in df.columns:
                continue
            
            if (i + 1) % 50 == 0:
                self.logger.info(f"Processing feature {i+1}/{len(features)}...")

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

    def run_monotonicity_tests(self, df: pd.DataFrame, features: List[str], threshold: float = 0.2):
        self.logger.info(f"[Monotonicity] Running tests on {len(features)} features...")
        
        preds = self.predict(df)
        
        for feature in features:
            if feature not in df.columns:
                continue
            

            if df[feature].nunique() <= 1:
                corr = 0.0
            else:
                corr = np.corrcoef(df[feature], preds)[0, 1]
                if np.isnan(corr): corr = 0.0

            passed = abs(corr) < threshold
            
            self.report.add_result(TestResult(
                test_name=f"Monotonicity_{feature}",
                test_type="Monotonicity",
                feature=feature,
                passed=passed,
                metric_value=float(corr),
                threshold=threshold,
                message=f"Correlation: {corr:.4f}"
            ))



if __name__ == "__main__":
    MODELS_TO_TEST = ["model/gboost.onnx", "model/model_2.onnx"]
    DATA_PATH = "data/synth_data_for_training.csv"
    TARGET_COLUMNS_TO_DROP = ['checked', 'fraude_risico'] 
    

    if Path(DATA_PATH).exists():
        print(f"Loading data from {DATA_PATH}...")
        df_raw = pd.read_csv(DATA_PATH)
        
        for col in TARGET_COLUMNS_TO_DROP:
            if col in df_raw.columns:
                print(f"Dropping target column: {col}")
                df_raw = df_raw.drop(columns=[col])
        
        df = df_raw.select_dtypes(include=[np.number])
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

        try:
            tester = ModelTester(model_path)
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
        

        tester.run_partitioning_tests(df, features=all_features_to_test, threshold=0.05)
        tester.run_metamorphic_tests(df, features=all_features_to_test, threshold=0.05)
        tester.run_monotonicity_tests(df, features=all_features_to_test, threshold=0.05)



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