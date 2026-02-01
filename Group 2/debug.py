from data_handling import get_testing_data
from outlier_detection import identify_outliers
from prediction_models import get_trained_models, SklearnModel

from metamorphic_suite import (
	MetamorphicSuite, shuffle_columns, flip_columns, add_noise_to_columns,
	scale_columns, shift_columns, permute_within_quantiles, quantize_columns
)
from partition_suite import PartitionSuite
from boundary_suite import BoundarySuite
from consistency_suite import ConsistencySuite
from monotonicity_suite import MonotonicitySuite

from metrics import (
	STANDARD_PERFORMANCE_METRICS, FAIRNESS_METRICS, ROBUSTNESS_METRICS
)

from term_styling import style, fg, bg

features, target, problem_cols = get_testing_data()

identify_outliers( features, features.columns, 2, 1 )
identify_outliers( features, features.columns, 3, 1 )
identify_outliers( features, features.columns, 4, 1 )

GOOD_TITLE = fg.green + "Good Model" + fg.reset
BAD_TITLE = fg.red + "Bad Model" + fg.reset

models = get_trained_models(features, target, problem_cols['full'])
titles = [GOOD_TITLE, BAD_TITLE]

classical_metrics = STANDARD_PERFORMANCE_METRICS
fairness_metrics = FAIRNESS_METRICS
robustness_metrics = ROBUSTNESS_METRICS

partition_suite = PartitionSuite(
	problem_cols,
	classical_metrics=classical_metrics,
	test_metrics=fairness_metrics,
	verbosity=1
)

partition_results = partition_suite.run(models, titles, features, target)

shuffle_suite = MetamorphicSuite(
	shuffle_columns,
	"Shuffle",
	tries=5,
	problem_columns=problem_cols,
	classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=1
)

shuffle_results = shuffle_suite.run(models, titles, features, target)

flip_suite = MetamorphicSuite(
	flip_columns,
	"Flip",
	tries=1,
	problem_columns=problem_cols,
	classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=1
)

flip_results = flip_suite.run(models, titles, features, target)

noise_suite = MetamorphicSuite(
	add_noise_to_columns,
	"Noise",
	tries=5,
	problem_columns=problem_cols,
	classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=1,
	noise_scale=2.0
)

noise_results = noise_suite.run(models, titles, features, target)

scale_suite = MetamorphicSuite(
	scale_columns,
	"Scale",
	tries=1,
	problem_columns=problem_cols,
    classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=1,
	scale_factor=1.5
)

scale_results = scale_suite.run(models, titles, features, target)

shift_suite = MetamorphicSuite(
	shift_columns,
	"Scale",
	tries=5,
	problem_columns=problem_cols,
    classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=1
)

shift_results = shift_suite

permute_suite = MetamorphicSuite(
	permute_within_quantiles,
	"Scale",
	tries=1,
	problem_columns=problem_cols,
    classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=1
)

permutation_results = permute_suite.run(models, titles, features, target).run(models, titles, features, target)

consistency_suite = ConsistencySuite(
	n_trials=10,
	sample_size=100,
	consistency_threshold=1.0,
	verbosity=1
)

consistency_results = consistency_suite.run(models, titles, features, target)

features_to_test = features.columns

boundary_suite = BoundarySuite(
	features_to_test=features_to_test,
    classical_metrics=classical_metrics,
	percentile_low=0.05,
	percentile_high=0.95,
	verbosity=1
)

boundary_results = boundary_suite.run(models, titles, features, target)

monotonicity_specs = {
	'persoon_leeftijd_bij_onderzoek': 'none', # Age should not affect predictions (fairness)
	'persoon_geslacht_vrouw': 'none', # Gender should not affect predictions (fairness)
	'competentie_ethisch_en_integer_handelen': 'increasing'
}

monotonicity_suite = MonotonicitySuite(
	monotonicity_specs=monotonicity_specs,
	violation_threshold=0.10,
	n_samples=100,
	verbosity=1
)

monotonicity_results = monotonicity_suite.run(models, titles, features, target)

try:
	model1 = SklearnModel("models/model1_1.onnx")
	model2 = SklearnModel("models/model1_2.onnx")
	
	M1_TITLE = fg.cyan + "Model A" + style.reset
	M2_TITLE = fg.purple + "Model B" + style.reset
	
	external_models = [model1, model2]
	external_titles = [M1_TITLE, M2_TITLE]
	
	partition_suite.run(external_models, external_titles, features, target)
	
	shuffle_suite.run(external_models, external_titles, features, target)
	
	flip_suite.run(external_models, external_titles, features, target)
	
	noise_suite.run(external_models, external_titles, features, target)
	
	consistency_suite.run(external_models, external_titles, features, target)
	
	boundary_suite.run(external_models, external_titles, features, target)
	
	monotonicity_suite.run(external_models, external_titles, features, target)
	
except FileNotFoundError:
	print(f"{fg.yellow}External models not found, skipping black-box testing{fg.reset}\n")
