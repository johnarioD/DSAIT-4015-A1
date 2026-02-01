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
import os

from metrics import (
	STANDARD_PERFORMANCE_METRICS, FAIRNESS_METRICS, ROBUSTNESS_METRICS
)

from term_styling import style, fg, bg
VERBOSITY = 0

if not os.path.exists('results'):
	os.mkdir('results')
	os.mkdir('results/group1')
	os.mkdir('results/group2')

features, target, problem_cols = get_testing_data()

identify_outliers( features, features.columns, 2, VERBOSITY )
identify_outliers( features, features.columns, 3, VERBOSITY )
identify_outliers( features, features.columns, 4, VERBOSITY )

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
	verbosity=VERBOSITY
)

partition_results = partition_suite.run(models, titles, features, target)
partition_suite.save_json(partition_results, 'results/group2/results_partition.json')

shuffle_suite = MetamorphicSuite(
	shuffle_columns,
	"Shuffle",
	tries=5,
	problem_columns=problem_cols,
	classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=VERBOSITY
)

shuffle_results = shuffle_suite.run(models, titles, features, target)
shuffle_suite.save_json(shuffle_results, 'results/group2/results_shuffle.json')

flip_suite = MetamorphicSuite(
	flip_columns,
	"Flip",
	tries=1,
	problem_columns=problem_cols,
	classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=VERBOSITY
)

flip_results = flip_suite.run(models, titles, features, target)
flip_suite.save_json(flip_results, 'results/group2/results_flip.json')

noise_suite = MetamorphicSuite(
	add_noise_to_columns,
	"Noise",
	tries=5,
	problem_columns=problem_cols,
	classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=VERBOSITY,
	noise_scale=2.0
)

noise_results = noise_suite.run(models, titles, features, target)
noise_suite.save_json(noise_results, 'results/group2/results_noise.json')

scale_suite = MetamorphicSuite(
	scale_columns,
	"Scale",
	tries=1,
	problem_columns=problem_cols,
	classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=VERBOSITY,
	scale_factor=1.5
)

scale_results = scale_suite.run(models, titles, features, target)
scale_suite.save_json(scale_results, 'results/group2/results_scale.json')

shift_suite = MetamorphicSuite(
	shift_columns,
	"Shift",
	tries=5,
	problem_columns=problem_cols,
	classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=VERBOSITY
)

shift_results = shift_suite.run(models, titles, features, target)
shift_suite.save_json(shift_results, 'results/group2/results_shift.json')

quantization_suite = MetamorphicSuite(
	quantize_columns,
	"Quantization",
	tries=1,
	problem_columns=problem_cols,
	classical_metrics=classical_metrics,
	test_metrics=robustness_metrics,
	verbosity=VERBOSITY
)

quantization_results = quantization_suite.run(models, titles, features, target)
quantization_suite.save_json(quantization_results, 'results/group2/results_quantization.json')

consistency_suite = ConsistencySuite(
	n_trials=10,
	sample_size=100,
	consistency_threshold=1.0,
	verbosity=VERBOSITY
)

consistency_results = consistency_suite.run(models, titles, features, target)
consistency_suite.save_json(consistency_results, 'results/group2/results_consistency.json')

features_to_test = features.columns

boundary_suite = BoundarySuite(
	features_to_test=features_to_test,
	classical_metrics=classical_metrics,
	percentile_low=0.05,
	percentile_high=0.95,
	verbosity=VERBOSITY
)

boundary_results = boundary_suite.run(models, titles, features, target)
boundary_suite.save_json(boundary_results, 'results/group2/results_boundary.json')

monotonicity_specs = {
	'persoon_leeftijd_bij_onderzoek': 'none', # Age should not affect predictions (fairness)
	'persoon_geslacht_vrouw': 'none', # Gender should not affect predictions (fairness)
	'competentie_ethisch_en_integer_handelen': 'decreasing'
}

monotonicity_suite = MonotonicitySuite(
	monotonicity_specs=monotonicity_specs,
	violation_threshold=0.10,
	n_samples=100,
	verbosity=VERBOSITY
)

monotonicity_results = monotonicity_suite.run(models, titles, features, target)
monotonicity_suite.save_json(monotonicity_results, 'results/group2/results_monotonicity.json')

try:
	model1 = SklearnModel("models/model1_1.onnx")
	model2 = SklearnModel("models/model1_2.onnx")
	
	M1_TITLE = fg.cyan + "Model A" + style.reset
	M2_TITLE = fg.purple + "Model B" + style.reset
	
	g1_models = [model1, model2]
	g1_titles = [M1_TITLE, M2_TITLE]
	
	partition_suite.run(g1_models, g1_titles, features, target)
	partition_suite.save_json(partition_results, 'results/group1/results_partition.json')
	
	shuffle_suite.run(g1_models, g1_titles, features, target)
	shuffle_suite.save_json(shuffle_results, 'results/group1/results_shuffle.json')
	
	flip_suite.run(g1_models, g1_titles, features, target)
	flip_suite.save_json(flip_results, 'results/group1/results_flip.json')
	
	noise_suite.run(g1_models, g1_titles, features, target)
	noise_suite.save_json(noise_results, 'results/group1/results_noise.json')
	
	scale_results = scale_suite.run(g1_models, g1_titles, features, target)
	scale_suite.save_json(scale_results, 'results/group1/results_scale.json')
	
	shift_results = shift_suite.run(g1_models, g1_titles, features, target)
	shift_suite.save_json(shift_results, 'results/group1/results_shift.json')

	quantization_results = quantization_suite.run(g1_models, g1_titles, features, target)
	quantization_suite.save_json(quantization_results, 'results/group1/results_quantization.json')
	
	consistency_suite.run(g1_models, g1_titles, features, target)
	consistency_suite.save_json(consistency_results, 'results/group1/results_consistency.json')
	
	boundary_suite.run(g1_models, g1_titles, features, target)
	boundary_suite.save_json(boundary_results, 'results/group1/results_boundary.json')
	
	monotonicity_suite.run(g1_models, g1_titles, features, target)
	monotonicity_suite.save_json(monotonicity_results, 'results/group1/results_monotonicity.json')
	
except FileNotFoundError:
	print(f"{fg.yellow}External models not found, skipping black-box testing{fg.reset}\n")
