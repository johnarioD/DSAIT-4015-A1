import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def get_testing_data():
	# Let's load the dataset
	data = pd.read_csv('data/investigation_train_large_checked.csv')

	# Let's specify the features and the target
	target = data['checked']
	features = data.drop(columns=[ 'checked', 'Ja', 'Nee' ])
	features = features.astype(np.float32)
	
	problem_cols = get_problematic_columns( features )
	problem_cols_full = []
	for problem in problem_cols:
		problem_cols_full += problem_cols[problem]

	partition_splits = {
            'psychological': 2,
            'medical': 2,
            'racial': 4,
            'subjective': 3,
            'relationship': 3,
            'irrelevant': 2
           }

	#####################
	# NOTE: partition spemantics
	# psychological - well, unwell
	# medical - well, unwell
	# racial -  Germanic language native, Romance native, PIE native, Non-PIE native
	# subjective -  Low, Mid, High opinion
	# gender - Male, Female
	# age - Young Adult, Adult, Senior
	# relationship - Small, average, large social circle/family
	# irrelevant - Only for sports hobbyists, yes/no.

	problem_cols['full'] = []
	for problem in partition_splits:
		for problem_col in problem_cols[problem]:
			problem_cols['full'].append( features.columns.get_loc( problem_col ) )
		col_names = problem_cols[problem]
		grouped_subset = pca_grouping( features, col_names )
		problem_cols[problem] = {
			'partitions': n_wise_partition( grouped_subset, partition_splits[problem] ),
			'names': col_names
		}
	
	col_names = problem_cols['age']
	problem_cols['full'].append( features.columns.get_loc(col_names[0]) )
	grouped_subset = features[col_names[0]]
	problem_cols['age'] = {
		'partitions': n_wise_partition( grouped_subset, 3, [ 0, 30, 60, 200 ] ),
		'names': col_names
	}
	col_names = problem_cols['gender']
	problem_cols['full'].append( features.columns.get_loc(col_names[0]) )
	grouped_subset = features[col_names[0]]
	problem_cols['gender'] = {
		'partitions': n_wise_partition( grouped_subset, 2 ),
		'names': col_names
	}

	return features, target, problem_cols

def n_wise_partition( feature, n_partitions=2, thresholds=None ):
    feature = feature.copy()
    partitions = []
    if thresholds is None:
        mn, mx = feature.min(), feature.max()
        step = (mx-mn)/n_partitions
        thresholds = [ i for i in np.arange( mn, mx + 0.1*step, step ) ]
    else:
        assert n_partitions+1 == len(thresholds)

    for i in range( len(thresholds)-2 ):
        idx = np.where( (feature >= thresholds[i]) & ( feature < thresholds[i+1]) )
        partitions.append( idx )
    partitions.append( np.where( feature >= thresholds[-2] ) )

    return partitions

def pca_grouping( data, column_set ):
    pca = PCA( n_components=1 )
    return pca.fit_transform( data[column_set] )

def get_problematic_columns( data ):
    psychological_features = []
    medical_features = [ 'belemmering_hist_verslavingsproblematiek' ]
    racial_features = ['ontheffing_reden_hist_sociale_gronden']
    subjective_features = [ 'competentie_ethisch_en_integer_handelen', 'competentie_gedrevenheid_en_ambitie_tonen', 'competentie_met_druk_en_tegenslag_omgaan', 'competentie_omgaan_met_verandering_en_aanpassen',
                            'persoonlijke_eigenschappen_uitstroom_verw_vlgs_km', 'persoonlijke_eigenschappen_uitstroom_verw_vlgs_klant', 'afspraak_aantal_woorden', 'afspraak_laatstejaar_aantal_woorden',
                            'competentie_other', 'competentie_overtuigen_en_beïnvloeden'
                          ]
    age_features = ['persoon_leeftijd_bij_onderzoek']
    gender_features = ['persoon_geslacht_vrouw']
    relationship_features = []
    irrelevant_features = [ 'persoonlijke_eigenschappen_hobbies_sport' ]

    for col in data.columns:
        if 'relatie' in col:
            relationship_features.append( col )
        elif 'persoonlijke' in col:
            if '_nl_' in col or 'taal' in col:
                racial_features.append(col)
            elif '_opm' in col:
                subjective_features.append(col)
        elif 'adres_recenst' in col or 'sociaal' in col or 'taal' in col:
            racial_features.append(col)
        elif 'medische' in col or 'lichamelijke' in col:
            medical_features.append(col)
        elif 'psychische' in col:
            psychological_features.append(col)

    return {
            'psychological': psychological_features,
            'medical': medical_features,
            'racial': racial_features,
            'subjective': subjective_features,
            'gender': gender_features,
            'relationship': relationship_features,
            'age': age_features,
            'irrelevant': irrelevant_features
           }
