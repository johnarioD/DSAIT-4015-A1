import numpy as np

def partitioning_test(model, dataset, feature, group_a, group_b):
    #check if the feature avaliable
    if feature not in dataset.columns:
        print("Feature not found")
        return False

    #partitioning
    dataset_a = dataset[dataset[feature] == group_a]
    dataset_b = dataset[dataset[feature] == group_b]

    #check if there is one group has no element
    if len(dataset_a) == 0 or len(dataset_b) == 0:
        print("The length of one group is 0")
        return False
    print(f"The length of group_a is {len(dataset_a)}")
    print(f"The length of group_b is {len(dataset_b)}")

    #prediction
    try:
        scores_a = model.predict(dataset_a)
        scores_b = model.predict(dataset_b)
    except Exception as e:
        print(f"Error happened when Predict: {e}")
        return False

    #calculate the mean of two score sets
    mean_a = np.round(np.mean(scores_a), decimals=2)
    mean_b = np.round(np.mean(scores_b), decimals=2)

    diff = np.round(mean_a - mean_b, decimals=2)

    print(f"The mean of group_a is {mean_a}")
    print(f"The mean of group_b is {mean_b}")
    print(f"The mean difference is {diff}")

    return mean_a, mean_b, diff
