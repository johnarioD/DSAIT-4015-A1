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
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)

    diff = mean_a - mean_b

    print(f"The mean of group_a is {mean_a}")
    print(f"The mean of group_b is {mean_b}")
    print(f"The mean difference is {diff}")

    if abs(diff) > 0.2:     #set the threshold at 0.2
        print(f"The difference between two groups is too high, therefore, feature {feature} has a high bias")
        return False
    else:
        print("PASS THE TEST!!!")
        return True

