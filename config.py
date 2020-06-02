knn_classifier =  {
    "dataset": '../data/animals/*/*.jpg',
    "k": 1,
    "jobs": -1 # number of concurrent jobs to run when computing the distance. -1 will use all the available cores
}

linear_classifier = {
    "data_path": '../data/beagle.png',
    "K": 3, # number of classes or categories
    "N": 3072 # 1D size of the image
}

linear_classifier_with_GD = {
    "dataset": '../data/animals/*/*.jpg',
    "2_animal_dataset": '../data/2_animals/*/*.jpg',
    "epochs": 100,
    "alpha": 0.01 #learning rate
}

linear_classifier_with_SGD = {
    "dataset": '../data/animals/*/*.jpg',
    "2_animal_dataset": '../data/2_animals/*/*.jpg',
    "epochs": 100,
    "alpha": 0.01, #learning rate
    "batch_size": 32
}

regularization = {
    "dataset": "../data/animals/*/*.jpg",
    "alpha": 0.01
}

shallownet_animals = {
    "dataset_path": "../data/animals/*/*.jpg",
    "alpha": 0.05,
    "loss": "categorical_crossentropy",
    "batch_size": 32,
    "epochs": 100,
    "save_model_path": "../saved_models/shallownet_animals.hdf5"
}
