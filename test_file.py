import os

def test_metrics_exists():
    assert os.path.isfile("metrics.csv"), "metrics.csv file is missing!"

def test_model_exists():
    assert os.path.isfile("model.ckpt")==False, "model.ckpt must not be uploaded!"

def test_data_exists():
    assert os.path.isdir("data")==False, "data directory must not be uploaded!"

def test_val_acc():
    with open('metrics.csv') as file:
        props = dict(line.strip().split('=', 1) for line in file)
    assert int(props['ValAcc']) > 70, "Accuracy of Validation Set must be greater than 70%"

def test_cat_acc():
    with open('metrics.csv') as file:
        props = dict(line.strip().split('=', 1) for line in file)
    assert int(props['CatAcc']) > 70, "Accuracy of Cat Class must be greater than 70%"

def test_dog_acc():
    with open('metrics.csv') as file:
        props = dict(line.strip().split('=', 1) for line in file)
    assert int(props['DogAcc']) > 70, "Accuracy of Dog Class must be greater than 70%"
