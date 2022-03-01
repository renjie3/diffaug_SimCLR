import numpy as np
import pickle
import os

train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]

data = []
targets = []

sampled_class = {0:0,1:1,3:2,7:3}

for file_name, checksum in train_list:
    file_path = os.path.join('./data/cifar-10-batches-py/', file_name)
    with open(file_path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
        data.append(entry["data"])
        if "labels" in entry:
            targets.extend(entry["labels"])
            print("check labels")
        else:
            targets.extend(entry["fine_labels"])
            print("check fine_labels")

# print(len(data))
data = np.vstack(data).reshape(-1, 3, 32, 32)
# print(data.shape)
data = data.transpose((0, 2, 3, 1))

# print(len(targets))
sampled_data = []
sampled_target = []
classes_count = [0 for _ in range(10)]

# print(sampled_target.shape)
for i in range(40000,50000, 3):
    if classes_count[targets[i]] < 256 and targets[i] in sampled_class:
        classes_count[targets[i]] += 1
        sampled_target.append(sampled_class[targets[i]])
        sampled_data.append(data[i])
        # print(data[i].shape)
print(classes_count)
sampled_data = np.stack(sampled_data, axis=0)
print(sampled_data.shape)
print("len(sampled_target)", len(sampled_target))

test_data = []
test_targets = []
for file_name, checksum in test_list:
    file_path = os.path.join('./data/cifar-10-batches-py/', file_name)
    with open(file_path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
        test_data.append(entry["data"])
        if "labels" in entry:
            test_targets.extend(entry["labels"])
            print("check labels")
        else:
            test_targets.extend(entry["fine_labels"])
            print("check fine_labels")

test_data = np.vstack(test_data).reshape(-1, 3, 32, 32)
# print(data.shape)
test_data = test_data.transpose((0, 2, 3, 1))

sampled_target_test = []
sampled_data_test = []

# print(sampled_target.shape)
for i in range(len(test_targets)):
    if test_targets[i] in sampled_class:
        sampled_target_test.append(sampled_class[test_targets[i]])
        sampled_data_test.append(test_data[i])

sampled_data_test = np.stack(sampled_data_test, axis=0)
print(sampled_data_test.shape)
print("len(sampled_target_test)", len(sampled_target_test))

sampled = {}
sampled["train_data"] = sampled_data
sampled["train_targets"] = sampled_target
sampled["test_data"] = sampled_data_test
sampled["test_targets"] = sampled_target_test

file_path = './data/sampled_cifar10/cifar10_1024_4class16.pkl'
with open(file_path, "wb") as f:
    entry = pickle.dump(sampled, f)
