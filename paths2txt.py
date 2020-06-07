import os
import random

# root = 'data/Mussel/test_data'
# part_A_train = os.path.join(root, 'images/')
# paths = os.listdir(part_A_train)
# random.shuffle(paths)
# print(paths)
# print(len(paths))
# f = open("data/Mussel/paths_A/paths_test.txt",'a')
# for i in range(len(paths)):
#     f.write(os.path.join(part_A_train, paths[i]))
#     f.write("\n")

#root = 'data/Abalone/train_data'
#root = 'data/mussel300/train_data'
root = 'data/mussel300/test_data'
part_A_train = os.path.join(root, 'images/')
paths = os.listdir(part_A_train)
random.shuffle(paths)
print(paths)
print(len(paths))
#f = open("data/Abalone/paths_A/paths_train.txt",'a')
f = open("data/mussel300/paths_A/paths_test.txt",'a')
for i in range(len(paths)):
    f.write(os.path.join(part_A_train, paths[i]))
    f.write("\n")