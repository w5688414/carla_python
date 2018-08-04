import os

list_folder = '/home/kadn/VOCdevkit/list'
dataset_splits = os.path.join(list_folder, '*.txt')
for dataset_split in dataset_splits:
    dataset = os.path.basename(dataset_split)[:-4]

