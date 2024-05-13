import torch
import numpy as np
import pandas as pd

# Path to the file
file_path = './driver/canonical_drivers.txt'

# Initialize an empty list to store gene names
driver_gene_database = []

# Open the file and read each line
with open(file_path, 'r') as file:
    for line in file:
        # Strip whitespace and add the gene name to the list
        driver_gene_database.append(line.strip())

file_path = './canonical_drivers.txt'

# Initialize an empty list to store gene names
driver_gene_breast_database = []

# Open the file and read each line
with open(file_path, 'r') as file:
    for line in file:
        # Strip whitespace and add the gene name to the list
        driver_gene_breast_database.append(line.strip())

# Load the list of tuples
loaded_gene_positions = torch.load('gene_positions_breast.pt')
indices = list(loaded_gene_positions.items())
loaded_data_with_labels = torch.load('./driver/dist_mat_vol_breast_test.pt')
matrices_with_specific_label = [matrix for matrix, index in loaded_data_with_labels]
# add all the elments in the matrices_with_specific_label
sum_matrix = torch.zeros_like(matrices_with_specific_label[0])

for mat in matrices_with_specific_label:
    sum_matrix += mat

sum_matrix = sum_matrix / len(matrices_with_specific_label)
sum_matrix_np = sum_matrix.numpy()
row_sums = np.sum(sum_matrix_np, axis=1)


sorted_indices = np.argsort(row_sums)
gene_postions = torch.load('gene_positions_breast_new.pt')
gene_names = list(gene_postions.keys())
found_genes = []
for i in range(len(sorted_indices)):
        found_genes.append(gene_names[sorted_indices[i]])


set1= set(found_genes)
print(set1)
set2 = set(driver_gene_database)
set4 = set(driver_gene_breast_database)
picked_genes = list(set1.intersection(set2))
print(len(picked_genes),len(set1),len(set2))
picked_driver_breast = list(set1.intersection(set4))
print(len(picked_driver_breast))
