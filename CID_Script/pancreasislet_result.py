import torch
import numpy as np
# Load the data with specific cell type
loaded_data_alpha = torch.load('dist_mat_vol_beta.pt')
matrices = [matrix for matrix, index, label in loaded_data_alpha]
# add all the elments in the matrices_with_specific_label
sum_matrix = torch.zeros_like(matrices[0])
for mat in matrices:
    sum_matrix += mat
sum_matrix = sum_matrix / len(matrices)
# Load the distance matrix from a .pt file
print(sum_matrix)
sum_matrix_np = sum_matrix.numpy()
row_sums = np.sum(sum_matrix_np, axis=1)
sorted_indices = np.argsort(row_sums)
gene_postions = torch.load('gene_positions.pt')
gene_names = list(gene_postions.keys())
# get the gene names in the order of the sorted indices
for i in sorted_indices:
    print(gene_names[i])
    