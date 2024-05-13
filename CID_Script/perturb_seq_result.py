import torch
import numpy as np
import pandas as pd
import os
import scanpy as sc

# gene list
genes =['AURKA','AURKB','AURKC','CENPE','CEP55','CIT','CREB1','E2F4','ECT2','EGR1','ELF1','ELK1','GABPA','IRF1','NR2C2','OGG1','RACGAP1','TOR1AIP1','YY1']

# set the data directory
data_dir_1 = './perturb_data/result/result_new_2'
data_dir_2 = './perturb_data/result/result_new_2/'

results = []
gene_box_data = []

for gene in genes:
    print(gene)
    gene_prediction={}

    ref_file_path = os.path.join(data_dir_1, f'new_flag_mat_vol_perturb_ref_{gene}.pt')
    flag_file_path = os.path.join(data_dir_2, f'new_flag_mat_vol_perturb_{gene}.pt')
    gene_name_path = os.path.join(data_dir_2, f'flag_mat_vol_perturb_ref_genename_{gene}.pt')

    flag_with_labels_ref = np.array(torch.load(ref_file_path))
    gene_name = torch.load(gene_name_path)
    print(gene_name,len(gene_name))
    flag_ref_init = flag_with_labels_ref[0]
    flag_ref = flag_ref_init[:30]

    flag_with_labels = torch.load(flag_file_path)
    flag_with_specific_label = [matrix for matrix in flag_with_labels]
    sum_matrix_flag = torch.zeros_like(flag_with_specific_label[0])

    for mat in flag_with_specific_label:
        sum_matrix_flag += mat
    sum_matrix_flag = sum_matrix_flag / len(flag_with_specific_label)
    sum_matrix_flag = np.array(sum_matrix_flag)


    ko_data = sc.read_h5ad(f"./perturb_data/perturb_h5ad/perturb_{gene}.h5ad")
    ctrl_data = sc.read_h5ad(f"./perturb_data/perturb_h5ad/perturb_ctrl.h5ad")
    sc.pp.normalize_total(ko_data, target_sum=1e4)
    sc.pp.normalize_total(ctrl_data, target_sum=1e4)

    ko_data.obs['condition'] = 'KO'
    ctrl_data.obs['condition'] = 'Control'
    data = ko_data.concatenate(ctrl_data)

    sc.tl.rank_genes_groups(data, groupby='condition', method='wilcoxon', groups=['KO'], reference='Control')
    results = data.uns['rank_genes_groups']

    # extract the names, scores, p-values, and log fold changes of the differentially expressed genes
    gene_names = results['names']['KO']
    log_fold_changes = results['logfoldchanges']['KO']  # log2 fold change
    pvals_adj = results['pvals_adj']['KO']  # adjusted p-values

    diff_genes_df = pd.DataFrame({
        'gene': gene_names,
        'log_fold_change': log_fold_changes,
        'pval_adj': pvals_adj
    })
    for i in range(len(gene_name)):
        if sum_matrix_flag[i] >= 0.5:
            flag = 1
        elif sum_matrix_flag[i] >= 0:
            flag = 0
        else:
            flag = -1
        gene_prediction = {
            'gene': gene_name[i],
            'flag': flag,
            'log_fold_change': diff_genes_df.loc[diff_genes_df['gene'] == gene_name[i], 'log_fold_change'].iloc[0],
            'pval_adj': diff_genes_df['pval_adj'][i]
        }
        gene_box_data.append(gene_prediction)


df = pd.DataFrame(gene_box_data)
print(df)
# save the dataframe to a csv file
df.to_csv('perturb_result_all.csv', index=False)






