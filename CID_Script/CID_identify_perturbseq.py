from performer_pytorch import PerformerLM
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import sys
from utils import *
import gc
import pandas as pd
import argparse
# Define the model parameters
SEQ_LEN =  16907 # args.gene_num + 1
POS_EMBED_USING = True  # Whether to use position embedding
MASK_TOKEN_ID= 5
CLASS = 7
# Initialize the PerformerLM model

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,      # Dimensionality of feature vectors
    depth = 6,      # Number of layers
    max_seq_len = SEQ_LEN,
    heads = 10,     # Number of attention heads
    local_attn_heads = 0,  # Number of local attention heads
    g2v_position_emb = POS_EMBED_USING
)



class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
weights = np.load('../data/gene2vec_16906.npy', allow_pickle=True)

if isinstance(weights, dict):
    for name, param in model.named_parameters():
        if name in weights:
            param.data = torch.Tensor(weights[name])



def apply_mask_pattern_for_pairs(token_ids, MASK_TOKEN_ID, indices):

    print("the length of indice:",len(indices))
    non_zero_indices = torch.tensor(indices, dtype=torch.long)
    if non_zero_indices.dim() == 0:
        non_zero_indices = non_zero_indices.unsqueeze(0)
    length = len(non_zero_indices)
    num_sequences = int((length+1)*length/2)
    batch_token_ids = token_ids.repeat(num_sequences, 1)

    for i in range(length):
        batch_token_ids[i, non_zero_indices[i]] = MASK_TOKEN_ID
        if i != (length - 1):
            for j in range(i+1, length):
                    index = (i+1)*length - (i+1)*i//2 + j-i-1
                    batch_token_ids[index, non_zero_indices[j]] = MASK_TOKEN_ID
                    batch_token_ids[index, non_zero_indices[i]] = MASK_TOKEN_ID

    return batch_token_ids, non_zero_indices

def apply_mask_pattern_for_pairs_perturb(token_ids, MASK_TOKEN_ID, indices):

    print("the length of indice:",len(indices))

    non_zero_indices = torch.tensor(indices, dtype=torch.long)

    #non_zero_indices = torch.nonzero(token_ids.squeeze() > 4, as_tuple=False).squeeze()

    if non_zero_indices.dim() == 0:
        non_zero_indices = non_zero_indices.unsqueeze(0)
    length = len(non_zero_indices)
    num_sequences = int((length-1)*2)
    batch_token_ids = token_ids.repeat(num_sequences, 1)

    for i in range(length-1):
        batch_token_ids[i, non_zero_indices[i]] = MASK_TOKEN_ID
        batch_token_ids[i+length-1, non_zero_indices[i]] = MASK_TOKEN_ID
        batch_token_ids[i+length-1, non_zero_indices[length -1]] = MASK_TOKEN_ID

    return batch_token_ids, non_zero_indices

def apply_mask_pattern_for_one_gene(token_ids, MASK_TOKEN_ID, ko_gene):
    # read gene pos dict from a pt file
    #loaded_gene_positions = torch.load('../data/gene_positions_breast.pt')
    loaded_gene_positions={'ECT2':4225,'CIT':2769}
    indices = [loaded_gene_positions[ko_gene]]
    non_zero_indices = torch.tensor(indices, dtype=torch.long)
    #non_zero_indices = torch.nonzero(token_ids.squeeze() > 4, as_tuple=False).squeeze()

    if non_zero_indices.dim() == 0:
        non_zero_indices = non_zero_indices.unsqueeze(0)
    length = len(non_zero_indices)
    num_sequences = int((length+1)*length/2)
    batch_token_ids = token_ids.repeat(num_sequences, 1)

    for i in range(length):
        batch_token_ids[i, non_zero_indices[i]] = MASK_TOKEN_ID

    return batch_token_ids, non_zero_indices



class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        return full_seq.to(device)


    def __len__(self):
        return self.data.shape[0]

def calculate_euclidean_distance(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2, p=2, dim=-1)


def batch_inference(model, batch_data, batch_size):
    num_samples = batch_data.size(0)
    results = []

    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Inference Progress"):
        end_idx = min(start_idx + batch_size, num_samples)
        mini_batch = batch_data[start_idx:end_idx].to(device)
        
        with torch.no_grad():
            batch_size = mini_batch.shape[0]
            for index in range(batch_size):
                full_seq = mini_batch[index]
                full_seq = torch.where(full_seq > (CLASS - 2), torch.tensor(CLASS - 2, device=full_seq.device), full_seq)
                full_seq = torch.cat((full_seq, torch.tensor([0], device=full_seq.device)))
                full_seq = full_seq.unsqueeze(0)
                result = model(full_seq)
                results.append(result)  # Move results to CPU if not needed on GPU immediately
        
        # Consider commenting out the following lines to test without them first
        # torch.cuda.empty_cache()  # Clear unused memory from CUDA (use cautiously)
        # gc.collect()  # Explicitly invokes garbage collection (use cautiously)

    total_results = torch.cat(results, dim=0)
    return total_results

def cal_mse_tensors(pred1, pred2, diff_genes_indice):
    sum_mse=0
    for index1 in range(pred1.shape[0]):
        for index2 in range(pred2.shape[0]):
            # Extract features at indices 3, 4, 5 from both tensors
            features1 = pred1[index1, diff_genes_indice, :]
            features2 = pred2[index2, diff_genes_indice, :]

            print(features1)
            print(features2)
        
            # Calculate MSE for each feature pair across all samples
            mse_values = F.mse_loss(features1, features2, reduction='none').mean(dim=0)
            # Calculate the mean of these MSE values
            mean_mse = mse_values.mean()
            sum_mse = sum_mse + mean_mse
    sum_mean_mse = sum_mse / (pred1.shape[0] * pred2.shape[0])
    return sum_mean_mse


def cal_mse_tensors(pred1, pred2, diff_genes_indice):
    sum_mse=0
    for index1 in range(pred1.shape[0]):
        for index2 in range(pred2.shape[0]):
            # Extract features at indices 3, 4, 5 from both tensors
            features1 = pred1[index1, diff_genes_indice, :]
            features2 = pred2[index2, diff_genes_indice, :]

        
            # Calculate MSE for each feature pair across all samples
            mse_values = F.mse_loss(features1, features2, reduction='none').mean(dim=0)
            # Calculate the mean of these MSE values
            mean_mse = mse_values.mean()
            sum_mse = sum_mse + mean_mse
    sum_mean_mse = sum_mse / (pred1.shape[0] * pred2.shape[0])
    return sum_mean_mse


def get_gene_up_down(pred1, pred2, diff_genes_indice):
    matrix_count = []  # Initialize as an empty list
    for index1 in range(pred1.shape[0]):
        row = []  # Initialize a new row for each index1
        for index2 in range(pred2.shape[0]):
            count = []
            for diff_genes_index in diff_genes_indice:
                # Extract features at indices 3, 4, 5 from both tensors
                max_pos_before = torch.argmax(pred1[index1, diff_genes_index, :])
                max_pos_after = torch.argmax(pred2[index2, diff_genes_index, :])
                if max_pos_before == max_pos_after:
                    prob_change = pred1[index1, diff_genes_index, max_pos_before] - pred2[index2, diff_genes_index, max_pos_after]
                    if (prob_change > 0.0).all():  # Changed to handle tensor comparison
                        count.append(-1)
                    else:
                        count.append(1)
                elif max_pos_before > max_pos_after:
                    count.append(-1)
                else:
                    count.append(1)
            row.append(count)  # Append the count list to the current row
        matrix_count.append(row)  # Append the row to the main list
    # Reshape the matrix_count into a vector with the length of diff_genes_indice
    vote_result = np.zeros(len(diff_genes_indice))

    # Sum the counts for each gene across all comparisons
    for index1 in range(len(matrix_count)):
        for index2 in range(len(matrix_count[index1])):
            for gene_index in range(len(diff_genes_indice)):
                vote_result[gene_index] += matrix_count[index1][index2][gene_index]
    print(vote_result)
    return vote_result


def get_gene_up_down_old(preds, index, k):
    max_pos_before = torch.argmax(preds[k, indices[k], :])
    max_pos_after = torch.argmax(preds[index, indices[k], :])
    if max_pos_before == max_pos_after:
        prob_change = preds[index, indices[k],max_pos_after] - preds[k, indices[k], max_pos_before]
        if prob_change >0.0:
            return 1
        else:
            return -1
    elif max_pos_before > max_pos_after:
        return -1
    else:
        return 1


def get_gene_up_down_old_perturb(preds, length, k):
    max_pos_before = torch.argmax(preds[k, indices[k], :])
    max_pos_after = torch.argmax(preds[k+length-1, indices[k], :])
    if max_pos_before == max_pos_after:
        prob_change = preds[k+length-1, indices[k], max_pos_after] - preds[k, indices[k], max_pos_before]
        if prob_change > 0.0:
            return 1
        else:
            return -1
    elif max_pos_before > max_pos_after:
        return -1
    else:
        return 1

def get_gene_up_down_old_perturb_new(preds, length, k):
    max_pos_before = torch.argmax(preds[k, indices[k], :])
    max_pos_after = torch.argmax(preds[k+length-1, indices[k], :])
    if max_pos_before == max_pos_after:
        prob_change = preds[k+length-1, indices[k], max_pos_after] - preds[k, indices[k], max_pos_before]
        if prob_change > 0.0003:
            return 1
        elif prob_change >=0:
            return 0
        else:
            return -1
    elif max_pos_before > max_pos_after:
        return -1
    else:
        return 1



local_rank = int(os.getenv('LOCAL_RANK', '0'))
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
for ko_gene_name in ['AURKA','AURKB','AURKC','CENPE','CEP55','CIT','CREB1','E2F4','ECT2','EGR1','ELF1','ELK1','GABPA','IRF1','NR2C2','OGG1','RACGAP1','TOR1AIP1','YY1']:
    print(f"Control data and {ko_gene_name} knockout data loaded successfully.")
    ko_data_1 = sc.read_h5ad(f"../data/perturb_dixit/perturb_{ko_gene_name}.h5ad")
    ctrl_data = sc.read_h5ad(f"../data/perturb_ctrl.h5ad")
    sc.pp.normalize_total(ko_data_1, target_sum=1e4)
    sc.pp.normalize_total(ctrl_data, target_sum=1e4)
    ko_data_1.obs['condition'] = 'KO'
    ctrl_data.obs['condition'] = 'CTRL'
    adata = ko_data_1.concatenate(ctrl_data)
    sc.tl.rank_genes_groups(adata, groupby='condition', method='wilcoxon', groups=['KO'], reference='CTRL')
    group = 'KO'
    all_gene_names = adata.uns['rank_genes_groups']['names'][group]
    results = adata.uns['rank_genes_groups']
    gene_names = results['names']['KO']  
    log_fold_changes = results['logfoldchanges']['KO']  # log2 fold change
    pvals_adj = results['pvals_adj']['KO']  # adjusted p-values


    diff_genes_df = pd.DataFrame({
        'gene': gene_names,
        'log_fold_change': log_fold_changes,
        'pval_adj': pvals_adj
    })


    ko_expressions = ko_data_1.to_df()
    ctrl_expressions = ctrl_data.to_df()

    # check the gene that is expressed in both groups
    expressed_in_ko = (ko_expressions > 0).sum(axis=0)>=20
    expressed_in_ctrl = (ctrl_expressions > 0).sum(axis=0)>=20
    expressed_in_both = expressed_in_ko & expressed_in_ctrl


    # filter the gene with common expression in both KO and control group
    common_genes = expressed_in_both[expressed_in_both].index
    common_diff_genes = diff_genes_df[diff_genes_df['gene'].isin(common_genes)]

    # filter the genes with log fold change > 1 and pval_adj < 0.05
    positive_logfc_genes = common_diff_genes[(common_diff_genes['log_fold_change'] > 1)&(common_diff_genes['pval_adj'] <0.05)]
    positive_logfc_genes = positive_logfc_genes.sample(n=10)
 
    # filter the genes with log fold change < -1 and pval_adj < 0.05
    negative_logfc_genes = common_diff_genes[(common_diff_genes['log_fold_change'] < -1)&(common_diff_genes['pval_adj'] <0.05)]
    negative_logfc_genes= negative_logfc_genes.sample(n=10)

    # filter the genes with log fold change > -0.5 and log fold change < 0.5
    zero_logfc_genes = common_diff_genes[(common_diff_genes['log_fold_change'] > -0.5) & (common_diff_genes['log_fold_change'] < 0.5)]
  
    zero_logfc_genes=zero_logfc_genes.sample(n=10)




    # combine the three groups of genes total 30
    final_gene_names = np.concatenate([positive_logfc_genes['gene'], zero_logfc_genes['gene'], negative_logfc_genes['gene']])

    # save the gene names to a pt file
    torch.save(final_gene_names, f'../data/perturb_dixit/result/flag_mat_vol_perturb_ref_genename_{ko_gene_name}.pt')

    # get the indices of the genes
    top_gene_indices_1 = [adata.var_names.get_loc(gene) for gene in final_gene_names]
    top_gene_indices_1.append(adata.var_names.get_loc(ko_gene_name))

    print('this is length of target gene',len(top_gene_indices_1))
    # simulate cell two

    dist_mat_vol = []
    flag_mat_vol = []
    prob_mat_vol = []

    path = './ckpts/dixi_pretrain_10.pth'
    ckpt = torch.load(path,map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    #model.to_out = Identity(dropout=0., h_dim=128, out_dim=5)
    model = model.to(device)
    model.eval()

    for index in range(1, ctrl_data.shape[0]):
        print(index)
        cell_expression_data = ctrl_data[index, :].X
        cell_expression_dense = cell_expression_data.toarray()
        cell_expression = torch.from_numpy(cell_expression_dense).long()
        batch_mask,indices = apply_mask_pattern_for_pairs_perturb(cell_expression, MASK_TOKEN_ID, top_gene_indices_1)
        print(batch_mask.shape, indices.shape)
        # Save the tensor to a file
        dataset_mask=SCDataset(batch_mask)
        with torch.no_grad():
            batch_mask = batch_mask.to(device)
            preds = batch_inference(model, batch_mask, 10)
        torch.cuda.empty_cache()  # Clear unused memory from CUDA
        gc.collect()  # Explicitly invokes garbage collection
        #preds=model(batch_mask)
        dist_vec = torch.zeros(len(indices)-1)
        flag_gene= torch.ones(len(indices)-1)
        for i in range(len(indices)-1):
            dist = calculate_euclidean_distance(preds[i, indices[i],:], preds[i+len(indices)-1, indices[i], :])
            dist_vec[i] = dist
            flag_gene[i] = get_gene_up_down_old_perturb_new(preds, len(indices), i)
        
        dist_mat_vol.append(dist_vec)
        flag_mat_vol.append(flag_gene)
    torch.save(dist_mat_vol, f'../data/perturb_dixit/result/new_dist_mat_vol_perturb_{ko_gene_name}.pt')
    torch.save(flag_mat_vol, f'../data/perturb_dixit/result/new_flag_mat_vol_perturb_{ko_gene_name}.pt')
    data1 = ctrl_data[:, top_gene_indices_1]
    data2 = ko_data_1[:, top_gene_indices_1]
    expr_diff = data2.X.mean(axis=0) - data1.X.mean(axis=0)
    flags = np.sign(expr_diff).astype(int)
    torch.save(flags, f'../data/perturb_dixit/result/new_flag_mat_vol_perturb_ref_{ko_gene_name}.pt')
