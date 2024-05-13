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
# Define the model parameters
SEQ_LEN =  16907 # args.gene_num + 1
POS_EMBED_USING = True  # Whether to use position embedding
MASK_TOKEN_ID= 5
CLASS = 7
# bin the expression level. the pad id mask id comes from class +1 by Fu
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



def apply_mask_pattern_for_pairs(token_ids, MASK_TOKEN_ID):
    # read gene pos dict from a pt file
    loaded_gene_positions = torch.load('../data/gene_positions_breast.pt')
    indices = list(loaded_gene_positions.values())
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


def get_gene_up_down(preds, index, k):
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


local_rank = int(os.getenv('LOCAL_RANK', '0'))
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
input_data = sc.read_h5ad('../data/breast_pre.h5ad')
dist_mat_vol = []
flag_mat_vol = []
prob_mat_vol = []

path = './ckpts/breast_pretrain_10.pth'
ckpt = torch.load(path,map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False
#model.to_out = Identity(dropout=0., h_dim=128, out_dim=5)
model = model.to(device)
model.eval()
for index in range(1, input_data.shape[0]):
    cell_expression_data = input_data[index, :].X
    cell_expression_dense = cell_expression_data.toarray()
    cell_expression = torch.from_numpy(cell_expression_dense).long()
    batch_mask,indices = apply_mask_pattern_for_pairs(cell_expression, MASK_TOKEN_ID)
    # Save the tensor to a file
    dataset_mask=SCDataset(batch_mask)
    with torch.no_grad():
        batch_mask = batch_mask.to(device)
        preds = batch_inference(model, batch_mask, 10)
    torch.cuda.empty_cache()  # Clear unused memory from CUDA
    gc.collect()  # Explicitly invokes garbage collection
    dist_matrix = torch.zeros((len(indices), len(indices)))
    for i in range(len(indices)):
        if i != (len(indices) - 1):
            for j in range(i + 1, len(indices)):
                index = (i + 1) * len(indices) - (i + 1) * i // 2 + j - i - 1
                dist_1 = calculate_euclidean_distance(preds[i, indices[i], :], preds[index, indices[i], :])
                dist_2 = calculate_euclidean_distance(preds[j,indices[j], :], preds[index,indices[j], :])
                dist_matrix[i, j]=dist_1
                dist_matrix[j, i]=dist_2

    print(dist_matrix)

    dist_mat_vol.append((dist_matrix,index))


torch.save(dist_mat_vol, 'dist_mat_vol_breast_test.pt')

