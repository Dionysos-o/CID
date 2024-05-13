# CID--Driver gene detection via causal inference on single cell embeddings

### CID a causal inference based on scBERT to identify driver genes from scRNA-seq data.
Driver genes are pivotal in different biological processes. Traditional methods gen-
erally identify driver genes by associative analysis. Leveraging on the development of current
large language models (LLM) in single cell biology, we propose a causal inference based ap-
proach called CID to identify driver genes from scRNA-seq data. Through experiments on three
different datasets, we show that CID can (1) identify biologically meaningful driver genes that
have not been captured by traditional associative-analysis based methods, and (2) accurately
predict the change directions of target genes if a driver gene is knocked out
insert figure here
![img.png](img.png)
# Install

Please refer the following link: https://github.com/TencentAILabHealthcare/scBERT for the installation of scBERT and the pre-trained model checkpoint.

The cuda version used in CID is 12.1 and the python version is 3.11.8.

# Data

The data can be downloaded from these links. 
- Pancreas islet data (Muraro et al., 2016): https://rdrr.io/github/LTLA/scRNAseq/man/MuraroPancreasData.html
- Perturb-seq data(Dixit et al., 2016): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96769
- Breast cancer data (GSE75688): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75688


and use the preprocess file to preprocess the data.
```
python preprocess.py --data_path "data_path" --output_path "output_path"
```
    
# Experiments

The expected input data of CID is a pre-processed gene expression matrix with format h5ad, which should be normalized and log-transformed by the preprocess file.  

If you want to use the pre-trained model by scBERT, you can directly set the model_path to the pre-trained ckpt file. 

- Otherwise, you can retrain the model by the following steps.
```
srun --time=5:00:00 --mem=100G --gres=gpu:2 torchrun --standalone --nnodes=1 --nproc_per_node=2 pretrain.py --data_path ../data/Muraro_pre.h5ad 
```

to conduct the experiments in the paper, you can run the following commands.
## Pancreas islet experiment
```
srun --time=5:00:00 --mem=50G --gpus=v100:1 torchrun --standalone --nnodes=1 --nproc_per_node=1 CID_identify_pancreaislet.py 
```

## Perturb-seq experiment
```
srun --time=5:00:00 --mem=50G --gpus=v100:1 torchrun --standalone --nnodes=1 --nproc_per_node=1 CID_identify_perturbseq.py 
```
## Breast cancer experiment
```
srun --time=5:00:00 --mem=50G --gpus=v100:1 torchrun --standalone --nnodes=1 --nproc_per_node=1 CID_identify_breastcancer.py 
```
(adjust the time and memory according to the dataset size and the number of gpus you have.)



# Citation
Yang, F., Wang, W., Wang, F. et al. scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data. Nat Mach Intell (2022). https://doi.org/10.1038/s42256-022-00534-z

