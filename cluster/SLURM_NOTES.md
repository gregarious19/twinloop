# Cluster notes (owner fills this in on the CLUSTER before starting the Claude session)

## 1. Paste the cluster's Slurm instructions here
<!-- Paste everything you were given: login, partitions, accounts, example sbatch scripts, module
     commands, storage/scratch rules, limits. Do NOT paste passwords, tokens or keys. -->


## 2. Required facts (fill every row; write "not used" where it does not apply)
| Fact | Value |
|---|---|
| Partition name (faculty-level, full partition) | |
| Account (`--account`) | |
| QOS (`--qos`) | |
| How to request 1 GPU (e.g. `--gres=gpu:1` or `--gpus=1`) | |
| GPU model and memory (e.g. A100 80GB) | |
| Max wall time per job on that partition | |
| Max concurrent jobs / GPUs per user | |
| Module commands needed before Python/CUDA (e.g. `module load cuda/12.4`) | |
| Data/scratch directory with ≥ 150 GB free (absolute path) | |
| Is the login node allowed to reach the internet (pip)? yes/no | |
| Are environment variables exported into jobs by default? yes/no | |
| Absolute path of this repo on the cluster | |

## 3. Extracted (written by Claude in C00)
<!-- One line per fact with the value used in cluster/slurm_flags_*.txt and cluster/cluster.env -->
