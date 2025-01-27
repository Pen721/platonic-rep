import os
import argparse 

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import metrics
from tasks import get_models
import utils
from pprint import pprint
import pickle

def prepare_features(feats, q=0.95, exact=False):
    """
    Prepare features by removing outliers and normalizing
    Args:
        feats: a torch tensor of any share
        q: the quantile to remove outliers
    Returns:
        feats: a torch tensor of the same shape as the input
    """
    feats = metrics.remove_outliers(feats.float(), q=q, exact=exact)
    return feats.cuda()

# def prepare_features(feats, q=0.95, exact=False):
#     """
#     Prepare features by removing outliers and normalizing, doing quantile calculation on CPU
#     Args:
#         feats: a torch tensor of any shape
#         q: the quantile to remove outliers
#         exact: whether to use exact quantile computation
#     Returns:
#         feats: a torch tensor of the same shape as the input
#     """
#     # Move to CPU for quantile calculation
#     feats_cpu = feats.float().cpu()
#     feats_flat = feats_cpu.abs().flatten(start_dim=1)
#     q_val = torch.quantile(feats_flat, q, dim=1).mean()
    
#     # Move back to GPU for the thresholding operation
#     feats = feats.cuda()
#     q_val = q_val.cuda()
    
#     # Apply thresholding
#     feats = torch.minimum(feats.abs(), q_val) * torch.sign(feats)
    
#     return feats

def compute_score(x_feats, y_feats, metric="mutual_knn", topk=10, normalize=True):
    """
    Uses different layer combinations of x_feats and y_feats to find the best alignment
    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
    Returns:
        best_alignment_score: the best alignment score
        best_alignment: the indices of the best alignment
    """
    best_alignment_indices = None
    best_alignment_score = 0

    for i in range(-1, x_feats.shape[1]):
        x = x_feats.flatten(1, 2) if i == -1 else x_feats[:, i, :]

        for j in range(-1, y_feats.shape[1]):
            y = y_feats.flatten(1, 2) if j == -1 else y_feats[:, j, :]

            kwargs = {}
            if 'knn' in metric:
                kwargs['topk'] = topk
                    
            if normalize:
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)
            
            score = metrics.AlignmentMetrics.measure(metric, x, y, **kwargs)

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)
    
    return best_alignment_score, best_alignment_indices

def compute_score_and_return_all(x_feats, y_feats, metric="mutual_knn", topk=10, normalize=True):
    """
    Uses different layer combinations of x_feats and y_feats to find the best alignment
    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
    Returns:
        best_alignment_score: the best alignment score
        best_alignment: the indices of the best alignment
        all_alignments: a numpy array of shape (L+1) x (L+1) containing alignment scores
    """
    best_alignment_indices = None
    best_alignment_score = 0
    all_alignments = np.zeros((x_feats.shape[1], y_feats.shape[1]))

    for i in range(-1, x_feats.shape[1]):
        x = x_feats.flatten(1, 2) if i == -1 else x_feats[:, i, :]

        for j in range(-1, y_feats.shape[1]):
            y = y_feats.flatten(1, 2) if j == -1 else y_feats[:, j, :]

            kwargs = {}
            if 'knn' in metric:
                kwargs['topk'] = topk
                    
            if normalize:
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)
            
            score = metrics.AlignmentMetrics.measure(metric, x, y, **kwargs)
            all_alignments[i, j] = score

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)
    
    return best_alignment_score, best_alignment_indices, all_alignments

    
def compute_alignment(x_feat_paths, y_feat_paths, metric, topk, precise=True):
    """
    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
        metric: the metric to use
        topk: the number of nearest neighbors to use (specific to knn metrics)
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32 
            used in measure_alignment.py
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x 2
    """
    
    os.makedirs(args.output_dir, exist_ok=True)

    symmetric_metric = (x_feat_paths == y_feat_paths)
    if metric == "cycle_knn":
        symmetric_metric = False

    alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths)))
    alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), 2))

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))

    for i, x_fp in enumerate(x_feat_paths):
        x_feats = prepare_features(torch.load(x_fp, map_location="cuda:0")["feats"].float(), exact=precise)
            
        for j, y_fp in enumerate(y_feat_paths):
            if symmetric_metric:
                if i > j:
                    pbar.update(1)
                    continue           
                        
            y_feats = prepare_features(torch.load(y_fp, map_location="cuda:0")["feats"].float(), exact=precise)
            best_score, best_indices = compute_score(y_feats, x_feats, metric=metric, topk=topk)
            
            alignment_scores[i, j] = best_score
            alignment_indices[i, j] = best_indices
            
            if symmetric_metric:
                alignment_scores[j, i] = best_score
                alignment_indices[j, i] = best_indices[::-1]

            pbar.update(1)

            del y_feats
            torch.cuda.empty_cache()

    return alignment_scores, alignment_indices, all_alignments

def compute_alignment_and_save_all_results(x_feat_paths, y_feat_paths, metric, topk, precise=True, batch_size=1000):
    """
    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
        metric: the metric to use
        topk: the number of nearest neighbors to use (specific to knn metrics)
        precise: if true use exact quantiling
        batch_size: size of batches for processing features
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x 2
    """
    
    os.makedirs(args.output_dir, exist_ok=True)

    symmetric_metric = (x_feat_paths == y_feat_paths)
    if metric == "cycle_knn":
        symmetric_metric = False

    alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths)))
    alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), 2))

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))

    all_model_all_layer_alignments = {}

    # Process largest models first (assuming they're ordered by size)
    for i, x_fp in enumerate(x_feat_paths[::-1]):
        # Load features in half precision to save memory
        x_feats = prepare_features(torch.load(x_fp, map_location="cuda:0")["feats"].half(), exact=precise)
            
        for j, y_fp in enumerate(y_feat_paths[::-1]):
            if symmetric_metric and i > j:
                pbar.update(1)
                continue           
                        
            y_feats = prepare_features(torch.load(y_fp, map_location="cuda:0")["feats"].half(), exact=precise)
            
            try:
                best_score, best_indices, all_alignments = compute_score_and_return_all(
                    y_feats, x_feats, 
                    metric=metric, 
                    topk=topk,
                    batch_size=batch_size
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # If OOM occurs, try with a smaller batch size
                    torch.cuda.empty_cache()
                    best_score, best_indices, all_alignments = compute_score_and_return_all(
                        y_feats, x_feats, 
                        metric=metric, 
                        topk=topk,
                        batch_size=batch_size // 2
                    )
                else:
                    raise e
            
            # Store results
            alignment_scores[i, j] = best_score
            alignment_indices[i, j] = best_indices
            
            # Move alignments to CPU immediately
            all_model_all_layer_alignments[(i, j)] = all_alignments.detach().cpu()
            
            if symmetric_metric:
                alignment_scores[j, i] = best_score
                alignment_indices[j, i] = best_indices[::-1]

            pbar.update(1)

            # Clean up GPU memory
            del y_feats
            torch.cuda.empty_cache()
        
        # Clean up GPU memory after processing each x_feat
        del x_feats
        torch.cuda.empty_cache()

    return alignment_scores, alignment_indices, all_model_all_layer_alignments

if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str, default="minhuh/prh")
    parser.add_argument("--subset",         type=str, default="wit_1024")

    parser.add_argument("--modality_x",     type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--prompt_x",       action="store_true")
    parser.add_argument("--pool_x",         type=str, default=None, choices=['avg', 'cls'])
    
    parser.add_argument("--modality_y",     type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--prompt_y",       action="store_true")
    parser.add_argument("--pool_y",         type=str, default=None, choices=['avg', 'cls'])

    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--metric",         type=str, default="mutual_knn", choices=metrics.AlignmentMetrics.SUPPORTED_METRICS)
    parser.add_argument("--topk",           type=int, default=10)

    parser.add_argument("--input_dir",      type=str, default="./results/features")
    parser.add_argument("--output_dir",     type=str, default="./results/alignment")
    parser.add_argument("--precise",        action="store_true")
    parser.add_argument("--force_remake",   action="store_true")

    parser.add_argument("--dataset_x",      type=str, default="minhuh/prh")
    parser.add_argument("--dataset_y",      type=str, default="ninhuh/prh")

    parser.add_argument("--save_all_alignments",     type=bool, default=False)

    args = parser.parse_args()

    if args.dataset_x == args.dataset_y:
        args.dataset_x = args.dataset
        args.dataset_y = args.dataset
    else:
        args.dataset = args.dataset_x + "_" + args.dataset_y    

    if not args.precise:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    save_path = utils.to_alignment_filename(
            args.output_dir, args.dataset, args.modelset,
            args.modality_x, args.pool_x, args.prompt_x,
            args.modality_y, args.pool_y, args.prompt_y,
            args.metric, args.topk
    )

    all_alignments_save_path = utils.to_all_alignment_filename(
            args.output_dir, args.dataset, args.modelset,
            args.modality_x, args.pool_x, args.prompt_x,
            args.modality_y, args.pool_y, args.prompt_y,
            args.metric, args.topk
    )
    
    # if os.path.exists(save_path) and not args.force_remake:
    #     print(f"alignment already exists at {save_path}")
    #     exit()
    
    llm_models, lvm_models = get_models(args.modelset, modality='all')
    models_x = llm_models if args.modality_x == "language" else lvm_models
    models_y = llm_models if args.modality_y == "language" else lvm_models

    #models_x = ['bigscience/bloomz-1b1', 'bigscience/bloomz-560m']
    #models_y = ['bigscience/bloomz-1b1', 'bigscience/bloomz-560m']
    
    models_x_paths = [utils.to_feature_filename(args.input_dir, args.dataset_x, args.subset, m, args.pool_x, args.prompt_x) for m in models_x]
    models_y_paths = [utils.to_feature_filename(args.input_dir, args.dataset_y, args.subset, m, args.pool_y, args.prompt_y) for m in models_y]
    
    for fn in models_x_paths + models_y_paths:
        print(fn)
        assert os.path.exists(fn), fn
    
    print(f"dataset:\t{args.dataset}")
    print(f"metric: \t{args.metric}")
    if 'knn' in args.metric:
        print(f"topk:\t{args.topk}")
    
    print(f"models_x_paths:")    
    pprint(models_x_paths)
    print("\nmodels_y_paths:")
    pprint(models_y_paths)
    
    print('\nmeasuring alignment')

    if args.save_all_alignments:
        alignment_scores, alignment_indices, all_alignments = compute_alignment_and_save_all_results(models_x_paths, models_y_paths, args.metric, args.topk, args.precise)
    else:
        alignment_scores, alignment_indices = compute_alignment(models_x_paths, models_y_paths, args.metric, args.topk, args.precise)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, {"scores": alignment_scores, "indices": alignment_indices})
    print(f"saved to {save_path}")
    
    # Save all alignments
    if args.save_all_alignments:
        os.makedirs(os.path.dirname(all_alignments_save_path), exist_ok=True)
        with open(all_alignments_save_path, 'wb') as f:
            pickle.dump(all_alignments, f)
        print(f"saved all alignments to {all_alignments_save_path}")
    