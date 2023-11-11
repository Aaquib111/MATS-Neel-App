#%% Imports
import os
import re
# Edge Attribution Patching!
from acdcpp.ACDCPPExperiment import ACDCPPExperiment

import numpy as np

import torch as t
from torch import Tensor

from transformer_lens import utils, HookedTransformer, ActivationCache
import transformer_lens.patching as patching
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP

import circuitsvis as cv
import einops

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from plotly_utils import imshow, line, scatter, bar
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.style.use('seaborn-v0_8-paper')

from jaxtyping import Float, Bool
from typing import Callable, Tuple, Union, Dict, Optional, List
from functools import partial
device = 'cuda' if t.cuda.is_available() else 'cpu'
#%% Model setup

model = HookedTransformer.from_pretrained(
    'EleutherAI/gpt-neo-125M',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=True,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
#%% Dataset setup

from acdcpp.ioi_task.ioi_dataset import IOIDataset, format_prompt, make_table
N = 50
# To make life easier
clean_dataset = IOIDataset(
    prompt_type=[
        #"While [A] and [B] were working at the [PLACE], [B] gave a [OBJECT] to",
        #"While [A] and [B] commuted to the [PLACE], [B] gave a [OBJECT] to",
        "When [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to",
        "When [B] and [A] argued at the [PLACE], [B] said something angrily to",
        "When [B] and [A] found [OBJECT] at [PLACE]. [B] gave a [OBJECT] to",
        "While [B] and [A] worked at the [PLACE], [B] assigned some work to",
        "When [B] and [A] raced to the [PLACE], [B] was running slower than",
        "When [B] and [A] had a small fight, [B] threw the [OBJECT] at",
    ],
    N=N,
    tokenizer=None,#model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=device
)
rand_dataset = clean_dataset.gen_flipped_prompts('ABB->XYZ, BAB->XYZ')

#%% Dataset sanity check
make_table(
colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "XYZ prompt"],
cols = [
    map(format_prompt, clean_dataset.sentences),
    model.to_string(clean_dataset.s_tokenIDs).split(),
    model.to_string(clean_dataset.io_tokenIDs).split(),
    map(format_prompt, rand_dataset.sentences),
],
title = "Sentences from IOI vs ABC distribution",
)
# %% Metric setup 
def ave_logit_diff(
    logits: Float[Tensor, 'batch seq d_vocab'],
    ioi_dataset: IOIDataset,
    per_prompt: bool = False
):
    '''
        Return average logit difference between correct and incorrect answers
    '''
    # Get logits for indirect objects
    io_logits = logits[range(logits.size(0)), -1, ioi_dataset.io_tokenIDs]
    s_logits = logits[range(logits.size(0)), -1, ioi_dataset.s_tokenIDs]
    #print(io_logits)
    #print(s_logits)
    # Get logits for subject
    logit_diff = io_logits - s_logits
    return logit_diff if per_prompt else logit_diff.mean()

with t.no_grad():
    clean_logits = model(clean_dataset.toks)
    corrupt_logits = model(rand_dataset.toks)
    clean_logit_diff = ave_logit_diff(clean_logits, clean_dataset).item()
    corrupt_logit_diff = ave_logit_diff(corrupt_logits, rand_dataset).item()

def ioi_metric(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    corrupted_logit_diff: float = corrupt_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
    ioi_dataset: IOIDataset = clean_dataset
 ):
    patched_logit_diff = ave_logit_diff(logits, ioi_dataset)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def abs_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return abs(ioi_metric(logits))

def negative_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -ioi_metric(logits)

# Get clean and corrupt logit differences
with t.no_grad():
    clean_metric = ioi_metric(clean_logits, corrupt_logit_diff, clean_logit_diff, clean_dataset)
    corrupt_metric = ioi_metric(corrupt_logits, corrupt_logit_diff, clean_logit_diff, rand_dataset)

print(f'Clean direction: {clean_logit_diff}, Corrupt direction: {corrupt_logit_diff}')
print(f'Clean metric: {clean_metric}, Corrupt metric: {corrupt_metric}')
# %% Running Edge Attribution Patching
THRESHOLDS = [-10] # Negative threshold initially since I dont want to prune
RUN_NAME = 'abs_edge'

acdcpp_exp = ACDCPPExperiment(
    model=model,
    clean_data=clean_dataset.toks,
    corr_data=rand_dataset.toks,
    acdc_metric=negative_ioi_metric,
    acdcpp_metric=ioi_metric,
    thresholds=THRESHOLDS,
    run_name=RUN_NAME,
    verbose=False,
    attr_absolute_val=True,
    save_graphs_after=-10000, # I want the graphs saved always
    pruning_mode='edge',
    no_pruned_nodes_attr=1,
)

_, _, acdcpp_pruned_attrs, _, edges_after_acdcpp,_ = acdcpp_exp.run()
attrs = list(acdcpp_pruned_attrs[-10].values())
attr_std = np.std(attrs)
attr_mean = np.mean(attrs)
print(f'{attr_std=} {attr_mean=}')
print(ave_logit_diff(model(clean_dataset.toks), clean_dataset))
plt.title('Distribution of Edge Scores for GPT-Neo on IOI')
plt.xlabel('Change in Logit Difference')
plt.ylabel('Count')
plt.hist(attrs, log=True)
# %% Pruning now

THRESHOLDS = [attr_mean + (2 * attr_std)] # Try pruning away +/- 2 std. dev 
RUN_NAME = 'abs_edge_prune_2std'

acdcpp_exp = ACDCPPExperiment(
    model=model,
    clean_data=clean_dataset.toks,
    corr_data=rand_dataset.toks,
    acdc_metric=negative_ioi_metric,
    acdcpp_metric=ioi_metric,
    thresholds=THRESHOLDS,
    run_name=RUN_NAME,
    verbose=False,
    attr_absolute_val=True,
    save_graphs_after=-10000, # I want the graphs saved always
    pruning_mode='edge',
    no_pruned_nodes_attr=1,
)
print(ave_logit_diff(model(clean_dataset.toks), clean_dataset))
_, _, acdcpp_pruned_attrs, _, edges_after_acdcpp,_ = acdcpp_exp.run()
# %% Traversing subnetwork

from traverse_gv import find_all_paths, get_edge_values

# Find all paths
start_node = 'embed'
end_node = '<resid_post>'
# just gets the first .gv file, there should only be one anyway
all_paths = find_all_paths('ims/abs_edge_prune_2std/*.gv', start_node, end_node)
paths_with_edge_vals = get_edge_values('ims/abs_edge_prune_2std/*.gv', all_paths)
paths_with_edge_vals = sorted(paths_with_edge_vals, key=lambda path: -sum(abs(float(edge[2])) for edge in path if edge[2] is not None))
topk_paths = paths_with_edge_vals[:30]
for path in topk_paths:
    path_str = ''
    for (start, end, weight) in path:
        path_str += f'{start} -->{float(weight):.2f}--> '
    path_str += '<resid_post>'
    print(path_str)
#%% Mean ablating all other nodes 

# Not sure what sequence positions to ablate, so I will keep all of them
def z_hook(
    z: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    attn_means: Float[Tensor, "layer seq head d_head"],
    attns_to_keep: List, 
) -> Float[Tensor, "batch seq head d_head"]:
    '''
    Ablates the z output of a transformer head.

    means
        Tensor of mean z values of the means_dataset over each group of prompts
        with the same template. This tells us what values to mask with.
    '''
    # Get the mask for this layer, and add d_head=1 dimension so it broadcasts correctly
    heads_to_keep = []
    for (layer, head) in attns_to_keep:
        if layer == hook.layer():
            heads_to_keep.append(head)
    layer_mask = t.zeros(len(rand_dataset), rand_dataset.max_len, model.cfg.n_heads, dtype=t.bool)
    layer_mask[:, :, heads_to_keep] = 1
    layer_mask = layer_mask.unsqueeze(-1).to(device)

    repeated_means = einops.repeat(
        attn_means[hook.layer()],
        "seq_len head d_head -> batch seq_len head d_head",
        batch=len(rand_dataset)
    ).to(device)
    # Set z values to the mean 
    z = t.where(layer_mask, z, repeated_means)

    return z

def mlp_hook(
    mlp_out: Float[Tensor, "batch seq d_model"],
    hook: HookPoint, 
    mlp_means: Float[Tensor, "layer seq d_model"],
    mlps_to_keep: List
) -> Float[Tensor, "batch seq d_model"]:
    '''
        Ablates MLP output. Does this for all seq pos
    '''
    batch = mlp_out.shape[0]
    if hook.layer() not in mlps_to_keep:
        return einops.repeat(
            mlp_means[hook.layer()],
            "seq_len d_model -> batch seq_len d_model",
            batch=batch
        ).to(device)
    return mlp_out

def compute_means(
    means_dataset: IOIDataset, 
    model: HookedTransformer
) -> Float[Tensor, "layer batch seq head_idx d_head"]:
    '''
        Return mean activations of every attention head and MLP layer
    '''
    # Cache the outputs of every head, MLP
    _, means_cache = model.run_with_cache(
        means_dataset.toks.long(),
        return_type=None,
        names_filter=lambda name: name.endswith("z") or name.endswith("mlp_out"),
    )
    # Create tensor to store means
    n_layers, n_heads, d_head, d_model = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model
    batch, seq_len = len(means_dataset), means_dataset.max_len
    attn_means = t.zeros(size=(n_layers, seq_len, n_heads, d_head), device=model.cfg.device)
    mlp_means = t.zeros(size=(n_layers, seq_len, d_model))

    get_layer = re.compile('^blocks.([0-9]+)(.attn)?.hook_(z|mlp_out)$')
    for act_name in means_cache.keys():
        layer_match = get_layer.match(act_name)
        assert layer_match, f"Layer Number not found for {act_name}" 
        layer_num = int(layer_match.group(1))
        act_type = layer_match.group(3)

        if act_type == 'z':
            # Take the mean over all batches
            mean_act = einops.reduce(
                means_cache[act_name], 
                "batch seq_len head d_head -> seq_len head d_head",
                "mean"
            )
            attn_means[layer_num] = mean_act
        elif act_type == 'mlp_out':
            mean_act = einops.reduce(
                means_cache[act_name], 
                "batch seq_len d_mlp -> seq_len d_mlp",
                "mean"
            )
            mlp_means[layer_num] = mean_act
        else:
            print(f'Not Supported Type: {act_type}')
    return attn_means, mlp_means


def add_mean_ablation_hook(
    model: HookedTransformer, 
    keep_heads: List,
    keep_mlps: List, 
    is_permanent: bool = True,
) -> HookedTransformer:
    '''
    Adds a permanent hook to the model, which ablates according to the circuit and 
    seq_pos_to_keep dictionaries.

    In other words, when the model is run on ioi_dataset, every head's output will 
    be replaced with the mean over means_dataset for sequences with the same template,
    except for a subset of heads and sequence positions as specified by the circuit
    and seq_pos_to_keep dicts.
    '''

    model.reset_hooks(including_permanent=True)

    # Compute the mean of each head's output on the ABC dataset, grouped by template
    attn_means, mlp_means = compute_means(rand_dataset, model)

    # Get a hook function which will patch in the mean z values for each head, at 
    # all positions which aren't important for the circuit
    attn_hook_fn = partial(
        z_hook, 
        attn_means=attn_means,
        attns_to_keep=keep_heads, 
    )
    mlp_hook_fn = partial(
        mlp_hook,
        mlp_means=mlp_means,
        mlps_to_keep=keep_mlps,
    )
    
    # Apply hooks
    model.add_hook(
        lambda name: name.endswith("z"), 
        attn_hook_fn, 
        is_permanent=is_permanent
    )
    model.add_hook(
        lambda name: name.endswith("mlp_out"), 
        mlp_hook_fn, 
        is_permanent=is_permanent
    )

    return model
# %% Add hooks to model
model.reset_hooks(including_permanent=True)
print(f'Before Ablation: {ave_logit_diff(model(clean_dataset.toks), clean_dataset)}')

two_std_heads = [
    (0, 2), (0, 5), (0, 6), (0, 7), (0, 10), (0, 11),
    (1, 8), (1, 11),
    (2, 6),
    (3, 1), (3, 11),
    (4, 8), (4, 6), (4, 11),
    (5, 0), (5, 6), (5, 7), (5, 8), (5, 10),
    (6, 1), (6, 4), (6, 6), (6, 7), (6, 11),
    (7, 3), (7, 6), (7, 10), (7, 11),
    (8, 4), (8, 6), (8, 9),
    (9, 2), (9, 4), (9,9),
    (10, 0), (10, 3), (10, 6),
    (11, 2), (11, 4), (11, 6),
]
two_std_mlps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

traversed_heads = [
    (0, 2), (6, 4), (7, 10), (9, 4), (11, 6)
]
traversed_mlps = [0, 2, 5, 6, 7, 8, 10, 11]

add_mean_ablation_hook(
    model, 
    keep_heads=two_std_heads, 
    keep_mlps=two_std_mlps
)
print(f'After 2std Ablation: {ave_logit_diff(model(clean_dataset.toks), clean_dataset)}')

model.reset_hooks(including_permanent=True)

add_mean_ablation_hook(
    model, 
    keep_heads=traversed_heads, 
    keep_mlps=traversed_mlps
)
print(f'After traversal ablation: {ave_logit_diff(model(clean_dataset.toks), clean_dataset)}')

model.reset_hooks(including_permanent=True)

add_mean_ablation_hook(
    model, 
    keep_heads=traversed_heads + [(0, 5), (0, 6), (0, 11)], 
    keep_mlps=traversed_mlps
)
print(f'After traversal with additions ablation: {ave_logit_diff(model(clean_dataset.toks), clean_dataset)}')
# %%
