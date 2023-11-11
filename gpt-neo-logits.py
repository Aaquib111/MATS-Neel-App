#%% Imports
import numpy as np

import torch as t
from torch import Tensor

from transformer_lens import utils, HookedTransformer
import transformer_lens.patching as patching
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
from typing import Callable, Tuple, Union, Dict, Optional
from functools import partial
device = 'cuda' if t.cuda.is_available() else 'cpu'
#%% Model Setup
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
#%% Dataset Setup
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
corr_dataset = clean_dataset.gen_flipped_prompts('ABB->ABA, BAB->BAA')
corr_dataset.io_tokenIDs, corr_dataset.s_tokenIDs = corr_dataset.s_tokenIDs, corr_dataset.io_tokenIDs
rand_dataset = clean_dataset.gen_flipped_prompts('ABB->XYZ, BAB->XYZ')
#%% Sanity Check on Dataset
make_table(
colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
cols = [
    map(format_prompt, clean_dataset.sentences),
    model.to_string(clean_dataset.s_tokenIDs).split(),
    model.to_string(clean_dataset.io_tokenIDs).split(),
    map(format_prompt, corr_dataset.sentences),
],
title = "Sentences from IOI vs ABC distribution",
)
# %% Get resid directions, logit difference direction
correct_resid_dir = model.tokens_to_residual_directions(
    t.tensor(clean_dataset.io_tokenIDs)
)
wrong_resid_dir = model.tokens_to_residual_directions(
    t.tensor(clean_dataset.s_tokenIDs)
)
logit_diff_dir = correct_resid_dir - wrong_resid_dir
# %% Metric Setup
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
    corrupt_logits = model(corr_dataset.toks)
    clean_logit_diff = ave_logit_diff(clean_logits, clean_dataset).item()
    corrupt_logit_diff = ave_logit_diff(corrupt_logits, corr_dataset).item()

def ioi_metric(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    corrupted_logit_diff: float = corrupt_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
    ioi_dataset: IOIDataset = clean_dataset
 ):
    patched_logit_diff = ave_logit_diff(logits, ioi_dataset)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

#%% Plotting logit diff
original_logits, cache = model.run_with_cache(clean_dataset.toks)

final_residual_stream: Float[Tensor, "batch seq d_model"] = cache["resid_post", -1]
final_token_residual_stream = final_residual_stream[:, -1, :]
# Apply LayerNorm scaling (to just the final sequence position)
# pos_slice is the subset of the positions we take - here the final token of each prompt
scaled_final_token_residual_stream = cache.apply_ln_to_stack(final_token_residual_stream, layer=-1, pos_slice=-1)

average_logit_diff = einops.einsum(
    scaled_final_token_residual_stream, logit_diff_dir,
    "batch d_model, batch d_model ->"
) / len(clean_dataset.toks)

print(f"Calculated average logit diff: {average_logit_diff:.10f}")

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    cache,
    logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_dir,
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given 
    stack of components in the residual stream.
    '''

    scaled_residual_stream = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    average_logit_diff = einops.einsum(
        scaled_residual_stream, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / residual_stack.size(-2)

    return average_logit_diff


accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)

# accumulated_residual has shape (component, batch, d_model)

logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(accumulated_residual, cache)

line(
    logit_lens_logit_diffs, 
    hovermode="x unified",
    title="Logit Difference From Accumulated Residual Stream",
    labels={"x": "Layer", "y": "Logit Diff"},
    xaxis_tickvals=labels,
    width=800
)

per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache)

line(
    per_layer_logit_diffs, 
    hovermode="x unified",
    title="Logit Difference From Each Layer",
    labels={"x": "Layer", "y": "Logit Diff"},
    xaxis_tickvals=labels,
    width=800
)
#%% Activation patching on residual stream
resid_pre_act_patch_results = patching.get_act_patch_resid_pre(
    model,
    corr_dataset.toks, 
    cache, 
    ioi_metric
)
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_dataset.toks[0]))]

imshow(
    resid_pre_act_patch_results, 
    labels={"x": "Position", "y": "Layer"},
    x=labels,
    title="resid_pre Activation Patching",
    width=600
)

# %% Plotting attention patterns
from IPython.display import display
text = clean_dataset.sentences[0]
logits, cache = model.run_with_cache(text, remove_batch_dim=True)

str_tokens = model.to_str_tokens(text)
for layer in range(model.cfg.n_layers):
    print(f"LAYER {layer}")
    attention_pattern = cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))

# %%

original_logits, cache = model.run_with_cache(clean_dataset.toks)
act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(
    model, 
    corr_dataset.toks, 
    cache, 
    ioi_metric
)

imshow(
    act_patch_attn_head_out_all_pos, 
    labels={"y": "Layer", "x": "Head"}, 
    title="attn_head_out Activation Patching (All Pos)",
    width=600
)
# %% Patch MLPs 0-8
def mlp_patch_hook(mlp_out, hook, layer, mean_cache):
    if hook.layer() == layer:
        print(f'Patching at layer {layer}, {hook.name}')
        return mean_cache[hook.name]
    return mlp_out

logit_diffs = []
_, mean_cache = model.run_with_cache(
    corr_dataset.toks,
    stop_at_layer=9,
    names_filter=lambda name: name.endswith('mlp_out')
)
for patch_mlp in range(9):
    hook_fn = partial(
        mlp_patch_hook,
        layer=patch_mlp,
        mean_cache=mean_cache
    )
    with t.no_grad():
        logits = model.run_with_hooks(
            clean_dataset.toks,
            fwd_hooks=[
                (utils.get_act_name('mlp_out', patch_mlp), hook_fn)
            ]
        )
        logit_diff = ave_logit_diff(logits, clean_dataset)
    logit_diffs.append(logit_diff.item())
line(
    logit_diffs, 
    hovermode="x unified",
    title="Logit Difference After Ablating MLP",
    labels={"x": "MLP Layer", "y": "Logit Diff"},
    xaxis_tickvals=list(range(9)),
    width=700
)
#%% Patch all MLPs 1-8 at once

def mlp_patch_all(mlp_out, hook, mean_cache):
    if hook.layer() > 0 and hook.layer() < 9:
        print(f'Patching at layer {hook.layer()}, {hook.name}')
        return mean_cache[hook.name]
    return mlp_out

_, cache = model.run_with_cache(
    corr_dataset.toks,
    stop_at_layer=9,
    names_filter=lambda name: name.endswith('mlp_out')
)
hook_fn = partial(mlp_patch_all, mean_cache=cache)
with t.no_grad():
    logits = model.run_with_hooks(
        clean_dataset.toks,
        fwd_hooks=[
            (utils.get_act_name('mlp_out', i), hook_fn)
            for i in range(1, 9)
        ]
    )
    logit_diff = ave_logit_diff(logits, clean_dataset)
print(f'Logit diff after patching all MLPs: {logit_diff}')
# %% Patch attn heads before L9H4
import plotly.graph_objects as go

heads_to_patch = [
    (0, 6), (0, 7), (0, 11),
    (5, 1), (5, 11), 
    (6, 1), (6, 2), (6, 11),
    (7, 10)
]
def attn_patch_hook(z, hook, layer, head, cache):
    if hook.layer() == layer:
        print(f'Patching {layer=} {head=}')
        z[:, :, head, :] = cache[hook.name][:, :, head, :]
    return z

_, mean_cache = model.run_with_cache(
    corr_dataset.toks,
    stop_at_layer=8,
    names_filter=lambda name: name.endswith('hook_z')
)
logit_diffs = []
for (layer, head) in heads_to_patch:
    hook_fn = partial(
        attn_patch_hook,
        layer=layer,
        head=head,
        cache=mean_cache
    )

    with t.no_grad():
        logits = model.run_with_hooks(
            clean_dataset.toks,
            fwd_hooks=[
                (utils.get_act_name('z', layer), hook_fn)
            ]
        )
        logit_diff = ave_logit_diff(logits, clean_dataset)
    logit_diffs.append(logit_diff.item())
fig = go.Figure(
    data=go.Bar(
        x=[f'L{layer}H{head}' for (layer, head) in heads_to_patch], 
        y=logit_diffs
    )
)
# add title and labels
fig.update_layout(
    title='Logit Diff After Patching Attn Heads',
    xaxis=dict(
        title='Patched Attn Heads',
    ),
    yaxis=dict(
        title='Average Logit Diff',
    )
)
fig.show()
#%% What does L9H4 actually do??

# Get resid stream after first MLP
_, mlp_output = model.run_with_cache(
    clean_dataset.toks,
    stop_at_layer=9,
    names_filter=lambda name: name.endswith('mlp_out')
)
mlp_output = mlp_output['blocks.8.hook_mlp_out']
ov_matrix = einops.einsum(
    model.W_V[9, 4, :, :],
    model.W_O[9, 4, :, :],
    "d_model1 d_head, d_head d_model2 -> d_model1 d_model2"
)
for name_pos in [1, 3, 9]: #S1, IO, S2
    resid_at_name = mlp_output[:, name_pos, :]
    resid_ov = einops.einsum(
        ov_matrix, 
        resid_at_name,
        "d_model1 d_model2, batch d_model2 -> batch d_model1"
    )
    resid_embed = einops.einsum(
        resid_ov, 
        model.W_U,
        "batch d_model, d_model d_embed -> batch d_embed"
    )
    
    logits = model.ln_final(resid_embed)
    pred_toks = t.argsort(resid_embed, dim=1, descending=True)[:, :5]
    print((pred_toks == clean_dataset.toks[:, 9].unsqueeze(1)).any(dim=1))
# %% Path patching from all heads to L9H4 query vector
