#%%
import sys
sys.path.append('sparse_autoencoder')
import torch as t
import numpy as np
import pandas as pd
from sparse_autoencoder import SparseAutoencoder 
from transformer_lens import utils, HookedTransformer, ActivationCache

device = 'cuda:1' if t.cuda.is_available() else 'cpu'
#%% SAE Setup
LAYER = 10
d_mlp = 3072
encoder = SparseAutoencoder(d_mlp, d_mlp * 8, t.zeros(d_mlp)).to(device)
encoder.load_state_dict(t.load(f'models/encoder_mlp{LAYER}_neo.pt'))
encoder.eval()
# %% Model Setup
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
model.eval()
#%% Dataset setup

from acdcpp.ioi_task.ioi_dataset import IOIDataset, format_prompt, make_table
N = 50
# To make life easier
clean_dataset = IOIDataset(
    prompt_type=[
        #"While [A] and [B] were working at the [PLACE], [B] gave a [OBJECT] to",
        #"While [A] and [B] commuted to the [PLACE], [B] gave a [OBJECT] to",
        "When [A] and [B] went to the [PLACE], [B] gave a [OBJECT] to [A]",
        "When [A] and [B] argued at the [PLACE], [B] said something angrily to [A]",
        "When [A] and [B] found [OBJECT] at [PLACE]. [B] gave a [OBJECT] to [A]",
        "While [A] and [B] worked at the [PLACE], [B] assigned some work to [A]",
        "When [A] and [B] raced to the [PLACE], [B] was running slower than [A]",
        "When [A] and [B] had a small fight, [B] threw the [OBJECT] at [A]",
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
# %% Reconstruction loss functions
from functools import partial 

def replacement_hook(mlp_post, hook, encoder):
    mlp_post_recon = encoder(mlp_post)[1]
    return mlp_post_recon
def mean_ablation_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post
def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post

def get_recon_loss(encoder, toks):
    with t.no_grad():
        loss = model(toks, return_type='loss')
        recons_loss = model.run_with_hooks(
            toks, 
            return_type='loss',
            fwd_hooks=[
                (
                    utils.get_act_name('post', LAYER),
                    partial(replacement_hook, encoder=encoder)
                )
            ]
        )
        zero_abl_loss = model.run_with_hooks(
            toks, 
            return_type='loss',
            fwd_hooks=[
                (
                    utils.get_act_name('post', LAYER),
                    zero_ablate_hook
                )
            ]
        )
        recons_score = ((zero_abl_loss - recons_loss) / (zero_abl_loss - loss))

    return recons_score.item(), loss.item(), recons_loss.item(), zero_abl_loss.item()

def get_freqs(encoder, toks):
    with t.no_grad():
        freqs = t.zeros(8 * d_mlp, dtype=t.float32).to(device)
        total = 0
        _, cache = model.run_with_cache(
            toks,
            stop_at_layer=LAYER + 1,
            names_filter=utils.get_act_name('post', LAYER)
        )
        mlp_acts = cache[utils.get_act_name('post', LAYER)]
        mlp_acts = mlp_acts.reshape(-1, d_mlp)
        hidden = encoder(mlp_acts)[0]
        
        total += hidden.shape[0]
        freqs += (hidden > 0).sum(0) / total
        num_dead = (freqs == 0).float().mean()
    return freqs, num_dead

SPACE = "·"
NEWLINE="↩"
TAB = "→"
def process_token(s):
    if isinstance(s, t.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

def process_tokens(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, t.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [process_token(s) for s in l]

def list_flatten(nested_list):
    return [x for y in nested_list for x in y]
def make_token_df(tokens, len_prefix=5, len_suffix=1):
    str_tokens = [process_tokens(model.to_str_tokens(t)) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    batch = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        batch=batch,
        pos=pos,
        label=label,
    ))

#%% Get reconstruction loss 
score, loss, rec_loss, zero_loss = get_recon_loss(encoder, clean_dataset.toks)
print(f'{score=}, {loss=}, {rec_loss=}, {zero_loss=}')
# %% Feature Frequencies
freqs, num_dead = get_freqs(encoder, clean_dataset.toks)
print(f'{freqs=}, {num_dead=}')
# %% Inspecting top feature
feature_idx = t.argsort(freqs, descending=True)[5]
print(f'Frequency of top feature: {freqs[t.argmax(freqs)]}')
_, cache = model.run_with_cache(
    clean_dataset.toks,
    stop_at_layer=LAYER+1,
    names_filter=utils.get_act_name('post', LAYER)
)
mlp_acts = cache[utils.get_act_name('post', LAYER)].reshape(-1, d_mlp)
learned_act, decoded_act = encoder(mlp_acts)
df = make_token_df(clean_dataset.toks)
df['feature'] = utils.to_numpy(learned_act[:, feature_idx])
df.sort_values('feature', ascending=False).head(20).style.background_gradient('coolwarm')
# %%
