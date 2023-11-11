#%%
import sys
sys.path.append('sparse_autoencoder/')
from sparse_autoencoder import TensorActivationStore, SparseAutoencoder, pipeline
from sparse_autoencoder.source_data.pile_uncopyrighted import PileUncopyrightedDataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_device
from transformers import PreTrainedTokenizerBase
import torch as t
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda:3' if t.cuda.is_available() else 'cpu'
## Model setup

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
src_d_mlp: int = model.cfg.d_mlp
print(src_d_mlp)

tokenizer: PreTrainedTokenizerBase = model.tokenizer  # type: ignore
source_data = PileUncopyrightedDataset(tokenizer=tokenizer)

max_items = 1_000_000
store = TensorActivationStore(max_items, src_d_mlp, device)

autoencoder = SparseAutoencoder(src_d_mlp, src_d_mlp * 8, t.zeros(src_d_mlp)).to(device)

pipeline(
    src_model=model,
    src_model_activation_hook_point="blocks.5.mlp.hook_post",
    src_model_activation_layer=5,
    source_dataset=source_data,
    activation_store=store,
    num_activations_before_training=max_items,
    autoencoder=autoencoder,
    device=device,
    max_activations=100_000_000,
)
# %%
t.save(autoencoder.state_dict(), 'models/encoder_mlp5_neo.pt')
# %%