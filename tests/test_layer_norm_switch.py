"""model.net.layer_norm=false (models/lloca.py): every transformer block's pre-norms are identities, nothing else
changes (same parameters at the same seed), and the net still runs and depends on the input scale."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_layer_norm_off():
    from models.lloca import LLOCAMuPTransformer
    from models.transformer_lloca_mup import BaselineTransformerBlock
    kw = dict(num_scalars=4, hidden_channels_mlp=16, num_layers_mlp=2, in_channels=8, attn_reps="8x0n+2x1n",
              out_channels=1, num_blocks=2, num_heads=2)
    torch.manual_seed(0); a = LLOCAMuPTransformer(**kw)
    torch.manual_seed(0); b = LLOCAMuPTransformer(layer_norm=False, **kw)
    blocks = [m for m in b.net.modules() if isinstance(m, BaselineTransformerBlock)]
    assert len(blocks) == 2 and all(isinstance(m.norm1, torch.nn.Identity) and isinstance(m.norm2, torch.nn.Identity)
                                    for m in blocks)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


if __name__ == "__main__":
    test_layer_norm_off(); print("ok")
