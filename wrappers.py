import torch
from torch import nn
from torch_geometric.utils import dense_to_sparse


def encode_tokens(type_token, global_token, token_size, isgatr, batchsize, device):
    """Compute embedded type_token and global_token to be used within Transformers

    Parameters
    type_token: iterable of int
        list with type_tokens for each particle in the event
    global_token: int
    isgatr: bool
        whether the encoded tokens will be used within L-GATr or within the baseline Transformer
        This affects how many zeroes have to be padded to the global_token (4 more for the baseline Transformer)
    batchsize: int
    device: torch.device


    Returns:
    type_token: torch.Tensor with shape (batchsize, num_particles, type_token_max)
        one-hot-encoded type tokens, to be appended to each encoded 4-momenta in case of the
        baseline transformer / make up the full scalar channel for L-GATr
    global_token: torch.Tensor with shape (batchsize, 1, type_token_max+4)
        ont-hot-encoded dataset token, this will be the global_token and appended to the individual particles
    """
    type_token = nn.functional.one_hot(type_token, num_classes=token_size)
    type_token = (
        type_token.unsqueeze(1)
        .expand(type_token.shape[0], batchsize, *type_token.shape[1:])
        .float()
    )

    global_token = nn.functional.one_hot(
        global_token, num_classes=token_size + (0 if isgatr else 4)
    )
    global_token = (
        global_token.unsqueeze(1)
        .expand(global_token.shape[0], batchsize, *global_token.shape[1:])
        .unsqueeze(-2)
        .float()
    )
    return type_token, global_token


class AmplitudeMLPWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token, global_token, attn_mask=None):
        # ignore type_token, global_token and attn_mask (architecture is not permutation invariant)
        out = self.net(inputs)
        return out


class AmplitudeDSIWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token, global_token, attn_mask=None):
        out = self.net(inputs, type_token)
        return out


class AmplitudeTransformerWrapper(nn.Module):
    def __init__(self, net, token_size):
        super().__init__()
        self.net = net
        self.token_size = token_size

    def forward(self, inputs, type_token, global_token, attn_mask=None):
        nprocesses, batchsize, _, _ = inputs.shape

        type_token, global_token = encode_tokens(
            type_token,
            global_token,
            self.token_size,
            isgatr=False,
            batchsize=batchsize,
            device=inputs.device,
        )

        # type_token
        inputs = torch.cat((inputs, type_token), dim=-1)

        # global_token
        inputs = torch.cat((global_token, inputs), dim=-2)

        outputs = self.net(inputs, attention_mask=attn_mask)
        amplitudes = outputs[..., 0, :]

        return amplitudes