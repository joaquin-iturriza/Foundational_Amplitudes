import torch
from torch import nn
from torch_geometric.utils import dense_to_sparse
from lgatr.interface import embed_vector, extract_scalar


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
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token, global_token, attn_mask=None):
        nprocesses, batchsize, _ = inputs.shape

        outputs = self.net(inputs, attention_mask=attn_mask)

        return outputs


class AmplitudeGATrWrapper(nn.Module):
    def __init__(self, net, token_size, reinsert_type_token=False):
        super().__init__()
        self.net = net
        self.token_size = token_size

    def forward(self, inputs: torch.Tensor, type_token, global_token, attn_mask=None):
        multivector, scalars = self.embed_into_ga(inputs, type_token, global_token)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars
        )
        amplitude = self.extract_from_ga(multivector_outputs, scalar_outputs)
        return amplitude

    def embed_into_ga(self, inputs, type_token, global_token):
        inputs=inputs.reshape(1,inputs.shape[1], inputs.shape[2] // 4, 4)
        
        nprocesses, batchsize, num_objects, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        multivector = multivector.unsqueeze(-2)

        type_token, global_token = encode_tokens(
            type_token,
            global_token,
            self.token_size,
            isgatr=True,
            batchsize=batchsize,
            device=inputs.device,
        )
        type_token = type_token.to(inputs.dtype)
        global_token = global_token.to(inputs.dtype)

        # encode type_token in scalars
        scalars = type_token

        # global token
        global_token_mv = torch.zeros(
            (nprocesses, batchsize, 1, multivector.shape[-2], multivector.shape[-1]),
            dtype=multivector.dtype,
            device=multivector.device,
        )
        global_token_s = global_token
        multivector = torch.cat((global_token_mv, multivector), dim=-3)
        scalars = torch.cat((global_token_s, scalars), dim=-2)
        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        # Extract scalars from GA representation
        lorentz_scalars = extract_scalar(multivector)[..., 0]

        amplitude = lorentz_scalars[..., 0, :]
        return amplitude
        

class AmplitudeLLoCaWrapper(nn.Module):
    def __init__(self, net, token_size):
        super().__init__()
        self.net = net
        self.network_dtype = torch.float32
        self.token_size = token_size

    def forward(self, inputs, type_token, mean, std):
        particle_type = torch.nn.functional.one_hot(
            type_token, num_classes=type_token.max() + 1
        )
        particle_type = torch.nn.functional.one_hot(
            type_token, num_classes=self.token_size   # <-- use fixed size
        )
        particle_type = particle_type.to(dtype=self.network_dtype, device=inputs.device)
        outputs = self.net(inputs, particle_type, mean, std)
        return outputs.mean(dim=-2)
