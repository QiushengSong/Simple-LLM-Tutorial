import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..exceptions import *
from activation import SwiGLU

"""
layers:
    1.Word embedding layer(completed)
    2.Positional embedding layer(lack of RoPE and ABiLi)
    3.Normalization layer(completed)
    4.Multi-Query attention layer(completed) 
    5.MLP layer
    6.ModelBlock
    7.Model
"""

def repeat_kv(x: torch.Tensor,
              group_num: int
              ) -> torch.Tensor:

    batch, seq_len, head_num, d_kv = x.shape

    x = x[:, :, :, None, :]

    return x.expand(batch, seq_len, head_num, group_num, d_kv
                    ).view(batch, seq_len, head_num * group_num, d_kv)


def Attention(query, key, value, mask=None
              ) -> torch.Tensor:

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2,  -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    return torch.matmul(p_attn, value)


class WordEmbedding(nn.Module):
    """
    Word Embedding Module: Construct a vocabulary which includes "vocab_size" word and has "d_model" dimension of each word

    Args:
        vocab_size(int): Vocabulary size
        d_model(int): Dimension of each word
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=d_model)

    def forward(self, input_ids):

        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding Layer: The PE layer is used to encode the absolute position of each word
    """

    def __init__(self,
                 max_length: int,
                 d_model: int,
                 ):
        super().__init__()
        self.PE = torch.zeros(max_length, d_model)

        # position: torch.size(len_position, 1);
        position = torch.arange(max_length).unsqueeze(1)
        # div_term: torch.size(1, d_model
        div_term = torch.exp(torch.arange(0, d_model, step=2) * (-(math.log(10000) / d_model)))

        # position * div_term: (len_position * 1) * (1 * d_model) = (len_position * d_model);
        self.PE[:, 0::2] = torch.sin(position * div_term)
        self.PE[:, 1::2] = torch.cos(position * div_term)

        # insert batch dimension by broadcasting, PE size:[1, len_position, d_model];
        self.PE = self.PE.unsqueeze(0)

    def forward(self, input_ids):
        input_ids += self.PE[:, input_ids.size(1)]
        return input_ids


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 head_num: int,
                 ):
        super().__init__()
        if d_model % head_num != 0:
            raise AttentionError("The dimension of the model must be an integer multiple of the head_num")
        self.d_k = d_model / head_num
        self.head_num = head_num
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model),
                                     nn.Linear(d_model, d_model),
                                     nn.Linear(d_model, d_model),
                                     nn.Linear(d_model, d_model),
                                     ])

    def forward(self, query, key, value, mask=None):

        batch = query.size(0)

        seq_q_len = query.size(1)
        seq_k_len = key.size(1)
        seq_v_len = value.size(1)

        if mask is not None:
            mask = mask.unsqueeze(1)

        query = self.linear[0](query)
        key = self.linear[1](key)
        value = self.linear[2](value)

        query = query.view(batch, seq_q_len, self.head_num, self.d_k)
        key = key.view(batch, seq_k_len, self.head_num, self.d_k)
        value = value.view(batch, seq_v_len, self.head_num, self.d_k)

        # torch.size(batch, seq_q_len, head_num, d_k) -> torch.size(batch, head_num, seq_q_len, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        x = Attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch, -1, self.head_num * self.d_k)
        return self.linear[-1](x)


class MultiQueryAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 head_num: int,
                 device,
                 ):
        super().__init__()

        self.head_num = head_num
        self.d_model = d_model

        if d_model % head_num != 0:
            raise AttentionError("The dimension of the model must be an integer multiple of the head_num")

        self.d_v = int(d_model / head_num)
        self.query_linear = nn.Linear(d_model, d_model, device=device)
        self.key_linear = nn.Linear(d_model, self.d_v, device=device)
        self.value_linear = nn.Linear(d_model, self.d_v, device=device)
        self.output_linear = nn.Linear(d_model, d_model, device=device)

    def forward(self, x, mask=None):

        query = key = value = x

        batch = query.size(0)

        seq_q_len = query.size(1)

        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = query.view(batch, self.head_num, seq_q_len, self.d_v)
        key = torch.cat([key] * self.head_num, dim=1)
        value = torch.cat([value] * self.head_num, dim=1)

        x = Attention(query, key, value, mask=mask)
        x = x.view(batch, -1, self.head_num * self.d_k)

        return self.output_linear(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 query_head_num: int,
                 group_num: int,
                 device,
                 ):
        super().__init__()

        if d_model % query_head_num != 0 or query_head_num % group_num != 0:
            raise AttentionError("The dimension of the model must be an integer multiple of the head_num or")

        self.d_model = d_model
        self.query_head_num = query_head_num
        self.kv_head_num = int(query_head_num / group_num)
        self.group_num = group_num
        self.d_head = int(self.d_model / self.query_head_num)
        self.query_linear = nn.Linear(d_model, d_model, device=device)
        self.key_linear = nn.Linear(d_model, self.kv_head_num * self.d_head, device=device)
        self.value_linear = nn.Linear(d_model, self.kv_head_num * self.d_head, device=device)
        self.output_linear = nn.Linear(d_model, d_model, device=device)

    def forward(self, x, mask=None):

        query = key = value = x

        batch = x.size(0)

        seq_q_len = query.size(1)

        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = query.view(batch, seq_q_len, self.query_head_num, self.d_head)
        key = key.view(batch, seq_q_len, self.kv_head_num, self.d_head)
        value = value.view(batch, seq_q_len, self.kv_head_num, self.d_head)

        query = query.transpose(1, 2)


        key = repeat_kv(key, self.group_num)
        value = repeat_kv(value, self.group_num)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        x = Attention(query, key, value, mask=mask)

        x = x.view(batch, -1, self.query_head_num * self.d_head)

        return self.output_linear(x)


class Normalization(nn.Module):
    """
    Normalization Layer.

    Args:
       norm_type(str): Normalization method type, Option["LayerNorm", "RMSNorm", "DeepNorm"]
       d_model(int): Dimension of each token
       eps(float): A constant, it is used to prevent the denominator from being zero, thus increasing numerical stability.
       attention_layer(Attention): (optional)Attention layer, it is used in DeepNormalization
       feed_forward_layer(FeedForwardLayer): (optional)Feed forward layer, it is also used in DeepNormalization

    Example:

        >>> Norm = Normalization("LayerNorm", 512)
        >>> inputs = torch.randn(8, 64, 512)
        >>> outputs = Norm(inputs)

    """

    def __init__(self,
                 norm_type: str,
                 d_model: int,
                 eps: float = 1e-9,
                 attention_layer=None,
                 feed_forward_layer=None,
                 ):
        super().__init__()
        self.norm_type = norm_type
        self.eps = eps
        match self.norm_type:
            case "LayerNorm":
                self.factor_alpha = nn.Parameter(torch.ones(d_model))
                self.bias_beta = nn.Parameter(torch.zeros(d_model))

            case "RMSNorm":
                self.gamma = nn.Parameter(torch.ones(d_model))

            case "DeepNorm":
                if attention_layer is None or feed_forward_layer is None:
                    raise NormalizationError("DeepNormalization configuration lacks of"
                                             "attention layer or feed forward layer")
                self.factor_alpha = nn.Parameter(torch.ones(d_model))
                self.attention_layer = attention_layer
                self.feed_forward_layer = feed_forward_layer

            case _:
                raise NormalizationError("There are only 3 Normalization method: 'LayerNorm','RMSNorm' and 'DeepNorm'")

    def _LayerNorm(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keep=True)

        return self.factor_alpha * (x - mean) / std + self.bias_beta

    def _RMSNorm(self, x):

         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gamma

    def _DeepNorm(self, x):

        return self._LayerNorm(self.alpha * x +
                               self.feed_forward_layer(
                                   self.attention_layer(x)
                                   )
                               )

    def forward(self, x):

        match self.norm_type:
            case "LayerNorm":
                x = self._layernorm(x)
                return x

            case "RMSNorm":
                x = self._RMSNorm(x)
                return x

            case "DeepNorm":
                x = self._DeepNorm(x)
                return x

class AttentionLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 head_num: int,
                 group_num: int,
                 device,
                 ):
        super().__init__()
        self.attention = GroupedQueryAttention(d_model, head_num, group_num, device)
        self.normalization = Normalization("LayerNorm", d_model=d_model)

    def forward(self, x):

        origin_x = x
        x = self.normalization(x)
        x = self.attention(x)
        x += origin_x

        return x

class FeedForwardLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 hidden_size: int,
                 device,
                 ):
        super().__init__()
        self.ff_proj = nn.Linear(d_model, hidden_size, device=device)
        self.ff_out = nn.Linear(hidden_size, d_model, device=device)
        self.activation = SwiGLU(in_feature=d_model, out_feature=hidden_size)

    def forward(self, x):

        x = self.ff_proj(x)
        x = self.activation(x)
        x = self.ff_out(x)

        return x


class MLP(nn.Module):
    def __init__(self,
                 d_model,
                 hidden_size,
                 device,
                 ):
        super().__init__()
        self.feed_forward = FeedForwardLayer(d_model, hidden_size, device)
        self.normalization = Normalization("LayerNorm", d_model=d_model)

    def forward(self, x):
        origin_x = x
        x = self.normalization(x)
        x = self.feed_forward(x)
        x += origin_x

        return x

class LMBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 hidden_size: int,
                 head_num: int,
                 group_num: int,
                 device,
                 ):
        super().__init__()
        self.attention_layer = AttentionLayer(d_model, head_num, group_num, device)
        self.mlp = MLP(d_model, hidden_size, device)

    def forward(self, x):

        x = self.attention_layer(x)

        return self.mlp(x)


class CausalLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 hidden_size: int,
                 head_num: int,
                 group_num: int,
                 block_num: int,
                 device,
                 ):
        super().__init__()
        self.embedding = WordEmbedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([LMBlock(d_model, hidden_size, head_num, group_num, device) for _ in range(block_num)])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):

        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)

        output = self.output(x)

        return output