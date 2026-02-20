#layers.py
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def get_initializer(initializer):
    if initializer is None:
        return torch.nn.init.xavier_uniform_
    elif isinstance(initializer, str):
        return getattr(torch.nn.init, initializer)
    else:
        return initializer


def get_activation(activation):
    if activation is None:
        return F.relu
    elif isinstance(activation, str):
        return getattr(F, activation)
    else:
        return activation


def default_initialization(torch_layer: nn.Module):
    torch.nn.init.kaiming_uniform_(torch_layer.weight)
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(torch_layer.weight)
    bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
    torch.nn.init.uniform_(torch_layer.bias, -bound, bound)


# ───────────────────── Embedding (+ wavelength meta) ─────────────────────
class NonLinearEmbedding(nn.Module):
    """
    Token embedding with optional magnitude gating (inputs) and optional token metadata.
    If `token_metadata` is provided as a tensor of shape [num_tokens, meta_dim], we map it
    through a tiny MLP and ADD it to the token embedding. Column convention:

        [:,0] = lambda_norm in [0,1]   (0 for non-flux tokens)
        [:,1] = has_lambda {0,1}
        [:,2] = is_narrow {0,1}  (1=J-PAS narrow band, 0=broad band)

    Row 0 is the padding token and MUST be zeros.

    Forward:
      - `input_tokens` : (B,S) int
      - `inputs`       : (B,S,*) optional magnitudes; if provided, we use out*mag + bias.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embeddings_initializer: Callable[[torch.Tensor], None] = torch.nn.init.uniform_,
        kernel_initializer=None,
        bias_initializer=None,
        input_length: int = None,
        activation: str = "elu",
        use_bias: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        token_metadata: torch.Tensor = None,   
        meta_hidden: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = get_initializer(embeddings_initializer)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.input_length = input_length
        self.activation_fn = get_activation(activation)
        self.use_bias = use_bias
        self.padding_idx = 0

        factory_kwargs = {"device": device, "dtype": dtype}

        self.embeddings = Parameter(torch.empty((self.input_dim, self.output_dim), **factory_kwargs))
        self.bias = Parameter(torch.empty((self.input_dim, self.output_dim), **factory_kwargs))

        # metadata buffer + tiny MLP
        if token_metadata is not None:
            if not isinstance(token_metadata, torch.Tensor):
                token_metadata = torch.tensor(token_metadata, **factory_kwargs)
            else:
                token_metadata = token_metadata.to(device=device, dtype=dtype)
            assert token_metadata.dim() == 2 and token_metadata.size(0) == self.input_dim, \
                "token_metadata must be [input_dim, meta_dim] and include padding row at index 0."
            self.register_buffer("token_metadata", token_metadata, persistent=False)
            meta_dim = token_metadata.size(1)
            hid = max(meta_hidden, min(128, self.output_dim // 2))
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, hid, **factory_kwargs),
                nn.SiLU(),
                nn.Linear(hid, self.output_dim, **factory_kwargs),
            )
            for m in self.meta_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            self.register_buffer("token_metadata", None, persistent=False)
            self.meta_mlp = None

        self.reset_parameters()

    def reset_parameters(self):
        self.embeddings_initializer(self.embeddings)
        if self.use_bias:
            self.bias_initializer(self.bias)
        else:
            with torch.no_grad():
                torch.nn.init.zeros_(self.bias)
        with torch.no_grad():
            # zero pad row
            self.embeddings[self.padding_idx].fill_(0)
            self.bias[self.padding_idx].fill_(0)
            if self.token_metadata is not None:
                self.token_metadata[self.padding_idx].fill_(0)

    def gather_metadata(
        self, input_tokens: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns (lambda_norm, has_lambda, is_narrow) each (B,S) or None if no metadata.
        """
        if self.token_metadata is None:
            return None
        idx = input_tokens.long()
        meta = self.token_metadata[idx]         # (B,S,M)
        lam    = meta[..., 0]
        has    = meta[..., 1] if meta.size(-1) > 1 else torch.zeros_like(lam)
        narrow = meta[..., 2] if meta.size(-1) > 2 else torch.zeros_like(lam)
        return lam, has, narrow

    def forward(self, input_tokens: torch.Tensor, inputs: torch.Tensor = None) -> torch.Tensor:
        idx = input_tokens.long()
        out = F.embedding(idx, self.embeddings)  # (B,S,E)

        # add metadata-driven vector
        if self.token_metadata is not None:
            meta = self.token_metadata[idx]  # (B,S,M)
            gate = meta[..., 1:2].clamp(0, 1)  # has_lambda column
            meta_vec = self.meta_mlp(meta) * gate
            out = out + meta_vec

        if inputs is not None:
            mag = inputs
            bias = F.embedding(idx, self.bias)
            return self.activation_fn(out * mag + bias)
        else:
            return out


# ─────────────────── Flash self-attn (encoder) with precomputed mask ───────────────────
class FlashMHA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.0,
                 device:  torch.device = "cuda",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.dropout_p  = dropout
        factory_kwargs  = {"device": device, "dtype": dtype}
        self.qkv_proj   = nn.Linear(embed_dim, 3 * embed_dim, bias=True, **factory_kwargs)
        self.out_proj   = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                x: torch.Tensor,                           # (B,S,E)
                attn_mask_additive: Optional[torch.Tensor] = None,  # (B,1,S,S)
                is_causal: bool = False):
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)                                     # (B,S,3E)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                           # (B,H,S,D)

        attn = F.scaled_dot_product_attention(q, k, v,
                                              attn_mask=attn_mask_additive,
                                              dropout_p=self.dropout_p if self.training else 0.0,
                                              is_causal=is_causal)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, S, self.embed_dim)
        out  = self.out_proj(attn)
        return out


# ───────────────────────── Flash cross-attn (decoder) ─────────────────────────
class FlashCrossMHA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.0,
                 device:  torch.device = "cuda",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.dropout_p  = dropout
        factory_kwargs  = {"device": device, "dtype": dtype}

        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        # global K/V parameters per head (broadcast over batch)
        self.k_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.head_dim, **factory_kwargs))
        self.v_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.head_dim, **factory_kwargs))

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    @staticmethod
    def _kpad_to_add_mask(key_padding_mask: Optional[torch.Tensor],
                          Q: int, K: int, dtype, device, extra_k: int = 0):
        if key_padding_mask is None:
            return None
        B = key_padding_mask.size(0)
        bigneg = torch.tensor(-1e4, device=device, dtype=dtype)
        mask = torch.zeros((B, 1, Q, K + extra_k), device=device, dtype=dtype)
        if K > 0:
            pad_keys = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, Q, K)
            mask[:, :, :, :K] = mask[:, :, :, :K].masked_fill(pad_keys, bigneg)
        return mask

    def forward(self,
                q_in: torch.Tensor,            # (B,Q,E)
                k_in: torch.Tensor,            # (B,K,E)
                v_in: Optional[torch.Tensor] = None,  # (B,K,E)
                key_padding_mask: Optional[torch.Tensor] = None,  # (B,K) bool
                is_causal: bool = False):
        if v_in is None:
            v_in = k_in
        B, Q, _ = q_in.shape
        K = k_in.shape[1]

        q = self.q_proj(q_in).view(B, Q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k_in).view(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v_in).view(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # append global K/V (per head), unmasked
        kb = self.k_bias.expand(B, -1, -1, -1)  # (B,H,1,D)
        vb = self.v_bias.expand(B, -1, -1, -1)  # (B,H,1,D)
        k = torch.cat([k, kb], dim=2)           # (B,H,K+1,D)
        v = torch.cat([v, vb], dim=2)           # (B,H,K+1,D)

        attn_mask = self._kpad_to_add_mask(key_padding_mask, Q, K, q.dtype, q.device, extra_k=1)

        attn = F.scaled_dot_product_attention(q, k, v,
                                              attn_mask=attn_mask,
                                              dropout_p=self.dropout_p if self.training else 0.0,
                                              is_causal=is_causal)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, Q, self.embed_dim)
        out  = self.out_proj(attn)
        return out


# ─────────────────────────── Transformer block ───────────────────────────
class TransformerBlock(nn.Module):
    def __init__(
        self,
        head_num: int,
        dense_num: int,
        embedding_dim: int,
        dropout_rate: float,
        activation: Callable[[torch.Tensor], torch.Tensor],
        cross_attn: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.activation_fn = get_activation(activation)
        self.cross_attn = cross_attn

        if cross_attn:
            self.attention = FlashCrossMHA(embedding_dim, head_num,
                                           dropout=dropout_rate, **self.factory_kwargs)
        else:
            self.attention = FlashMHA(embedding_dim, head_num,
                                      dropout=dropout_rate, **self.factory_kwargs)

        self.dense_1 = torch.nn.Linear(embedding_dim, dense_num, **self.factory_kwargs)
        self.dense_2 = torch.nn.Linear(dense_num, embedding_dim, **self.factory_kwargs)
        self.layernorm_1 = torch.nn.LayerNorm(embedding_dim, **self.factory_kwargs)
        self.layernorm_2 = torch.nn.LayerNorm(embedding_dim, **self.factory_kwargs)
        self.dropout_1 = torch.nn.Dropout(dropout_rate)
        self.dropout_2 = torch.nn.Dropout(dropout_rate)

        default_initialization(self.dense_1)
        default_initialization(self.dense_2)

    def forward(
        self,
        input_query: torch.Tensor,
        input_value: torch.Tensor,
        input_key: torch.Tensor,
        attn_mask_additive: Optional[torch.Tensor] = None,  # (B,1,S,S) for self-attn
        key_padding_mask: Optional[torch.Tensor] = None,    # (B,K) for cross-attn
    ) -> torch.Tensor:
        if self.cross_attn:
            attention_out = self.attention(input_query,
                                           input_key, input_value,
                                           key_padding_mask=key_padding_mask,
                                           is_causal=False)
        else:
            attention_out = self.attention(input_query,
                                           attn_mask_additive=attn_mask_additive,
                                           is_causal=False)

        attention_out = self.dropout_1(attention_out)
        attention_out = self.layernorm_1(input_query + attention_out)

        x = self.activation_fn(self.dense_1(attention_out))
        x = self.dense_2(x)
        x = self.dropout_2(x)
        x = self.layernorm_2(attention_out + x)
        return x


# ─────────────────────────────── Encoder ───────────────────────────────
class Encoder(nn.Module):
    def __init__(
        self,
        encoder_head_num: int,
        encoder_dense_num: int,
        embedding_dim: int,
        encoder_dropout_rate: float,
        encoder_activation,
        transformer_class: nn.Module,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.activation = encoder_activation
        self.transformer_class = transformer_class

        self.encoder_transformer_block_1 = self.transformer_class(
            head_num=encoder_head_num,
            dense_num=encoder_dense_num,
            embedding_dim=embedding_dim,
            dropout_rate=encoder_dropout_rate,
            activation=encoder_activation,
            cross_attn=False,
            **self.factory_kwargs,
        )
        self.encoder_transformer_block_2 = self.transformer_class(
            head_num=encoder_head_num // 2,
            dense_num=encoder_dense_num // 2,
            embedding_dim=embedding_dim,
            dropout_rate=encoder_dropout_rate,
            activation=encoder_activation,
            cross_attn=False,
            **self.factory_kwargs,
        )

        # learnable relative-λ kernel params (shared) + per-block gains
        self.log_sigma_narrow = nn.Parameter(torch.tensor(-2.30, **self.factory_kwargs))  
        self.log_sigma_broad  = nn.Parameter(torch.tensor(-1.61, **self.factory_kwargs))  #
        self.w_narrow = nn.Parameter(torch.tensor(0.10, **self.factory_kwargs))
        self.w_broad  = nn.Parameter(torch.tensor(0.10, **self.factory_kwargs))

        self.rpb_gain1 = nn.Parameter(torch.tensor(1.0, **self.factory_kwargs))
        self.rpb_gain2 = nn.Parameter(torch.tensor(1.0, **self.factory_kwargs))

    @staticmethod
    def _pad_to_add_mask(key_padding_mask: Optional[torch.Tensor],
                         S: int, dtype, device):
        if key_padding_mask is None:
            return None
        B = key_padding_mask.size(0)
        bigneg = torch.tensor(-1e4, device=device, dtype=dtype)
        mask = torch.zeros((B, 1, S, S), device=device, dtype=dtype)
        pad_keys = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, S, S)
        mask = mask.masked_fill(pad_keys, bigneg)
        pad_queries = key_padding_mask.unsqueeze(1).unsqueeze(3).expand(B, 1, S, S)
        mask = mask.masked_fill(pad_queries, bigneg)
        return mask

    def _relpos_bias(self, lam: torch.Tensor, has: torch.Tensor, narrow: torch.Tensor) -> torch.Tensor:
        B, S = lam.shape
        lam_i = lam.unsqueeze(-1)
        lam_j = lam.unsqueeze(-2)
        dist  = (lam_i - lam_j).abs()

        both_narrow = (narrow.unsqueeze(-1) * narrow.unsqueeze(-2)).to(dist.dtype)

        sigma_n = self.log_sigma_narrow.exp()
        sigma_b = self.log_sigma_broad.exp()
        w_n     = self.w_narrow
        w_b     = self.w_broad

        kern_n = w_n * torch.exp(- (dist / (sigma_n + 1e-6))**2)
        kern_b = w_b * torch.exp(- (dist / (sigma_b + 1e-6))**2)
        bias   = both_narrow * kern_n + (1.0 - both_narrow) * kern_b

        # zero where λ unknown
        bias = bias * (has.unsqueeze(-1) * has.unsqueeze(-2))
        return bias.unsqueeze(1)

    def forward(
        self,
        inputs: torch.Tensor,            # (B,S,E)
        mask: torch.Tensor = None,       # (B,S) bool
        relpos_lambda: torch.Tensor = None,   # (B,S)
        relpos_has: torch.Tensor = None,      # (B,S)
        relpos_narrow: torch.Tensor = None,   # (B,S)
    ) -> torch.Tensor:
        add_mask = None
        if (relpos_lambda is not None) and (relpos_has is not None) and (relpos_narrow is not None):
            rpb = self._relpos_bias(relpos_lambda, relpos_has, relpos_narrow).to(inputs.dtype)
        else:
            rpb = None

        if mask is not None:
            pad_mask = self._pad_to_add_mask(mask, inputs.size(1), inputs.dtype, inputs.device)
        else:
            pad_mask = None

        if pad_mask is not None and rpb is not None:
            add_mask = pad_mask + rpb
        elif pad_mask is not None:
            add_mask = pad_mask
        elif rpb is not None:
            add_mask = rpb

        add_mask1 = add_mask * self.rpb_gain1 if add_mask is not None else None
        add_mask2 = add_mask * self.rpb_gain2 if add_mask is not None else None

        t1 = self.encoder_transformer_block_1(
            input_query=inputs, input_value=inputs, input_key=inputs,
            attn_mask_additive=add_mask1
        )
        t2 = self.encoder_transformer_block_2(
            input_query=t1, input_value=t1, input_key=t1,
            attn_mask_additive=add_mask2
        )
        return t2

# ─────────────────────────────── Decoder ───────────────────────────────
class Decoder(nn.Module):
    def __init__(
        self,
        decoder_head_num: int,
        decoder_dense_num: int,
        embedding_dim: int,
        decoder_dropout_rate: float,
        decoder_activation,
        transformer_class: nn.Module,
        num_classes: int,
        num_morph_classes: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.num_classes = num_classes
        self.transformer_class = transformer_class
        self.activation_fn = get_activation(decoder_activation)

        self.classification_out = torch.nn.Linear(decoder_dense_num // 8, num_classes, **self.factory_kwargs)
        self.morph_classification_out = torch.nn.Linear(decoder_dense_num // 8, num_morph_classes, **self.factory_kwargs)
        default_initialization(self.classification_out)
        default_initialization(self.morph_classification_out)

        self.decoder_transformer_block_1 = self.transformer_class(
            head_num=decoder_head_num,
            dense_num=decoder_dense_num,
            embedding_dim=embedding_dim,
            dropout_rate=decoder_dropout_rate,
            activation=decoder_activation,
            cross_attn=True,
            **self.factory_kwargs,
        )
        self.decoder_transformer_block_2 = self.transformer_class(
            head_num=decoder_head_num // 2,
            dense_num=decoder_dense_num // 2,
            embedding_dim=embedding_dim,
            dropout_rate=decoder_dropout_rate,
            activation=decoder_activation,
            cross_attn=True,
            **self.factory_kwargs,
        )
        self.decoder_transformer_block_3 = self.transformer_class(
            head_num=decoder_head_num // 4,
            dense_num=decoder_dense_num // 4,
            embedding_dim=embedding_dim,
            dropout_rate=decoder_dropout_rate,
            activation=decoder_activation,
            cross_attn=True,
            **self.factory_kwargs,
        )

        self.decoder_dense_3 = torch.nn.Linear(embedding_dim, decoder_dense_num, **self.factory_kwargs)
        self.decoder_dense_4 = torch.nn.Linear(decoder_dense_num, decoder_dense_num // 2, **self.factory_kwargs)
        self.decoder_dense_5 = torch.nn.Linear(decoder_dense_num // 2, decoder_dense_num // 4, **self.factory_kwargs)
        self.decoder_dense_6 = torch.nn.Linear(decoder_dense_num // 4, decoder_dense_num // 8, **self.factory_kwargs)
        self.decoder_out     = torch.nn.Linear(decoder_dense_num // 8, 1, **self.factory_kwargs)
        self.decoder_logvar_out = torch.nn.Linear(decoder_dense_num // 8, 1, **self.factory_kwargs)
        self.decoder_dropout_1 = torch.nn.Dropout(decoder_dropout_rate)

        default_initialization(self.decoder_dense_3)
        default_initialization(self.decoder_dense_4)
        default_initialization(self.decoder_dense_5)
        default_initialization(self.decoder_dense_6)
        default_initialization(self.decoder_out)
        default_initialization(self.decoder_logvar_out)

    def forward(
        self,
        unit_vec: torch.Tensor,          # (B, Q, E) typically Q=1
        encoder_outputs: torch.Tensor,   # (B, K, E)
        mask: torch.Tensor = None,       # (B, K) bool
    ) -> torch.Tensor:
        t1 = self.decoder_transformer_block_1(
            input_query=unit_vec, input_value=encoder_outputs, input_key=encoder_outputs,
            key_padding_mask=mask
        )
        t2 = self.decoder_transformer_block_2(
            input_query=t1, input_value=encoder_outputs, input_key=encoder_outputs,
            key_padding_mask=mask
        )
        t3 = self.decoder_transformer_block_3(
            input_query=t2, input_value=encoder_outputs, input_key=encoder_outputs,
            key_padding_mask=mask
        )

        x = self.activation_fn(self.decoder_dense_3(t3)); x = self.decoder_dropout_1(x)
        x = self.activation_fn(self.decoder_dense_4(x)); x = self.decoder_dropout_1(x)
        x = self.activation_fn(self.decoder_dense_5(x)); x = self.decoder_dropout_1(x)
        x = self.activation_fn(self.decoder_dense_6(x)); x = self.decoder_dropout_1(x)

        preds        = self.decoder_out(x)
        preds_logvar = self.decoder_logvar_out(x)
        class_preds  = self.classification_out(x)
        morph_preds  = self.morph_classification_out(x)

        return preds, preds_logvar, class_preds, morph_preds

class TransformerTorchModel(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Module,
        embedding_dim: int,
        encoder_head_num: int,
        encoder_dense_num: int,
        encoder_dropout_rate: float,
        encoder_activation: Callable[[torch.Tensor], torch.Tensor],
        decoder_head_num: int,
        decoder_dense_num: int,
        decoder_dropout_rate: float,
        decoder_activation: Callable[[torch.Tensor], torch.Tensor],
        num_classes: int,
        num_morph_classes: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding_layer = embedding_layer

        self.torch_encoder = Encoder(
            encoder_head_num=encoder_head_num,
            encoder_dense_num=encoder_dense_num,
            embedding_dim=embedding_dim,
            encoder_dropout_rate=encoder_dropout_rate,
            encoder_activation=encoder_activation,
            transformer_class=TransformerBlock,
            **factory_kwargs,
        )
        self.torch_decoder = Decoder(
            decoder_head_num=decoder_head_num,
            decoder_dense_num=decoder_dense_num,
            embedding_dim=embedding_dim,
            decoder_dropout_rate=decoder_dropout_rate,
            decoder_activation=decoder_activation,
            transformer_class=TransformerBlock,
            num_classes=num_classes,
            num_morph_classes=num_morph_classes,
            **factory_kwargs,
        )

    def forward(
        self,
        input_tensor,           # (B,S,1) valores de entrada
        input_token_tensor,     # (B,S)   tokens de entrada
        output_token_tensor,    # (B,Q)   tokens de salida a predecir
    ):
        # Encoder embeddings (+ per-token metadata projection)
        input_embedded = self.embedding_layer(input_token_tensor, input_tensor)

        # Wavelength sequences for relative-λ bias
        meta = self.embedding_layer.gather_metadata(input_token_tensor)
        if meta is not None:
            lam_seq, has_seq, narrow_seq = meta  # (B,S)
        else:
            lam_seq = has_seq = narrow_seq = None

        padding_mask = torch.eq(input_token_tensor, torch.zeros_like(input_token_tensor))  # (B,S) bool

        perception = self.torch_encoder(
            input_embedded,
            mask=padding_mask,
            relpos_lambda=lam_seq,
            relpos_has=has_seq,
            relpos_narrow=narrow_seq,
        )

        # mean-pool encoder features over valid (non-pad) tokens → (B,E)
        valid = (~padding_mask).float()                           # (B,S)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        enc_pooled = (perception * valid.unsqueeze(-1)).sum(dim=1) / denom

        # Decoder (cross-attn) sin condicionamiento
        output_embedding = self.embedding_layer(output_token_tensor)  # (B,Q,E)

        preds, preds_logvar, class_preds, morph_preds = self.torch_decoder(
            output_embedding, perception, mask=padding_mask
        )
        return preds, preds_logvar, class_preds, morph_preds, enc_pooled
