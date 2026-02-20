import re
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch
memory_format = torch.channels_last
import numpy as np
from typing import List, Optional, Tuple, Dict, Union

from .model_core import TranformerCore
from .layers import NonLinearEmbedding, TransformerTorchModel
from .nn_utils import TrainingGenerator

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lam: float = 1.0):
        super().__init__(); self.lam = lam
    def forward(self, x):
        return _GRL.apply(x, self.lam)

class DomainDiscriminator(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_domains: int = 2, device="cpu", dtype=torch.float32):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden, **fk), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden//2, **fk), torch.nn.GELU(),
            torch.nn.Linear(hidden//2, num_domains, **fk)
        )
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
    def forward(self, h): return self.net(h)

class ReweightMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=128, device="cpu", dtype=torch.float32):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden, **fk), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden//2, **fk), torch.nn.GELU(),
            torch.nn.Linear(hidden//2, 1, **fk)
        )
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
    def forward(self, h):
        return torch.nn.functional.softplus(self.net(h)).squeeze(-1) + 1e-6  # (B,)



def _median_heuristic_sigma(x: torch.Tensor) -> torch.Tensor:

    x = x.float() 
    with torch.no_grad():
        if x.size(0) > 2000:
            idx = torch.randperm(x.size(0))[:2000]
            x = x[idx]
            
        d2 = torch.cdist(x, x, p=2.0).pow(2)
        vals = d2[d2 > 0]
        if vals.numel() == 0:
            return torch.tensor(1.0, device=x.device)
        return torch.median(vals).sqrt().clamp_min(1e-6)

def rbf_mmd2_weighted(x: torch.Tensor, y: torch.Tensor,
                      w_x: Optional[torch.Tensor] = None,
                      w_y: Optional[torch.Tensor] = None,
                      sigma: Optional[torch.Tensor] = None):
    """
    Usa una sola sigma dinámica en lugar de multiescala fija.
    """
    x = x.float()
    y = y.float()
    if w_x is not None: w_x = w_x.float()
    if w_y is not None: w_y = w_y.float()

    if sigma is None:
        sigma = _median_heuristic_sigma(torch.cat([x, y], dim=0))

    gamma = 1.0 / (2 * sigma**2 + 1e-12)


    
    # Calculamos kernel completo K([x,y], [x,y]) de golpe para usar la sigma correcta
    xy_cat = torch.cat([x, y], dim=0)
    
    # 1. Distancias al cuadrado seguras
    sq_norm = xy_cat.pow(2).sum(1, keepdim=True)
    dist_mat = torch.addmm(sq_norm, xy_cat, xy_cat.t(), alpha=-2.0, beta=1.0) + sq_norm.t()
    dist_mat = dist_mat.clamp_min(0.0)
    
    # 2. Kernel
    K_mat = torch.exp(-gamma * dist_mat)
    
    # 3. Separar bloques
    m = x.size(0)
    n = y.size(0)
    k_xx = K_mat[:m, :m]
    k_yy = K_mat[m:, m:]
    k_xy = K_mat[:m, m:]

    # Term XX
    if w_x is None:
        term_xx = (k_xx.sum() - k_xx.diagonal().sum()) / (m * (m - 1) + 1e-12)
    else:
        w_x = w_x / (w_x.sum() + 1e-12) * m
        Wxx = torch.outer(w_x, w_x) * k_xx
        denom = (w_x.sum()**2 - (w_x**2).sum()).clamp_min(1e-12)
        term_xx = (Wxx.sum() - Wxx.diagonal().sum()) / denom

    # Term YY
    if w_y is None:
        term_yy = (k_yy.sum() - k_yy.diagonal().sum()) / (n * (n - 1) + 1e-12)
    else:
        w_y = w_y / (w_y.sum() + 1e-12) * n
        Wyy = torch.outer(w_y, w_y) * k_yy
        denom = (w_y.sum()**2 - (w_y**2).sum()).clamp_min(1e-12)
        term_yy = (Wyy.sum() - Wyy.diagonal().sum()) / denom

    # Term XY
    if w_x is None and w_y is None:
        term_xy = 2 * k_xy.mean()
    elif w_x is None:
        w_y = w_y / (w_y.sum() + 1e-12) * n
        term_xy = 2 * (k_xy * w_y.unsqueeze(0)).sum() / (m * w_y.sum() + 1e-12)
    elif w_y is None:
        w_x = w_x / (w_x.sum() + 1e-12) * m
        term_xy = 2 * (w_x.unsqueeze(1) * k_xy).sum() / (n * w_x.sum() + 1e-12)
    else:
        w_x = w_x / (w_x.sum() + 1e-12) * m
        w_y = w_y / (w_y.sum() + 1e-12) * n
        num = 2 * (w_x.unsqueeze(1) * k_xy * w_y.unsqueeze(0)).sum()
        den = (w_x.sum() * w_y.sum()).clamp_min(1e-12)
        term_xy = num / den

    return term_xx + term_yy - term_xy

class OJALA(TranformerCore):


    def _build_wavelength_metadata(self, log_scale: bool = True) -> torch.Tensor:
            """
            Return a tensor of shape [vocab_size+1, 3]:
            [:,0] = lambda_norm in [0,1] (Linear or Log scale)
            [:,1] = has_lambda (1 if known wavelength)
            [:,2] = is_narrow (1 for J-PAS narrow bands)

            Index 0 (padding) is zeros.
            """
            import numpy as np
            
            # Diccionario de longitudes de onda centrales (nm)
            known_flux_nm = {
                "FLUX_G": 477.0, "FLUX_R": 623.0, "FLUX_Z": 913.0, "MAG_uSDSS": 355.0, "MAG_gSDSS": 477.0, "MAG_rSDSS": 623.0,
                "MAG_iSDSS": 748.0, "MAG_zSDSS": 893.0, "MAG_ySDSS": 1030.0, "MAG_G": 477.0, "MAG_R": 623.0, "MAG_i": 748.0, "MAG_Z": 893.0,
                "MAG_W1" : 3368, "MAG_W2" : 4618, "MAG_W3" : 12082, "MAG_W4" : 22194,
                "MAG_J_2MASS" : 1235, "MAG_H_2MASS" : 1662, "MAG_Ks_2MASS" : 2159,
                "MAG_NUV" : 226.7, "MAG_FUV" : 151.6,
                "FLUX_W1": 3368, "FLUX_W2": 4618, 
            }

            def parse_lambda_and_bandtype(name: str):
                if re.fullmatch(r"J\d{4}", name):   # JPAS narrow filters like "J0480"
                    return float(name[1:]), 1.0
                if name in known_flux_nm:
                    return known_flux_nm[name], 0.0
                return None, 0.0

            raw_lambda, is_narrow = [], []
            for n in self.vocabs:
                lam, nar = parse_lambda_and_bandtype(n)
                raw_lambda.append(lam)
                is_narrow.append(nar)

            vals = [x for x in raw_lambda if x is not None and np.isfinite(x)]
            meta = np.zeros((self.vocab_size + 1, 3), dtype=np.float32)
            
            if len(vals) == 0:
                return torch.tensor(meta, device=self.device, dtype=self.dtype)

            if log_scale:
                vals_processed = [np.log10(x) for x in vals]
                vmin, vmax = min(vals_processed), max(vals_processed)
            else:
                vals_processed = vals
                vmin, vmax = min(vals_processed), max(vals_processed)

            span = max(vmax - vmin, 1e-6)

            lam_norm, has_gate = [], []
            for x in raw_lambda:
                if x is None or not np.isfinite(x):
                    lam_norm.append(0.0)
                    has_gate.append(0.0)
                else:
                    val_curr = np.log10(x) if log_scale else x
                    norm_val = (val_curr - vmin) / span
                    lam_norm.append(norm_val)
                    has_gate.append(1.0)

            meta[1:, 0] = np.array(lam_norm, dtype=np.float32)
            meta[1:, 1] = np.array(has_gate, dtype=np.float32)
            meta[1:, 2] = np.array(is_narrow, dtype=np.float32)
            
            return torch.tensor(meta, device=self.device, dtype=self.dtype)


    def __init__(
        self,
        vocabs: List[str],
        vocab_tokens: List[int] = None,
        context_length: int = 30,
        embedding_dim: int = 32,
        num_classes: int = 3,
        num_morph_classes: int = 6,
        embedding_activation=None,
        encoder_head_num: int = 2,
        encoder_dense_num: int = 128,
        encoder_dropout_rate: float = 0.1,
        encoder_activation=None,
        decoder_head_num: int = 2,
        decoder_dense_num: int = 128,
        decoder_dropout_rate: float = 0.1,
        decoder_activation=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        mixed_precision: bool = False,
        folder: str = "model_torch",
        built: bool = False,
    ) -> None:
        super().__init__(
            vocabs=vocabs,
            backend_framewoark=f"torch-{torch.__version__[:5]}",
            vocab_tokens=vocab_tokens,
            context_length=context_length,
            embedding_dim=embedding_dim,
            embedding_activation=embedding_activation,
            encoder_head_num=encoder_head_num,
            encoder_dense_num=encoder_dense_num,
            encoder_dropout_rate=encoder_dropout_rate,
            encoder_activation=encoder_activation,
            decoder_head_num=decoder_head_num,
            decoder_dense_num=decoder_dense_num,
            decoder_dropout_rate=decoder_dropout_rate,
            decoder_activation=decoder_activation,
            device=device,
            dtype=dtype,
            mixed_precision=mixed_precision,
            folder=folder,
            built=built,
        )
        self.implemented_backend = "torch"
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"

        # wavelength metadata per token
        token_metadata = self._build_wavelength_metadata()

        self.embedding_layer = NonLinearEmbedding(
            input_dim=self.vocab_size + 1,  # +1 for padding
            output_dim=self.embedding_dim,
            embeddings_initializer=torch.nn.init.xavier_uniform_,
            activation=self.embedding_activation,
            token_metadata=token_metadata,
            **self.factory_kwargs,
        )

        self.num_classes = num_classes
        self.num_morph_classes = num_morph_classes

        # ====================== Modelo principal ======================
        self.torch_model = TransformerTorchModel(
            self.embedding_layer,
            embedding_dim=self.embedding_dim,
            encoder_head_num=self.encoder_head_num,
            encoder_dense_num=self.encoder_dense_num,
            encoder_dropout_rate=self.encoder_dropout_rate,
            encoder_activation=self.encoder_activation,
            decoder_head_num=self.decoder_head_num,
            decoder_dense_num=self.decoder_dense_num,
            decoder_dropout_rate=self.decoder_dropout_rate,
            decoder_activation=self.decoder_activation,
            num_classes=self.num_classes,
            num_morph_classes=self.num_morph_classes,
            **self.factory_kwargs,
        )
        self.torch_encoder = self.torch_model.torch_encoder
        self.torch_decoder = self.torch_model.torch_decoder

        # ====================== DA heads ======================
        self.grl = GradientReversal(lam=1.0)

        # DANN S vs U
        self.domain_disc = DomainDiscriminator(
            in_dim=self.embedding_dim,
            hidden=max(128, self.embedding_dim),
            num_domains=2,
            device=device,
            dtype=dtype
        )
        # reweighter (MMD)
        self.reweighter  = ReweightMLP(
            in_dim=self.embedding_dim,
            hidden=max(128, self.embedding_dim//2),
            device=device,
            dtype=dtype
        )

        # DA hyperparams
        self.lambda_dom     = 0.0
        self.lambda_mmd     = 0.3
        self.entropy_coef   = 0.01



    def _save_internal(self, folder_name: str):
            import json
            import os

            if self.optimizer is None:
                raise ValueError("Optimizer is not initialized, please (re)-train the model first")
            
            os.makedirs(folder_name, exist_ok=True)


            with open(f"{folder_name}/config.json", "w") as f:
                json.dump(self.get_config(), f, indent=4)

            torch.save(
                {
                    "model_state_dict": self.torch_model.state_dict(),
                    "da_state_dict": {
                        "domain_disc": self.domain_disc.state_dict(),
                        "reweighter":  self.reweighter.state_dict(),
                    },
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "optimizer": self.optimizer.__class__.__name__,
                    "epoch": self.epoch,
                    "da_hparams": {
                        "lambda_dom":   self.lambda_dom,
                        "lambda_mmd":   self.lambda_mmd,
                        "entropy_coef": self.entropy_coef,
                    },
                },
                f"{folder_name}/weights.pt",
            )


    def load_compatible_weights(self, checkpoint_path: str, freeze_loaded: bool = False):
        """
        Carga pesos de un checkpoint permitiendo diferencias en la arquitectura 
        (ej. capas nuevas como LayerNorm).
        """
        print(f"🔄 Intentando cargar pesos compatibles desde: {checkpoint_path}")
        device = self.device
        
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
            
        pretrained_dict = ckpt["model_state_dict"]
        model_dict = self.torch_model.state_dict()
        
        compatible_dict = {}
        loaded_keys = []
        skipped_keys = []
        
        for k, v in pretrained_dict.items():
            k_clean = k.replace("_orig_mod.", "")
            
            if k_clean in model_dict:
                if model_dict[k_clean].shape == v.shape:
                    compatible_dict[k_clean] = v
                    loaded_keys.append(k_clean)
                else:
                    skipped_keys.append(f"{k_clean} (shape mismatch: ckpt {v.shape} vs model {model_dict[k_clean].shape})")
            else:
                pass

        missing_keys, unexpected_keys = self.torch_model.load_state_dict(compatible_dict, strict=False)
        
        print(f"✅ Se cargaron {len(compatible_dict)} tensores compatibles.")
        if len(missing_keys) > 0:
            print(f"⚠️ Capas nuevas inicializadas aleatoriamente (no estaban en ckpt): {missing_keys}")
        
        if freeze_loaded:
            print("❄️ Congelando capas cargadas...")
            for name, param in self.torch_model.named_parameters():
                if name in loaded_keys:
                    param.requires_grad = False
        else:
            print("🔥 Transfer Learning completo: Todos los pesos se mantendrán entrenables.")

        self._reinit_optimizer()
        
    def _reinit_optimizer(self):
        """Reinicia el optimizador para incluir cualquier cambio en parameters"""
        if self.optimizer is None: return
        base_lr = 1e-4 
        
        self.optimizer = torch.optim.AdamW([
            {"params": self.torch_model.parameters(), "lr": base_lr},
            {"params": self.domain_disc.parameters(), "lr": base_lr * 0.3},
            {"params": self.reweighter.parameters(),  "lr": base_lr * 0.3},
        ], fused=(self.device_type=="cuda"))


    def _load_internal(self, folder_name: str, **kwargs):
        map_location = kwargs.get("device", "cpu")

        try:
            ckpt = torch.load(f"{folder_name}/weights.pt", map_location=map_location, weights_only=True)
        except TypeError:
            ckpt = torch.load(f"{folder_name}/weights.pt", map_location=map_location)
            
        sd = ckpt["model_state_dict"]
        from collections import OrderedDict
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = OrderedDict((k.replace("_orig_mod.", "", 1), v) for k, v in sd.items())
        self.torch_model.load_state_dict(sd, strict=True)

        if "da_state_dict" in ckpt:
            da_sd = ckpt["da_state_dict"]
            if "domain_disc" in da_sd:
                self.domain_disc.load_state_dict(da_sd["domain_disc"], strict=False)
            if "reweighter" in da_sd:
                self.reweighter.load_state_dict(da_sd["reweighter"], strict=False)

        opt_class = getattr(torch.optim, ckpt.get("optimizer", "AdamW"))
        self.optimizer = opt_class(
            [
                {"params": self.torch_model.parameters()},
                {"params": self.domain_disc.parameters()},
                {"params": self.reweighter.parameters()},
            ]
        )
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                import warnings
                warnings.warn(f"Optimizer state_dict could not be fully loaded (continuing fresh): {repr(e)}")

        self.epoch = ckpt.get("epoch", 0)
        if "da_hparams" in ckpt:
            hp = ckpt["da_hparams"]
            self.lambda_dom   = hp.get("lambda_dom", self.lambda_dom)
            self.lambda_mmd   = hp.get("lambda_mmd", self.lambda_mmd)
            self.entropy_coef = hp.get("entropy_coef", self.entropy_coef)

    def get_parameters_sum(self):
        model_parameters = filter(lambda p: p.requires_grad,
                                list(self.torch_model.parameters())+
                                list(self.domain_disc.parameters())+
                                list(self.reweighter.parameters()))
        return sum([np.prod(p.size()) for p in model_parameters])

    # ---------------- Flash-only perception (no attention weights) ----------------
    def _perceive_internal(self, inputs, inputs_token, batch_size, return_attention_scores=False, inference_mode=True):
        if return_attention_scores:
            raise NotImplementedError("Attention weights are not available with Flash Attention.")
        self.torch_model.eval()
        with torch.inference_mode(mode=inference_mode):
            inputs_token = torch.as_tensor(inputs_token, device=self.factory_kwargs["device"], dtype=torch.int32)
            input_embedded = self.embedding_layer(
                inputs_token,
                torch.atleast_3d(torch.as_tensor(inputs, **self.factory_kwargs)),
            )
            padding_mask = torch.eq(inputs_token, torch.zeros_like(inputs_token))
            self._last_padding_mask = padding_mask
            data_length = len(inputs)
            num_batch = data_length // batch_size
            num_batch_remainder = data_length % batch_size
            if num_batch == 0:
                perception = self.torch_encoder(input_embedded, mask=padding_mask)
            else:
                with torch.autocast(device_type=self.device_type, enabled=self.mixed_precision):
                    perception = [self.torch_encoder(
                        input_embedded[i * batch_size : i * batch_size + batch_size],
                        mask=padding_mask[i * batch_size : i * batch_size + batch_size],
                    ) for i in range(num_batch)]
                if num_batch_remainder > 0:
                    perception.extend([self.torch_encoder(
                        input_embedded[num_batch * batch_size:],
                        mask=padding_mask[num_batch * batch_size:],
                    )])
                perception = torch.concat(perception)
            return perception, None


    def _request_internal(self, request_tokens, batch_size=64, return_attention_scores=False):
        if return_attention_scores:
            raise NotImplementedError("Attention weights are not available with Flash Attention.")
        self.torch_model.eval()
        with torch.inference_mode():
            data_length = len(self._perception_memory)
            num_outputs = len(request_tokens[0])
            try:
                class_label_token = self.tokenize(['SPECTYPE'])[0][0]
            except Exception:
                class_label_token = self.tokenize(['class_label'])[0][0]
            morph_token = self.tokenize(['MORPHTYPE'])[0][0]

            pred = np.full((data_length, num_outputs), np.nan)
            pred_err = np.full((data_length, num_outputs), np.nan)
            main_class_preds = np.full((data_length, num_outputs), -1, dtype=int)
            classification_probs = np.full((data_length, num_outputs, self.num_classes), np.nan)
            morph_preds = np.full((data_length, num_outputs, self.num_morph_classes), np.nan)

            perception = torch.as_tensor(self._perception_memory, **self.factory_kwargs)
            num_batches = (data_length + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, data_length)
                batch_size_actual = end_idx - start_idx
                batch_perception = perception[start_idx:end_idx]
                batch_mask = self._last_padding_mask[start_idx:end_idx]

                for i, token in enumerate(request_tokens[0]):
                    token_tensor = torch.tensor([token], device=self.factory_kwargs["device"], dtype=torch.int32)
                    unit_vector = self.embedding_layer(token_tensor)          # (1,1,E)
                    unit_vector = unit_vector.expand(batch_size_actual, -1, -1)  # (B,1,E)

                    _pred, _pred_logvar, _class_preds, _morph_preds = self.torch_decoder(
                        unit_vector, batch_perception, batch_mask
                    )
                    if token == class_label_token:
                        class_probs = torch.softmax(_class_preds.squeeze(dim=1), dim=-1).cpu().numpy()
                        class_pred = np.argmax(class_probs, axis=1)
                        main_class_preds[start_idx:end_idx, i] = class_pred
                        classification_probs[start_idx:end_idx, i, :] = class_probs
                    elif token == morph_token:
                        morph_prob = torch.softmax(_morph_preds.squeeze(dim=1), dim=-1).cpu().numpy()
                        morph_preds[start_idx:end_idx, i, :] = morph_prob
                    else:
                        _pred_np = _pred.detach().cpu().numpy()[:, 0, 0]
                        _pred_logvar_np = _pred_logvar.detach().cpu().numpy()[:, 0, 0]
                        pred[start_idx:end_idx, i] = _pred_np
                        pred_err[start_idx:end_idx, i] = np.sqrt(np.exp(_pred_logvar_np))

            return pred, pred_err, main_class_preds, morph_preds, classification_probs, None

    def fit(
                    self,
                    pseudo_batch_loader_S,
                    U_h5_path: str,
                    jpas_filter_names: List[str],
                    batch_size_S: int = 512,
                    batch_size_U: int = 2048,
                    epochs: int = 50,
                    z_loss_mode: str = "delta",
                    z_var_reg: float = 0.0,
                    lambda_dom: float = 0.3,
                    lambda_mmd: float = 0.3,
                    entropy_coef: float = 0.01,
                    lr_scheduler_factory=None,
                    checkpoint_every_n_epochs: int = 0,
                    terminate_on_nan: bool = True,
                    da_token_names: Optional[List[str]] = None,
                    apply_context_masking: bool = True,
                    observations_names: Optional[List[str]] = None,
                    target_names: Optional[List[str]] = None,
                    keep_frac_min: float = 0.5,
                    keep_frac_max: float = 1,
                    keep_min: int = 5,
                    weight_temp_start: float = 3.0,
                    weight_temp_end:   float = 1.0,
                    ess_target_frac:   float = 0.35,
                    ess_penalty:       float = 0.05,
                    clamp_mmd_nonneg:  bool  = True,
                    token_weights:     Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
                    n_tokens_predict:  int = 1, 
                    
                ) -> None:

                    import pathlib, time, tqdm, torch, numpy as np, pandas as pd
                    import torch.nn.functional as F
                    from torch.amp import GradScaler, autocast
                    from datetime import timedelta
                    
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

                    self.lambda_dom   = lambda_dom
                    self.lambda_mmd   = lambda_mmd
                    self.entropy_coef = entropy_coef

                    # ── Scheduler ──
                    if lr_scheduler_factory is None:
                        lr_scheduler = lambda opt: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            opt, T_0=1000, T_mult=1, eta_min=1e-10
                        )
                    else:
                        lr_scheduler = lr_scheduler_factory
                    scaler = GradScaler(enabled=(self.device_type == "cuda"))
                    scheduler = lr_scheduler(self.optimizer)

                    try:
                        class_tok = self.tokenize(['SPECTYPE'])[0][0]
                    except Exception:
                        class_tok = self.tokenize(['class_label'])[0][0]
                    morph_tok = self.tokenize(['MORPHTYPE'])[0][0]
                    z_gal_tok = self.tokenize(['Z_GAL'])[0][0]
                    z_qso_tok = self.tokenize(['Z_QSO'])[0][0]

                    if apply_context_masking:
                        if observations_names is None or target_names is None:
                            raise ValueError("apply_context_masking=True requiere observations_names y target_names.")
                        obs_lookup, tgt_lookup = self._build_token_lookups(observations_names, target_names)

                    raw_S_batches = []
                    print("⏳ Loading S data...")
                    for pb in pseudo_batch_loader_S():
                        raw_S_batches.append(pb)

                    if (da_token_names is None) or (len(da_token_names) == 0):
                        da_token_names = list(jpas_filter_names)
                    
                    print("⏳ Loading U data...")
                    def load_U_arrays(h5_path: str, token_names: List[str]):
                        use_cols = list(token_names) + [f"{t}_ERR" for t in token_names]
                        with pd.HDFStore(h5_path, 'r') as store:
                            df_full = store['data']
                            subset_cols = [c for c in use_cols if c in df_full.columns]
                            df = df_full[subset_cols].copy()
                        N = len(df)
                        Xvals = np.zeros((N, len(token_names)), dtype=np.float32)
                        Xerrs = np.zeros_like(Xvals, dtype=np.float32)
                        for j, t in enumerate(token_names):
                            Xvals[:, j] = df[t].to_numpy(dtype=np.float32) if t in df.columns else np.nan
                            err_col = f"{t}_ERR"
                            Xerrs[:, j] = df[err_col].to_numpy(dtype=np.float32) if err_col in df.columns else 0.0
                        return Xvals, Xerrs

                    U_vals, U_errs = load_U_arrays(U_h5_path, da_token_names)

                    print("⏳ Computing stats...")
                    def _prefit_and_freeze_stats():
                        import numpy as _np
                        from astropy.stats import mad_std as _mad
                        acc = {}
                        def _accumulate(values_2d: _np.ndarray, names_2d) -> None:
                            names_2d = _np.atleast_2d(names_2d)
                            col_names = _np.squeeze(names_2d[0])
                            toks = self.tokenize(col_names); toks = _np.squeeze(toks)
                            for j, t in enumerate(toks.tolist()):
                                if t == 0 or self.is_classification_token(int(t)): continue
                                col = values_2d[:, j].astype(float); col = col[_np.isfinite(col)]
                                if col.size == 0: continue
                                acc.setdefault(int(t), []).append(col)
                        for pb in raw_S_batches:
                            _accumulate(pb["training_labels"], pb["obs_names"])
                        if U_vals is not None and U_vals.size > 0 and len(da_token_names) > 0:
                            toks_U = self.tokenize(np.atleast_2d(da_token_names))[0]
                            for j, t in enumerate(toks_U.tolist()):
                                if t == 0 or self.is_classification_token(int(t)): continue
                                col = U_vals[:, j]; col = col[np.isfinite(col)]
                                if col.size == 0: continue
                                acc.setdefault(int(t), []).append(col)
                        for t, chunks in acc.items():
                            v = np.concatenate(chunks, axis=0)
                            if v.size == 0: mu, sg = 0.0, 1.0
                            else:
                                mu = float(np.nanmedian(v)); sg = float(_mad(v, ignore_nan=True))
                                if (not np.isfinite(sg)) or sg <= 0.0: sg = 1.0
                            self._input_mean[t] = mu; self._input_std[t] = sg; self._input_standardized[t] = True
                    _prefit_and_freeze_stats()

                    print("🚀 Initializing GPU Generators...")
                    self.train_generators = []
                    for pb in raw_S_batches:
                        std_x, x_tok, y_tok, std_err, _valid_mask = self._fit_checklist(
                            inputs=pb["training_labels"], inputs_name=pb["obs_names"], outputs_name=pb["obs_names"],
                            inputs_err=pb["training_labels_err"], update=False,
                        )
                        tg = TrainingGenerator(
                            batch_size=batch_size_S,
                            data={
                                "input": std_x, "input_token": x_tok, "output": std_x, "output_err": std_err,
                                "class_labels": pb["class_label"], "morph_labels": pb["morph_labels"],
                            },
                            data_probabilities=pb["inputs_probabilities"] / pb["inputs_probabilities"].sum(),
                            possible_output_tokens=y_tok, outputs_padding=self.context_length - 1, input_length=self.context_length,
                            factory_kwargs=self.factory_kwargs, class_label_token=class_tok, morph_label_token=morph_tok,
                            token_weights=token_weights,
                            n_tokens_predict=n_tokens_predict 
                        )
                        self.train_generators.append(tg)
                    del raw_S_batches

                    def build_U_tensors():
                        Btot = U_vals.shape[0]
                        names_epoch = np.tile(np.array(da_token_names), (Btot, 1))
                        tokens_epoch = self.tokenize(names_epoch, data_length=Btot)
                        vals_std, tokens_std, errs_std = self.standardize(U_vals, tokens_epoch, U_errs, update=False)
                        vals_pad, tokens_pad = self.padding(vals_std, tokens_std)
                        return (
                            torch.atleast_3d(torch.as_tensor(vals_pad, device=self.device, dtype=self.dtype)),
                            torch.as_tensor(tokens_pad, device=self.device, dtype=torch.int32),
                            torch.atleast_3d(torch.as_tensor(errs_std, device=self.device, dtype=self.dtype)),
                        )
                    xU_all, xU_tok_all, xU_err_all = build_U_tensors()
                    metaU_all = self.embedding_layer.gather_metadata(xU_tok_all)
                    NU = xU_all.shape[0]

                    # ── LOG HEADER ──
                    pathlib.Path(f"{self.root_folder}/checkpoints").mkdir(parents=True, exist_ok=True)
                    with open(f"{self.root_folder}/training.log", "a") as _lf:
                        _lf.write("="*80 + "\n")
                        _lf.write("FIT START (Fully Pre-Loaded GPU + JIT + Robust FP32 Losses)\n")
                        _lf.write(f"device={self.device} dtype={self.dtype} mixed_precision={self.mixed_precision}\n")
                        _lf.write(f"n_tokens_predict={n_tokens_predict}\n")
                        if apply_context_masking:
                            _lf.write(f"MASKING: on (keep_frac∈[{keep_frac_min:.2f},{keep_frac_max:.2f}])\n")

                    try: dom_crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
                    except TypeError: dom_crit = torch.nn.CrossEntropyLoss()

                    token_weights_tensor = None
                    if token_weights is not None:
                        if not isinstance(token_weights, torch.Tensor):
                            token_weights = torch.tensor(token_weights)
                        token_weights_tensor = token_weights.to(self.device, dtype=torch.float32)



                    # ── Start Epoch Loop ──
                    initial_epoch = getattr(self, "epoch", 0) or 0
                    for epoch in tqdm.tqdm(range(initial_epoch + 1, epochs + 1), desc="epoch"):
                        t0 = time.time()
                        self.epoch = epoch
                        self.torch_model.train(); self.domain_disc.train(); self.reweighter.train()

                        for tg in self.train_generators: tg.on_epoch_end()

                        agg = torch.zeros(6, device=self.device)
                        n_batches = 0
                        z_gal_sum = torch.zeros(1, device=self.device); z_qso_sum = torch.zeros(1, device=self.device)
                        z_gal_cnt = torch.zeros(1, device=self.device); z_qso_cnt = torch.zeros(1, device=self.device)
                        dz_gal_sum = torch.zeros(1, device=self.device); dz_qso_sum = torch.zeros(1, device=self.device)
                        ess_sum = torch.zeros(1, device=self.device); ess_n = 0
                        grad_norm_sum = torch.zeros(1, device=self.device); grad_norm_n = 0 
                        kept_ratio_S_acc = torch.zeros(1, device=self.device)
                        kept_ratio_U_acc = torch.zeros(1, device=self.device)
                        
                        all_dz_gal_list = []; all_dz_qso_list = []

                        p = float((epoch - 1) / max(epochs, 1))
                        dann_scale = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
                        dann_scale = 1.0
                        self.grl.lam = dann_scale
                        lam_dom_eff = self.lambda_dom * dann_scale; lam_mmd_eff = self.lambda_mmd * dann_scale
                        tau = weight_temp_start * (1.0 - p) + weight_temp_end * p
                        noise_std = 0.01


                        def _wrap_standard(pred, logvar, targ, terr, nu=3.0):
                                    with torch.autocast(device_type=self.device_type, enabled=False):
                                        pred = pred.float(); logvar = logvar.float(); targ = targ.float()
                                        terr = terr.float().clamp(max=100.0)
                                        
                                        LOGVAR_MIN = np.log(1e-4); LOGVAR_MAX = np.log(150)
                                        safe_logv  = logvar.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
                                        var = safe_logv.exp() + terr.pow(2) + 1e-12
                                        std = var.sqrt()
                                        
                                        z = (targ - pred) / std
                                        
                                         
                                        nll_term = 0.5 * (nu + 1) * torch.log(1 + (z.pow(2) / nu))
                                        log_scale_term = torch.log(std) 
                                        
                                        raw_loss = log_scale_term + nll_term
                                        # -----------------------------

                                        loss = torch.nan_to_num(raw_loss, nan=50.0, posinf=50.0, neginf=50.0).clamp(max=50.0)
                                        return loss


                        def _wrap_delta_for(sel_mask, tok_id, pred, logvar, targ, terr):
                            if (sel_mask.any().item() is False): return torch.tensor(0.0, device=self.device)
                            
                            with torch.autocast(device_type=self.device_type, enabled=False):
                                pred = pred.float(); logvar = logvar.float(); targ = targ.float()
                                terr = terr.float().clamp(max=100.0) 

                                z_mean = torch.as_tensor(self._input_mean[tok_id], device=self.device, dtype=torch.float32)
                                z_std  = torch.as_tensor(self._input_std[tok_id],  device=self.device, dtype=torch.float32)
                                sel = sel_mask.squeeze(1)
                                targ_u = targ[sel, :] * z_std + z_mean
                                pred_u = pred[sel, :] * z_std + z_mean
                                terr_u = terr[sel, :] * z_std
                                
                                LOGVAR_MIN = np.log(1e-4); LOGVAR_MAX = np.log(1e4)
                                logvar_u = logvar[sel, :].clamp(min=LOGVAR_MIN, max=LOGVAR_MAX) + 2.0*torch.log(z_std + 1e-12)
                                norm    = (1.0 + targ_u).clamp_min(1e-6) 
                                var_u   = logvar_u.exp() + terr_u.pow(2) + 1e-12
                                var_n   = var_u / (norm.pow(2) + 1e-12)
                                resid_n = (pred_u - targ_u) / norm
                                wrap    = 0.5 * (resid_n.pow(2) / var_n + var_n.log())
                                wrap = torch.nan_to_num(wrap, nan=50.0, posinf=50.0, neginf=50.0).clamp(max=50.0)                           
                                
                                if z_var_reg > 0: wrap = wrap + z_var_reg * (logvar_u**2)
                                return wrap.sum()

                        u_ptr = 0; u_perm = torch.randperm(NU, device=self.device)

                        for tg in self.train_generators:
                            for xS, xS_tok, y_tok_b, yS, yS_err, y_cls, y_morph, pS in tg:
                                self.optimizer.zero_grad(set_to_none=True)

                                if u_ptr + batch_size_U > NU:
                                    u_ptr = 0; u_perm = torch.randperm(NU, device=self.device)
                                sel = u_perm[u_ptr:u_ptr+batch_size_U]; u_ptr += batch_size_U
                                xU = xU_all[sel]; xU_tok = xU_tok_all[sel]

                                xS = torch.nan_to_num(xS, nan=0.0); xU = torch.nan_to_num(xU, nan=0.0)

                                if apply_context_masking:
                                    y_tok_for_masking = y_tok_b[:, 0:1] if y_tok_b.ndim > 1 else y_tok_b

                                    krS = self._apply_context_masking_S(
                                        xS, xS_tok, y_tok_for_masking, obs_lookup, tgt_lookup, 
                                        keep_frac_min, keep_frac_max, keep_min,
                                        token_weights=token_weights_tensor
                                    )
                                    krU = self._apply_context_masking_U(
                                        xU, xU_tok, None, obs_lookup, 
                                        keep_frac_min, keep_frac_max, keep_min,
                                        token_weights=token_weights_tensor
                                    )
                                    kept_ratio_S_acc += krS; kept_ratio_U_acc += krU

                                with autocast(device_type=self.device_type, enabled=self.mixed_precision):
                                    y_pred, y_logvar, cls_logits, morph_logits, encS = self.torch_model(xS, xS_tok, y_tok_b)
                                    
                                    padU = torch.eq(xU_tok, 0)
                                    if metaU_all: lamU, hasU, narU = metaU_all[0][sel], metaU_all[1][sel], metaU_all[2][sel]
                                    else: lamU = hasU = narU = None
                                    embU = self.embedding_layer(xU_tok, xU)
                                    encU_seq = self.torch_encoder(embU, mask=padU, relpos_lambda=lamU, relpos_has=hasU, relpos_narrow=narU)
                                    validU = (~padU).float(); denomU = validU.sum(dim=1, keepdim=True).clamp_min(1.0)
                                    encU = (encU_seq * validU.unsqueeze(-1)).sum(dim=1) / denomU

                                    if terminate_on_nan and ((not torch.isfinite(y_pred).all()) or (not torch.isfinite(y_logvar).all())):
                                        raise RuntimeError("NaN outputs in decoder")

                                    B_dim, N_dim = y_pred.shape[0], y_pred.shape[1]
                                    y_pred_flat = y_pred.reshape(-1, 1)        # [B*N, 1]
                                    y_logvar_flat = y_logvar.reshape(-1, 1)    # [B*N, 1]
                                    y_tok_flat = y_tok_b.reshape(-1)           # [B*N]
                                    
                                    yS_flat = yS.reshape(-1, 1)
                                    yS_err_flat = yS_err.reshape(-1, 1)

                                    is_reg = (y_tok_flat != class_tok) & (y_tok_flat != morph_tok)
                                    is_z_gal_row = (y_tok_flat == z_gal_tok).unsqueeze(1) # [B*N, 1]
                                    is_z_qso_row = (y_tok_flat == z_qso_tok).unsqueeze(1) # [B*N, 1]
                                    
                                    sum_non_z = torch.tensor(0.0, device=self.device); sum_z = torch.tensor(0.0, device=self.device)
                                    if is_reg.any():
                                        targ = yS_flat
                                        terr = yS_err_flat
                                        ws = _wrap_standard(y_pred_flat[is_reg,:], y_logvar_flat[is_reg,:], targ[is_reg,:], terr[is_reg,:])
                                        is_z_gal_reg = is_z_gal_row[is_reg, :]; is_z_qso_reg = is_z_qso_row[is_reg, :]
                                        sum_non_z = (ws * (~is_z_gal_reg & ~is_z_qso_reg).float()).sum()

                                        if z_loss_mode == "standard":
                                            sum_z = (ws * (is_z_gal_reg | is_z_qso_reg).float()).sum()
                                            z_gal_sum += (ws * is_z_gal_reg.float()).sum().detach()
                                            z_qso_sum += (ws * is_z_qso_reg.float()).sum().detach()
                                        else:
                                            wrap_gal = _wrap_delta_for(is_z_gal_row, z_gal_tok, y_pred_flat, y_logvar_flat, targ, terr).sum()
                                            wrap_qso = _wrap_delta_for(is_z_qso_row, z_qso_tok, y_pred_flat, y_logvar_flat, targ, terr).sum()
                                            sum_z = wrap_gal + wrap_qso
                                            z_gal_sum += wrap_gal.detach(); z_qso_sum += wrap_qso.detach()
                                    
                                    loss_reg = (sum_non_z + sum_z) / is_reg.sum().clamp_min(1.0)

                                with torch.autocast(device_type=self.device_type, enabled=False):
                                    cls_logits_32 = cls_logits.float()
                                    morph_logits_32 = morph_logits.float()
                                    

                                    cls_logits_2d = cls_logits_32.view(-1, cls_logits_32.shape[-1])
                                    is_cls = (y_tok_flat == class_tok)
                                    
                                    if is_cls.any():

                                        if y_cls is not None:
                                            # Expand: [B, 1] -> [B, N] -> [B*N]
                                            y_cls_expanded = y_cls.view(-1, 1).expand(B_dim, N_dim).reshape(-1)
                                            tgt = y_cls_expanded[is_cls].clamp(0, cls_logits_2d.size(1)-1).long()
                                            
                                            logits_cls_sel = cls_logits_2d[is_cls]
                                            loss_cls = F.cross_entropy(logits_cls_sel, tgt, reduction='sum').div(is_cls.sum().clamp_min(1.0))
                                        else:
                                            loss_cls = torch.tensor(0.0, device=self.device)
                                    else: loss_cls = torch.tensor(0.0, device=self.device)

                                    morph_logits_2d = morph_logits_32.view(-1, morph_logits_32.shape[-1])
                                    is_morph = (y_tok_flat == morph_tok)
                                    if is_morph.any():
                                        if y_morph is not None:
                                            y_morph_expanded = y_morph.view(-1, 1).expand(B_dim, N_dim).reshape(-1)
                                            tgt = y_morph_expanded[is_morph].clamp(0, morph_logits_2d.size(1)-1).long()
                                            
                                            logits_morph_sel = morph_logits_2d[is_morph]
                                            loss_morph = F.cross_entropy(logits_morph_sel, tgt, reduction='sum').div(is_morph.sum().clamp_min(1.0))
                                        else:
                                            loss_morph = torch.tensor(0.0, device=self.device)
                                    else: loss_morph = torch.tensor(0.0, device=self.device)

                                    hS = encS + noise_std * torch.randn_like(encS); hU = encU + noise_std * torch.randn_like(encU)
                                    logits_S = self.domain_disc(self.grl(hS)).float()
                                    logits_U = self.domain_disc(self.grl(hU)).float()
                                    
                                    loss_dom = dom_crit(torch.cat([logits_S, logits_U]), torch.cat([torch.zeros(hS.size(0), dtype=torch.long, device=self.device), torch.ones(hU.size(0), dtype=torch.long, device=self.device)]))

                                with autocast(device_type=self.device_type, enabled=self.mixed_precision):
                                    w_raw = self.reweighter(hU)
                                    if terminate_on_nan and not torch.isfinite(w_raw).all(): raise RuntimeError("NaN wU")
                                    
                                    wU = torch.softmax(torch.log(torch.clamp(torch.nan_to_num(w_raw), min=1e-5)) / max(tau, 1e-6), dim=0)
                                    sigma_dynamic = _median_heuristic_sigma(torch.cat([hS.detach(), hU], dim=0))
                                    mmd_raw = rbf_mmd2_weighted(hS.detach(), hU, w_x=None, w_y=wU, sigma=sigma_dynamic)

                                    ess_val = (wU.sum()**2) / (wU.pow(2).sum() + 1e-5)
                                    ess_target = torch.as_tensor(ess_target_frac * hU.size(0), device=self.device)
                                    term_ess = ess_penalty * torch.relu(ess_target - ess_val).pow(2) / (ess_target + 1e-5)
                                    term_entropy = self.entropy_coef * (wU.pow(2).sum() / (hU.size(0) + 1e-5))
                                    loss_mmd = (torch.relu(mmd_raw) if clamp_mmd_nonneg else mmd_raw) + term_entropy + term_ess

                                    total_loss = 2*loss_reg + 2*loss_cls + 2*loss_morph + lam_dom_eff*loss_dom + lam_mmd_eff*loss_mmd

                                if terminate_on_nan and not torch.isfinite(total_loss): continue

                                scaler.scale(total_loss).backward()
                                scaler.unscale_(self.optimizer)
                                
                                gn = torch.nn.utils.clip_grad_norm_(list(self.torch_model.parameters())+list(self.domain_disc.parameters())+list(self.reweighter.parameters()), 5.0)
                                
                                if torch.isfinite(gn):
                                    grad_norm_sum += gn.detach(); grad_norm_n += 1
                                
                                scaler.step(self.optimizer); scaler.update()

                                agg += torch.stack([total_loss, loss_reg, loss_cls, loss_morph, loss_dom, loss_mmd]).detach()
                                n_batches += 1
                                ess = (wU.sum()**2)/(wU.pow(2).sum()+1e-12); ess_sum += ess.detach(); ess_n += 1
                                z_gal_cnt += is_z_gal_row.float().sum().detach(); z_qso_cnt += is_z_qso_row.float().sum().detach()
                                
                                with torch.no_grad():
                                    def _dz(m, t): 
                                        if not m.any(): return torch.tensor(0.0, device=self.device), None
                                        zm = torch.as_tensor(self._input_mean[t], device=self.device); zs = torch.as_tensor(self._input_std[t], device=self.device)
                                        sl = m.squeeze(1)
                                        pu = y_pred_flat[sl,:]*zs+zm; tu = yS_flat[sl,:]*zs+zm
                                        raw_dz = (pu-tu)/(1.0+tu).clamp_min(1e-12)
                                        return raw_dz.abs().sum(), raw_dz.detach().reshape(-1)

                                    dz_g, raw_g = _dz(is_z_gal_row, z_gal_tok); dz_gal_sum += dz_g
                                    dz_q, raw_q = _dz(is_z_qso_row, z_qso_tok); dz_qso_sum += dz_q
                                    if raw_g is not None: all_dz_gal_list.append(raw_g)
                                    if raw_q is not None: all_dz_qso_list.append(raw_q)

                        avg = (agg / max(n_batches,1)).tolist()
                        lr = self.optimizer.param_groups[0]['lr']
                        dt = str(timedelta(seconds=time.time()-t0))
                        
                        all_dz_gal = torch.cat(all_dz_gal_list).float().cpu().numpy() if all_dz_gal_list else np.array([])
                        all_dz_qso = torch.cat(all_dz_qso_list).float().cpu().numpy() if all_dz_qso_list else np.array([])

                        z_gal_avg = (z_gal_sum / z_gal_cnt.clamp_min(1.0)).item()
                        z_qso_avg = (z_qso_sum / z_qso_cnt.clamp_min(1.0)).item()
                        dz_gal_avg = (dz_gal_sum / z_gal_cnt.clamp_min(1.0)).item()
                        dz_qso_avg = (dz_qso_sum / z_qso_cnt.clamp_min(1.0)).item()
                        ess_avg = (ess_sum / max(ess_n,1)).item()
                        kept_S_avg = (kept_ratio_S_acc / max(n_batches,1)).item()
                        kept_U_avg = (kept_ratio_U_acc / max(n_batches,1)).item()
                        grad_norm_avg = (grad_norm_sum / max(grad_norm_n,1)).item() 

                        with open(f"{self.root_folder}/training.log", "a") as log_f:
                            extra = f" | keepS~{kept_S_avg:.3f} keepU~{kept_U_avg:.3f}" if apply_context_masking else ""
                            
                            log_f.write(
                                f"Epoch {epoch}: loss={avg[0]:.4f} (reg={avg[1]:.4f} cls={avg[2]:.4f} morph={avg[3]:.4f}) "
                                f"lr={lr:.2e} time={dt} | z_gal={z_gal_avg:.4f} z_qso={z_qso_avg:.4f} "
                                f"dz1p_gal={dz_gal_avg:.4f} dz1p_qso={dz_qso_avg:.4f} nz_gal={int(z_gal_cnt.item())} nz_qso={int(z_qso_cnt.item())} "
                                f"| dom={avg[4]:.4f} mmd={avg[5]:.4f} essU={ess_avg:.1f} grad_norm={grad_norm_avg:.3f}{extra}\n"
                            )
                            
                            def _stats(dz):
                                if dz.size==0: return 0.0, 0.0
                                return float(np.nanmedian(dz)), float(1.48*np.nanmedian(np.abs(dz)))
                            bg, sg = _stats(all_dz_gal); bq, sq = _stats(all_dz_qso)
                            log_f.write(f"  STATS: GAL(bias={bg:.4f}, sig={sg:.4f}) QSO(bias={bq:.4f}, sig={sq:.4f})\n")

                        with open(f"{self.root_folder}/training_metrics.csv", "a") as csv_f:
                            csv_f.write(f"{epoch},{avg[0]:.6e},{avg[1]:.6e},{avg[2]:.6e},{avg[3]:.6e},{lr:.6e},{z_gal_avg:.6e},{z_qso_avg:.6e},{dz_gal_avg:.6e},{dz_qso_avg:.6e},{int(z_gal_cnt.item())},{int(z_qso_cnt.item())},{avg[4]:.6e},{avg[5]:.6e},{ess_avg:.3f}\n")

                        scheduler.step()
                        
                        if checkpoint_every_n_epochs and (epoch % checkpoint_every_n_epochs == 0):
                            ckpt_folder = f"{self.root_folder}/checkpoints/epoch_{epoch}"
                            pathlib.Path(ckpt_folder).mkdir(parents=True, exist_ok=True)
                            self._save_internal(ckpt_folder)


    def _apply_context_masking_S(
                self,
                xS: torch.Tensor,              
                xS_tok: torch.Tensor,          
                y_tok_b: torch.Tensor,         
                obs_lookup: torch.Tensor,      
                tgt_lookup: torch.Tensor,      
                keep_frac_min: float,
                keep_frac_max: float,
                keep_min:   int,
                token_weights: Optional[torch.Tensor] = None, 
            ) -> torch.Tensor:

                B, S = xS_tok.shape
                tok = xS_tok 

                present = tok != 0                           
                y = y_tok_b.view(-1)                         
                is_target = tok == y.unsqueeze(1)            

                allow_all = (obs_lookup | tgt_lookup)        
                allow_obs = obs_lookup

                target_in_obs = allow_obs[y]                 
                allowed_all_mask = allow_all[tok.long()]     
                allowed_obs_mask = allow_obs[tok.long()]     
                allowed = torch.where(target_in_obs.unsqueeze(1), allowed_all_mask, allowed_obs_mask)  

                candidates = present & (~is_target) & allowed   
                n_cand = candidates.sum(dim=1)                  

                if (n_cand == 0).all():
                    return torch.tensor(0.0, device=self.device)

                base_keep_frac = torch.empty((B, 1), device=self.device, dtype=torch.float32).uniform_(keep_frac_min, keep_frac_max)
                rnd = torch.rand((B, S), device=self.device, dtype=torch.float32)
                

                if token_weights is not None:
                    raw_weights = token_weights[tok.long()] 
                    

                    masked_weights = raw_weights * candidates.float()
                    sum_weights = masked_weights.sum(dim=1, keepdim=True)
                    count_cand = candidates.sum(dim=1, keepdim=True).clamp_min(1.0)
                    mean_weight = sum_weights / count_cand


                    survival_boost = raw_weights / mean_weight.clamp_min(1e-6)
                    
                    effective_keep_thresh = base_keep_frac * survival_boost
                    
                    keep1 = (rnd < effective_keep_thresh) & candidates
                else:
                    keep1 = (rnd < base_keep_frac) & candidates    

                kept_counts = keep1.sum(dim=1)            
                need_more = kept_counts < torch.minimum(n_cand, torch.tensor(keep_min, device=self.device))

                if need_more.any():
                    if token_weights is not None:

                        rescue_score = token_weights[tok.long()]
                        rescue_score = torch.where(candidates & (~keep1), rescue_score, torch.tensor(-1.0, device=self.device))
                    else:
                        rnd_masked = torch.where(candidates, rnd, torch.full_like(rnd, 2.0))
                        rescue_score = -rnd_masked 

                    k = min(keep_min, S) 
                    topk_idx = torch.topk(rescue_score, k=k, dim=1).indices  
                    
                    extra = torch.zeros_like(candidates, dtype=torch.bool)
                    rows = torch.arange(B, device=self.device).unsqueeze(1).expand_as(topk_idx)
                    extra[rows, topk_idx] = True
                    
                    valid_extra = (rescue_score.gather(1, topk_idx) > -0.5)
                    extra[rows[~valid_extra], topk_idx[~valid_extra]] = False
                    
                    need_more_mask = need_more.unsqueeze(1).expand_as(extra)
                    keep1 = torch.where(need_more_mask, keep1 | extra, keep1)

                drop = present & (~keep1)
                xS_tok.masked_fill_(drop, 0)

                kept_final = (keep1 & candidates).sum(dim=1).float()
                ratio = torch.where(n_cand > 0, kept_final / n_cand.clamp_min(1), torch.zeros_like(kept_final))
                return ratio.mean()

    def _apply_context_masking_U(
                self,
                xU: torch.Tensor,              
                xU_tok: torch.Tensor,          
                xU_err: Optional[torch.Tensor],
                obs_lookup: torch.Tensor,      
                keep_frac_min: float,
                keep_frac_max: float,
                keep_min:   int,
                token_weights: Optional[torch.Tensor] = None, 
            ) -> torch.Tensor:

                B, S = xU_tok.shape
                tok = xU_tok 

                present = tok != 0
                allowed = obs_lookup[tok.long()]
                candidates = present & allowed
                n_cand = candidates.sum(dim=1)

                if (n_cand == 0).all():
                    return torch.tensor(0.0, device=self.device)

                base_keep_frac = torch.empty((B, 1), device=self.device, dtype=torch.float32).uniform_(keep_frac_min, keep_frac_max)
                rnd = torch.rand((B, S), device=self.device, dtype=torch.float32)


                if token_weights is not None:
                    raw_weights = token_weights[tok.long()]
                    
                    masked_weights = raw_weights * candidates.float()
                    sum_weights = masked_weights.sum(dim=1, keepdim=True)
                    count_cand = candidates.sum(dim=1, keepdim=True).clamp_min(1.0)
                    mean_weight = sum_weights / count_cand
                    
                    survival_boost = raw_weights / mean_weight.clamp_min(1e-6)
                    effective_keep_thresh = base_keep_frac * survival_boost
                    
                    keep1 = (rnd < effective_keep_thresh) & candidates
                else:
                    keep1 = (rnd < base_keep_frac) & candidates

                kept_counts = keep1.sum(dim=1)
                need_more = kept_counts < torch.minimum(n_cand, torch.tensor(keep_min, device=self.device))

                if need_more.any():
                    if token_weights is not None:
                        rescue_score = token_weights[tok.long()]
                        rescue_score = torch.where(candidates & (~keep1), rescue_score, torch.tensor(-1.0, device=self.device))
                    else:
                        rnd_masked = torch.where(candidates, rnd, torch.full_like(rnd, 2.0))
                        rescue_score = -rnd_masked

                    k = min(keep_min, S)
                    topk_idx = torch.topk(rescue_score, k=k, dim=1).indices
                    
                    extra = torch.zeros_like(candidates, dtype=torch.bool)
                    rows = torch.arange(B, device=self.device).unsqueeze(1).expand_as(topk_idx)
                    extra[rows, topk_idx] = True
                    
                    valid_extra = (rescue_score.gather(1, topk_idx) > -0.5)
                    extra[rows[~valid_extra], topk_idx[~valid_extra]] = False

                    need_more_mask = need_more.unsqueeze(1).expand_as(extra)
                    keep1 = torch.where(need_more_mask, keep1 | extra, keep1)

                drop = present & (~keep1)
                xU_tok.masked_fill_(drop, 0)
                if xU_err is not None:
                    xU_err.masked_fill_(drop.unsqueeze(-1), 0.0)

                kept_final = (keep1 & candidates).sum(dim=1).float()
                ratio = torch.where(n_cand > 0, kept_final / n_cand.clamp_min(1), torch.zeros_like(kept_final))
                return ratio.mean()




    def _build_token_lookups(self, observations_names: List[str], target_names: List[str]):
        """Devuelve (obs_lookup, tgt_lookup) tensores booleanos indexados por id de token."""
        import numpy as np, torch
        obs_toks = np.squeeze(self.tokenize(np.atleast_2d(observations_names))[0]).astype(int).tolist()
        tgt_toks = np.squeeze(self.tokenize(np.atleast_2d(target_names))[0]).astype(int).tolist()
        vmax = int(max(self.vocab_tokens)) if len(self.vocab_tokens) else 0
        obs_lookup = torch.zeros(vmax + 1, dtype=torch.bool, device=self.device)
        tgt_lookup = torch.zeros(vmax + 1, dtype=torch.bool, device=self.device)
        for t in obs_toks:
            if t > 0: obs_lookup[t] = True
        for t in tgt_toks:
            if t > 0: tgt_lookup[t] = True
        return obs_lookup, tgt_lookup

