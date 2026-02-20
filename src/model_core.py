# model_core.py
import os
import json
import copy
import pathlib
import warnings
import numpy as np
from astropy.stats import mad_std
from abc import ABC, abstractmethod
import torch
from typing import List, Optional, Tuple, Union, Dict, Any
from numpy.typing import NDArray


class TranformerCore(ABC):


    def __init__(
        self,
        vocabs: List[str],
        backend_framewoark: str,
        vocab_tokens: List[int] = None,
        context_length: int = 30,
        embedding_dim: int = 16,
        embedding_activation: str = None,
        encoder_head_num: int = 2,
        encoder_dense_num: int = 128,
        encoder_dropout_rate: float = 0.1,
        encoder_activation: str = None,
        decoder_head_num: int = 2,
        decoder_dense_num: int = 128,
        decoder_dropout_rate: float = 0.1,
        decoder_activation: str = None,
        device: str = None,
        dtype=None,
        mixed_precision: bool = False,
        folder: str = "model_torch",
        built: bool = False,
    ) -> None:
        self._built = built
        self.backend_framewoark = backend_framewoark

        # vocabs → lista única
        if isinstance(vocabs, np.ndarray):
            self.vocabs = vocabs.tolist()
        else:
            self.vocabs = list(vocabs)
        if len(self.vocabs) != len(set(self.vocabs)):
            raise ValueError("Vocabs are not unique!")

        self.vocab_size = len(self.vocabs)

        if vocab_tokens is None:
            self.vocab_tokens = [i for i in range(1, self.vocab_size + 1)]
        else:
            self.vocab_tokens = list(vocab_tokens)

        vmax = int(np.max(self.vocab_tokens)) if len(self.vocab_tokens) else 0
        self._input_mean = np.zeros(vmax + 1, dtype=float)
        self._input_std = np.ones(vmax + 1, dtype=float)
        self._input_standardized = np.zeros(vmax + 1, dtype=bool)

        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.embedding_activation = embedding_activation
        self.encoder_head_num = encoder_head_num
        self.encoder_dense_num = encoder_dense_num
        self.encoder_dropout_rate = encoder_dropout_rate
        self.encoder_activation = encoder_activation
        self.decoder_head_num = decoder_head_num
        self.decoder_dense_num = decoder_dense_num
        self.decoder_dropout_rate = decoder_dropout_rate
        self.decoder_activation = decoder_activation

        self.epochs = None
        self.epoch = None
        self.loss = None
        self.val_loss = None
        self.learning_rate = None
        self.optimizer = None
        self.metrics = None

        self._perception_memory = None
        self._last_padding_mask = None

        if "torch" in backend_framewoark:
            self.device = device
            self.dtype = dtype
            if "cpu" in str(self.device):
                if mixed_precision:
                    warnings.warn("Mixed precision is not supported on CPU")
                    self.mixed_precision = False
                else:
                    self.mixed_precision = mixed_precision
            else:
                self.mixed_precision = mixed_precision

        self.root_folder = os.path.abspath(folder)
        folder_path = pathlib.Path(self.root_folder)
        if not self._built:
            if folder_path.exists():
                raise FileExistsError(
                    f"Model folder at {self.root_folder} already existed, please rename it"
                )
            else:
                folder_path.mkdir(parents=True)

        self.system_info = {}

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def _perceive_internal(self):
        raise NotImplementedError

    @abstractmethod
    def _request_internal(self):
        raise NotImplementedError

    @abstractmethod
    def _load_internal(self):
        raise NotImplementedError

    @abstractmethod
    def _save_internal(self):
        raise NotImplementedError

    def _built_only(self):
        if not self._built:
            raise NotImplementedError("This model is not trained")

    def clear_perception(self) -> None:
        del self._perception_memory
        self._perception_memory = None

    def _perception_check(self, mode: int) -> None:
        if self._perception_memory is None and mode == 1:
            raise ValueError("You did not setup a perception, so can't continue")
        elif self._perception_memory is not None and mode == 2:
            warnings.warn("Existing perception memory will be reset")
            self.clear_perception()

    def is_classification_token(self, token) -> bool:
        try:
            class_label_token = self.tokenize(['SPECTYPE'])[0][0]
        except Exception:
            class_label_token = self.tokenize(['class_label'])[0][0]
        morph_label_token = self.tokenize(['MORPHTYPE'])[0][0]
        return token == class_label_token or token == morph_label_token

    # ──────────────────────── checklist para fit ─────────────────────────

    def _fit_checklist(
        self,
        inputs: NDArray,
        inputs_name: Union[NDArray, list],
        outputs_name: List[str],
        inputs_err: Optional[NDArray] = None,
        update: bool = True,  
    ) -> Tuple[NDArray, NDArray, NDArray, Optional[NDArray], NDArray]:

        inputs_name = np.asarray(inputs_name)
        if inputs_name.ndim == 2 and not np.all(
            [len(set(inputs_name[:, idx])) == 1 for idx in range(inputs_name.shape[1])]
        ):
            raise ValueError("Inputs must be ordered; they cannot be pre-randomized.")

        additional_padding_needed = self.context_length - inputs.shape[1]
        if additional_padding_needed > 0:
            warnings.warn(
                f"Input width={inputs.shape[1]} is smaller than context width {self.context_length}; padding will be added."
            )
            inputs = self.end_zero_padding(inputs, additional_padding_needed)
            inputs_err = self.end_zero_padding(inputs_err, additional_padding_needed) if inputs_err is not None else None
            inputs_name = self.end_zero_padding(inputs_name, additional_padding_needed)

        data_length = len(inputs)

        inputs_token = self.tokenize(inputs_name, data_length=data_length)
        standardized_inputs, inputs_token, standardized_inputs_err = self.standardize(
            inputs, inputs_token, inputs_err, update=update
        )

        valid_mask = (inputs_token != 0)

        outputs_tokens = self.tokenize(outputs_name)[0]
        self._built = True
        return (standardized_inputs, inputs_token, outputs_tokens, standardized_inputs_err, valid_mask)

    def _tokenize_core_logic(self, one_str: str) -> int:
        if one_str not in self.vocabs:
            raise NameError(f"'{one_str}' is not among vocabs: {self.vocabs}")
        return 0 if one_str == "[pad]" else self.vocab_tokens[self.vocabs.index(one_str)]

    def tokenize(
        self,
        names: Union[List[Union[str, int]], NDArray[Union[np.str_, np.integer]]],
        data_length: Optional[int] = None,
    ) -> NDArray[np.integer]:

        names = np.atleast_2d(names)
        if data_length is None:
            data_length = len(names)

        out_tokens = np.zeros(names.shape, dtype=int)

        if np.issubdtype(names.dtype, np.integer):
            out_tokens = names.astype(int)
            if data_length != len(names):
                out_tokens = np.tile(out_tokens, (data_length, 1))
            return out_tokens

        nice_order = np.all([len(set(i)) == 1 for i in names.T])
        if nice_order or (data_length != len(names)):
            _temp_names = names[0] if (nice_order and data_length == len(names)) else names[0]
            row = [self._tokenize_core_logic(i) for i in np.atleast_1d(np.squeeze(_temp_names))]
            out_tokens = np.tile(np.asarray(row, dtype=int), (data_length, 1))
        else:
            for i in np.unique(names):
                idx = (names == i)
                out_tokens[idx] = self._tokenize_core_logic(i)

        return out_tokens

    def detokenize(self, tokens: NDArray) -> NDArray:
        tokens = np.asarray(tokens)
        out = np.empty(tokens.shape, dtype=object)
        for idx in np.ndindex(tokens.shape):
            t = int(tokens[idx])
            if t <= 0:
                out[idx] = "[pad]"
            else:
                out[idx] = self.vocabs[t - 1]
        return out

    def end_zero_padding(self, x: NDArray, n: int = 0) -> NDArray:
        if x is None:
            return None
        if n < 0:
            raise ValueError("'n' must be non-negative")
        return np.pad(x, ((0, 0), (0, n)), mode="constant", constant_values=0)

    def padding(
        self,
        inputs: NDArray[Union[np.float32, np.float64]],
        inputs_token: NDArray[np.integer],
    ) -> Tuple[NDArray[Union[np.float32, np.float64]], NDArray[np.integer]]:
        assert inputs.shape == inputs_token.shape, "Input and token shapes must match."
        additional_padding_needed = self.context_length - inputs.shape[1]
        if additional_padding_needed < 0:
            raise ValueError(f"Input width is larger than context width {self.context_length}")
        padded_inputs = self.end_zero_padding(inputs, additional_padding_needed)
        padded_inputs_token = self.end_zero_padding(inputs_token, additional_padding_needed)
        return padded_inputs, padded_inputs_token



    def standardize(
        self,
        inputs: NDArray[Union[np.float32, np.float64]],
        inputs_token: NDArray[np.int_],
        inputs_error: Optional[NDArray] = None,
        dtype: np.dtype = np.float32,
        update: bool = False,  
    ) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """
        - If update=True  (training): robustly fit per-token (median, MAD) on this data
        and store them in self._input_mean/_std/_standardized.
        - If update=False (inference): DO NOT change stored stats; apply the saved ones.
        If a token has no saved stats, compute (μ, σ) on-the-fly for this call only,
        but do not persist.

        Padding (token=0) and classification tokens are never standardized.
        Non-finite inputs are converted to padding (token=0, value=0, err=0).
        """
        _inputs = copy.deepcopy(inputs).astype(dtype, copy=False)
        _inputs_token = self.tokenize(copy.deepcopy(inputs_token))
        _inputs_error = copy.deepcopy(inputs_error)
        if _inputs_error is not None:
            _inputs_error = _inputs_error.astype(dtype, copy=False)

        # Non-finite → padding
        not_finite = ~np.isfinite(_inputs)
        if np.any(not_finite):
            _inputs[not_finite] = 0.0
            _inputs_token = np.where(not_finite, 0, _inputs_token)
            if _inputs_error is not None:
                _inputs_error[not_finite] = 0.0

        unique_tokens = np.unique(_inputs_token)
        for t in unique_tokens:
            if t == 0 or self.is_classification_token(t):
                continue

            mask = (_inputs_token == t)
            if not np.any(mask):
                continue

            # Decide which stats to use
            mu_apply: float
            sigma_apply: float

            if update or (not bool(self._input_standardized[t])):
                # Fit robust stats on this batch
                vals = _inputs[mask]
                vals_finite = vals[np.isfinite(vals)]
                if vals_finite.size == 0:
                    mu = 0.0
                    sigma = 1.0
                else:
                    mu = float(np.nanmedian(vals_finite))
                    sigma = float(mad_std(vals_finite, ignore_nan=True))
                    if not np.isfinite(sigma) or sigma <= 0.0:
                        sigma = 1.0

                # Persist only when update=True (i.e., training time)
                if update:
                    self._input_mean[t] = mu
                    self._input_std[t] = sigma
                    self._input_standardized[t] = True

                # Apply (use fitted here; for non-updating case, do not persist)
                mu_apply = mu if not bool(self._input_standardized[t]) else float(self._input_mean[t])
                sigma_apply = sigma if not bool(self._input_standardized[t]) else float(self._input_std[t])
            else:
                # Already have saved stats → just apply
                mu_apply = float(self._input_mean[t])
                sigma_apply = float(self._input_std[t])

            if not np.isfinite(sigma_apply) or sigma_apply == 0.0:
                sigma_apply = 1.0

            _inputs[mask] = (_inputs[mask] - mu_apply) / sigma_apply
            if _inputs_error is not None:
                _inputs_error[mask] = _inputs_error[mask] / sigma_apply

        return _inputs.astype(dtype, copy=False), _inputs_token.astype(int, copy=False), (
            _inputs_error.astype(dtype, copy=False) if _inputs_error is not None else None
        )


    def inverse_standardize(
        self,
        inputs: NDArray[Union[np.float32, np.float64]],
        inputs_token: NDArray[np.int_],
        inputs_error: Optional[NDArray[Union[np.float32, np.float64]]] = None,
        dtype: np.dtype = np.float32,
    ) -> Tuple[NDArray, ...]:
        """
        Deshace la estandarización por token (excepto padding y tokens de clasificación).
        x_orig = x_std * σ + μ
        err_orig = err_std * σ
        """
        _inputs = copy.deepcopy(inputs).astype(dtype, copy=False)
        _inputs_token = self.tokenize(copy.deepcopy(inputs_token))
        _inputs_error = copy.deepcopy(inputs_error)
        if _inputs_error is not None:
            _inputs_error = _inputs_error.astype(dtype, copy=False)

        # si inputs es (B,L) y tokens es (L,) → tile
        if _inputs.ndim == 2 and np.squeeze(_inputs_token).ndim == 1:
            _inputs_token = np.tile(_inputs_token, (len(_inputs), 1))

        unique_tokens = np.unique(_inputs_token)
        for t in unique_tokens:
            if t == 0 or self.is_classification_token(t):
                continue
            mu = float(self._input_mean[t]) if np.isfinite(self._input_mean[t]) else 0.0
            sigma = float(self._input_std[t]) if np.isfinite(self._input_std[t]) and self._input_std[t] != 0.0 else 1.0
            mask = (_inputs_token == t)
            if not np.any(mask):
                continue
            _inputs[mask] = _inputs[mask] * sigma + mu
            if _inputs_error is not None:
                _inputs_error[mask] = _inputs_error[mask] * sigma

        if _inputs_error is None:
            return (_inputs.astype(dtype, copy=False),)
        else:
            return (_inputs.astype(dtype, copy=False), _inputs_error.astype(dtype, copy=False))

    def _set_standardization(self, means: NDArray, stddev: NDArray, names: List[str]) -> None:
        tokens = self.tokenize(names, data_length=1)[0]
        for idx, t in enumerate(tokens):
            self._input_mean[t] = means[idx]
            self._input_std[t] = stddev[idx]
            self._input_standardized[t] = True

    # ───────────────────────────── persistencia ─────────────────────────────
    def get_config(self) -> Dict[str, Any]:
        nn_config = {
            "backend_framewoark": self.backend_framewoark,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "embedding_activation": self.embedding_activation,
            "encoder_head_num": self.encoder_head_num,
            "encoder_dense_num": self.encoder_dense_num,
            "encoder_dropout_rate": self.encoder_dropout_rate,
            "encoder_activation": self.encoder_activation,
            "decoder_head_num": self.decoder_head_num,
            "decoder_dense_num": self.decoder_dense_num,
            "decoder_dropout_rate": self.decoder_dropout_rate,
            "decoder_activation": self.decoder_activation,
        }
        tokenizer_config = {"vocabs": self.vocabs, "vocab_tokens": self.vocab_tokens}
        norm_config = {
            "_input_mean": self._input_mean.tolist(),
            "_input_std": self._input_std.tolist(),
            "_input_standardized": self._input_standardized.tolist(),
        }

        return {"nn_config": nn_config, "tokenizer_config": tokenizer_config, "norm_config": norm_config}

    def save(self, folder_name: str = "model") -> None:
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
        json_path = f"{folder_name}/config.json"
        with open(json_path, "w") as f:
            json.dump(self.get_config(), f, indent=4)
        self._save_internal(folder_name)

    @classmethod
    def load(cls, folder_name: str, checkpoint_epoch: int = -1, mixed_precision: bool = False, device: str = "cpu"):
        if checkpoint_epoch != -1:
            folder_name = f"{folder_name}/checkpoints/epoch_{checkpoint_epoch}"
            if not os.path.exists(folder_name):
                raise FileNotFoundError(f"Checkpoint at epoch {checkpoint_epoch} not found!")
        elif checkpoint_epoch < -1:
            raise ValueError("checkpoint_epoch must be >= 0")

        if not os.path.exists(folder_name):
            raise FileNotFoundError
        else:
            with open(f"{folder_name}/config.json", "r") as f:
                config = json.load(f)

        nn = cls(
            vocabs=np.array(config["tokenizer_config"]["vocabs"]),
            vocab_tokens=config["tokenizer_config"]["vocab_tokens"],
            context_length=config["nn_config"]["context_length"],
            embedding_dim=config["nn_config"]["embedding_dim"],
            embedding_activation=config["nn_config"]["embedding_activation"],
            encoder_head_num=config["nn_config"]["encoder_head_num"],
            encoder_dense_num=config["nn_config"]["encoder_dense_num"],
            encoder_dropout_rate=config["nn_config"]["encoder_dropout_rate"],
            encoder_activation=config["nn_config"]["encoder_activation"],
            decoder_head_num=config["nn_config"]["decoder_head_num"],
            decoder_dense_num=config["nn_config"]["decoder_dense_num"],
            decoder_dropout_rate=config["nn_config"]["decoder_dropout_rate"],
            decoder_activation=config["nn_config"]["decoder_activation"],
            device=device,
            mixed_precision=mixed_precision,
            folder=folder_name,
            built=True,
        )

        assert (nn.implemented_backend in config["nn_config"]["backend_framewoark"]), (
            f"You are loading a model trained with '{config['nn_config']['backend_framewoark']}' but you are using '{nn.implemented_backend}'"
        )

        nn._input_mean = np.array(config["norm_config"]["_input_mean"])
        nn._input_std = np.array(config["norm_config"]["_input_std"])
        nn._input_standardized = np.array(config["norm_config"]["_input_standardized"])
        if "_input_prescale" in config["norm_config"]:
            nn._input_prescale = np.array(config["norm_config"]["_input_prescale"])
        else:
            nn._input_prescale = np.ones_like(nn._input_mean)
        nn._load_internal(folder_name, device=device)

        if checkpoint_epoch != -1:
            base = folder_name.rsplit("/checkpoints/epoch_", 1)[0]
            nn.root_folder = os.path.abspath(base)

        nn.mixed_precision = mixed_precision
        return nn


    def perceive(
            self,
            inputs: Union[List[float], NDArray],
            inputs_token: List[Union[int, str]],
            batch_size: int = 1024,
            return_attention_scores: bool = False,
            inference_mode: bool = True,
        ) -> Optional[NDArray]:
            """
            Prepara la memoria de percepción del modelo.
            """
            self._perception_check(mode=2)
            self._built_only()
            inputs = np.atleast_2d(inputs)
            inputs_token = np.atleast_2d(inputs_token)
            inputs_token = self.tokenize(inputs_token, data_length=len(inputs))

            # ← freeze stats at inference
            inputs, inputs_token, _ = self.standardize(inputs, inputs_token, inputs_error=None, update=False)
            inputs, inputs_token = self.padding(inputs, inputs_token)
            self._last_padding_mask = (inputs_token == 0)

            self._perception_memory, attention_scores = self._perceive_internal(
                inputs, inputs_token, batch_size=batch_size,
                return_attention_scores=return_attention_scores, inference_mode=inference_mode
            )
            if return_attention_scores:
                return attention_scores

    def request(self, request_tokens, batch_size=64, return_attention_scores=False):
        if isinstance(request_tokens, (list, np.ndarray)):
            request_tokens = np.array(request_tokens)
        if request_tokens.ndim == 1:
            request_tokens = np.atleast_2d(request_tokens)
        if request_tokens.dtype.type == np.str_:
            request_tokens = self.tokenize(request_tokens)

        pred, pred_err, main_class_preds, morph_preds, classification_probs, attention_scores = self._request_internal(
            request_tokens,
            batch_size=batch_size,
            return_attention_scores=return_attention_scores,
        )

        inv = self.inverse_standardize(pred, request_tokens, pred_err)
        if len(inv) == 2:
            pred, pred_err = inv
        else:
            pred = inv[0]; pred_err = None

        output = {}
        try:
            main_class_token = self.tokenize(['SPECTYPE'])[0][0]
        except Exception:
            main_class_token = self.tokenize(['class_label'])[0][0]
        morph_token = self.tokenize(['MORPHTYPE'])[0][0]

        is_regression = ~((request_tokens[0] == main_class_token) | (request_tokens[0] == morph_token))
        if is_regression.any():
            output['regression_preds'] = pred[:, is_regression]
            output['regression_preds_err'] = pred_err[:, is_regression] if pred_err is not None else None

        is_main_class = (request_tokens[0] == main_class_token)
        if is_main_class.any():
            output['main_class_preds'] = main_class_preds[:, is_main_class]
            output['main_class_probs'] = classification_probs[:, is_main_class, :]

        is_morph = (request_tokens[0] == morph_token)
        if is_morph.any():
            output['morphology_preds'] = morph_preds[:, is_morph]

        if return_attention_scores:
            output['attention_scores'] = attention_scores
        return output

    # ────────────────────── extracción de embeddings ──────────────────────
    def get_encoder_representation(self, inputs, input_tokens, batch_size=1024):
        self._perception_check(mode=2)
        self._built_only()
        inputs = np.atleast_2d(inputs)
        input_tokens = np.atleast_2d(input_tokens)
        input_tokens = self.tokenize(input_tokens, data_length=len(inputs))

        # ← freeze stats at inference
        inputs, input_tokens, _ = self.standardize(inputs, input_tokens, inputs_error=None, update=False)
        inputs, input_tokens = self.padding(inputs, input_tokens)
        padding_mask = (input_tokens == 0)

        self.torch_model.eval()
        with torch.inference_mode():
            input_tokens_t = torch.as_tensor(input_tokens, device=self.device, dtype=torch.int32)
            input_embedded = self.embedding_layer(
                input_tokens_t,
                torch.atleast_3d(torch.as_tensor(inputs, device=self.device, dtype=self.dtype)),
            )
            data_length = len(inputs)
            num_batch = data_length // batch_size
            num_batch_remainder = data_length % batch_size
            if num_batch == 0:
                perception = self.torch_encoder(input_embedded, mask=torch.as_tensor(padding_mask, device=self.device))
            else:
                perception = []
                for i in range(num_batch):
                    perception.append(
                        self.torch_encoder(
                            input_embedded[i * batch_size: (i + 1) * batch_size],
                            mask=torch.as_tensor(padding_mask[i * batch_size: (i + 1) * batch_size], device=self.device)
                        )
                    )
                if num_batch_remainder > 0:
                    perception.append(
                        self.torch_encoder(
                            input_embedded[num_batch * batch_size:],
                            mask=torch.as_tensor(padding_mask[num_batch * batch_size:], device=self.device)
                        )
                    )
                perception = torch.cat(perception, dim=0)
        return perception.cpu().numpy()