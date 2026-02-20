# nn_utils.py
import os

import torch
import numpy as np
from typing import List

class TrainingGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        batch_size: int,
        data: dict,
        data_probabilities: np.ndarray,
        outputs_padding: int = 0,
        possible_output_tokens: List[int] = None,
        input_length: int = None,
        shuffle: bool = True,
        aggregate_nans: bool = True,
        factory_kwargs: dict = {"device": "cpu", "dtype": torch.float32},
        class_label_token: int = None,
        morph_label_token: int = None,
        token_weights: torch.Tensor = None,
        n_tokens_predict: int = 1, 
    ):

        self.class_label_token = class_label_token
        self.morph_label_token = morph_label_token
        self.n_tokens_predict = n_tokens_predict 

        target_device = factory_kwargs.get("device", "cpu")
        self.compute_device = target_device if isinstance(target_device, torch.device) else torch.device(target_device)
        self.dtype = factory_kwargs.get("dtype", torch.float32)

        self.class_labels = data.get("class_labels", None)
        if self.class_labels is not None:
            self.class_labels = torch.as_tensor(np.array(self.class_labels), dtype=torch.long) # CPU

        self.morph_labels = data.get("morph_labels", None)
        if self.morph_labels is not None:
            self.morph_labels = torch.as_tensor(np.array(self.morph_labels), dtype=torch.long) # CPU

        input_np = data["input"]
        if input_np.ndim == 2: input_np = np.expand_dims(input_np, axis=-1)
        self.input = torch.as_tensor(input_np, dtype=self.dtype) # CPU
        
        self.input_idx = torch.as_tensor(data["input_token"], dtype=torch.long) # CPU
        
        output_np = data["output"]
        if output_np.ndim == 2: output_np = np.expand_dims(output_np, axis=-1)
        self.output = torch.as_tensor(output_np, dtype=self.dtype) # CPU

        output_err_np = data["output_err"]
        if output_err_np.ndim == 2: output_err_np = np.expand_dims(output_err_np, axis=-1)
        self.output_err = torch.as_tensor(output_err_np, dtype=self.dtype) # CPU

        self.outputs_padding = outputs_padding
        self.data_length = len(self.input)
        self.data_width = self.input.shape[1]
        self.shuffle = shuffle 
        self.aggregate_nans = aggregate_nans

        if isinstance(data_probabilities, torch.Tensor):
            self.data_probabilities = data_probabilities.cpu().float().flatten()
        else:
            self.data_probabilities = torch.as_tensor(data_probabilities, dtype=torch.float).flatten()

        self.possible_output_tokens_tensor = torch.tensor(
            np.atleast_1d(possible_output_tokens).astype(int), device=self.compute_device, dtype=torch.long
        )


        present_cpu = torch.zeros((self.data_length, self.possible_output_tokens_tensor.size(0)), dtype=torch.bool)
        possible_tokens_cpu = self.possible_output_tokens_tensor.cpu()
        
        for j, t in enumerate(possible_tokens_cpu):
            present_cpu[:, j] = torch.any(self.input_idx == t, dim=1)

        prob_matrix = present_cpu.float()

        if token_weights is not None:

            if not isinstance(token_weights, torch.Tensor):
                token_weights = torch.tensor(token_weights, dtype=torch.float32)
            
            current_weights = token_weights[possible_tokens_cpu].float()
            
            prob_matrix = prob_matrix * current_weights.unsqueeze(0)
        

        row_sums = prob_matrix.sum(axis=1, keepdim=True)
        empty = (row_sums.squeeze() == 0)
        if torch.any(empty):
            prob_matrix[empty, :] = 1.0
            row_sums = prob_matrix.sum(axis=1, keepdim=True)
        
        self.output_prob_matrix = prob_matrix / row_sums.clamp_min(1e-6)

        if aggregate_nans:
            input_idx_np = data["input_token"]
            partialsort_idx_np = np.argsort(input_idx_np == 0, axis=1, kind="mergesort")
            partialsort_idx = torch.as_tensor(partialsort_idx_np, dtype=torch.long)
            
            self.input = torch.gather(self.input, 1, partialsort_idx.unsqueeze(-1).expand_as(self.input))
            self.input_idx = torch.gather(self.input_idx, 1, partialsort_idx)
            
            self.first_n_shuffle = self.data_width - np.sum(input_idx_np == 0, axis=1)
        else:
            self.first_n_shuffle = np.full(self.data_length, self.data_width, dtype=int)
        
        self.first_n_shuffle_tensor = torch.as_tensor(self.first_n_shuffle, dtype=torch.long).unsqueeze(1) # CPU

        self.batch_size = batch_size
        self.steps_per_epoch = max(1, self.data_length // self.batch_size)
        
        self.input_length = input_length
        if self.input_length is None:
            self.input_length = data["input"].shape[1]

    def on_epoch_end(self):

        pass

    def __iter__(self):

        num_samples_needed = self.steps_per_epoch * self.batch_size
        replacement_needed = num_samples_needed > self.data_length

        full_idxs = torch.multinomial(
            self.data_probabilities,
            num_samples_needed,
            replacement=replacement_needed
        )
        
        for i in range(self.steps_per_epoch):
            batch_idxs = full_idxs[i * self.batch_size:(i + 1) * self.batch_size]
            
            yield self._process_batch(batch_idxs)

    def __len__(self):
        return self.steps_per_epoch

    def _process_batch(self, batch_idxs):


        b_input = self.input[batch_idxs].to(self.compute_device, non_blocking=True)
        b_input_idx = self.input_idx[batch_idxs].to(self.compute_device, non_blocking=True)
        b_first_n = self.first_n_shuffle_tensor[batch_idxs].to(self.compute_device, non_blocking=True)
        
        B = len(batch_idxs)
        S = self.data_width

        if self.shuffle:
            rand_vals = torch.rand(B, S, device=self.compute_device)
            seq_range = torch.arange(S, device=self.compute_device).unsqueeze(0)
            
            shuffle_mask = seq_range < b_first_n
            rand_vals.masked_fill_(~shuffle_mask, torch.inf)
            
            sort_idx = rand_vals.argsort(dim=1, stable=True)
            
            b_input = torch.gather(b_input, 1, sort_idx.unsqueeze(-1).expand_as(b_input))
            b_input_idx = torch.gather(b_input_idx, 1, sort_idx)

        b_input = b_input[:, : self.input_length]
        b_input_idx = b_input_idx[:, : self.input_length]
        S_new = self.input_length

        if self.outputs_padding != 0:
            padding_length = torch.randint(0, self.outputs_padding + 1, (B, 1), device=self.compute_device)
            pad_mask = torch.arange(S_new, device=self.compute_device).unsqueeze(0) >= (S_new - padding_length)
            
            b_input.masked_fill_(pad_mask.unsqueeze(-1).expand_as(b_input), 0.0)
            b_input_idx.masked_fill_(pad_mask, 0)


        b_prob_matrix = self.output_prob_matrix[batch_idxs].to(self.compute_device, non_blocking=True)
        
        output_idx_indices = torch.multinomial(b_prob_matrix, self.n_tokens_predict) # [B, N]
        b_output_token_idx = self.possible_output_tokens_tensor[output_idx_indices] # [B, N]

        mask_target = (b_input_idx.unsqueeze(-1) == b_output_token_idx.unsqueeze(1)).any(dim=-1) # [B, S]
        
        b_input.masked_fill_(mask_target.unsqueeze(-1).expand_as(b_input), 0.0)
        b_input_idx.masked_fill_(mask_target, 0)

        non_zero_mask = b_input_idx != 0
        sorted_indices = non_zero_mask.int().argsort(dim=1, descending=True, stable=True)
        
        b_input = torch.gather(b_input, 1, sorted_indices.unsqueeze(-1).expand_as(b_input))
        b_input_idx = torch.gather(b_input_idx, 1, sorted_indices)

        b_full_output = self.output[batch_idxs].to(self.compute_device, non_blocking=True)
        b_full_output_err = self.output_err[batch_idxs].to(self.compute_device, non_blocking=True)

        # Gather
        N_tokens = self.n_tokens_predict
        gather_idx = (b_output_token_idx - 1).long().unsqueeze(-1) # [B, N, 1]
        gather_idx = gather_idx.expand(B, N_tokens, b_full_output.shape[2])
        
        b_target_val = torch.gather(b_full_output, 1, gather_idx) # [B, N, 1]
        b_target_val = b_target_val.squeeze(-1) # [B, N] (si la última dim era 1)
        if b_target_val.ndim == 3 and b_target_val.shape[-1] == 1: b_target_val = b_target_val.squeeze(-1) # Extra check

        gather_err_idx = (b_output_token_idx - 1).long().unsqueeze(-1)
        gather_err_idx = gather_err_idx.expand(B, N_tokens, b_full_output_err.shape[2])
        b_target_err = torch.gather(b_full_output_err, 1, gather_err_idx) # [B, N, 1]
        b_target_err = b_target_err.squeeze(-1) # [B, N]

        is_class_pred = (b_output_token_idx == self.class_label_token) # [B, N]
        is_morph_pred = (b_output_token_idx == self.morph_label_token) # [B, N]

        b_class_labels = None
        if self.class_labels is not None:
            b_class_labels = self.class_labels[batch_idxs].to(self.compute_device, non_blocking=True)
            if torch.any(is_class_pred):
                cls_float = b_class_labels.float().unsqueeze(1) # [B, 1]
                b_target_val = torch.where(is_class_pred, cls_float, b_target_val) # [B, N]
                b_target_err = torch.where(is_class_pred, 0.0, b_target_err)

        b_morph_labels = None
        if self.morph_labels is not None:
            b_morph_labels = self.morph_labels[batch_idxs].to(self.compute_device, non_blocking=True)
            if torch.any(is_morph_pred):
                morph_float = b_morph_labels.float().unsqueeze(1) # [B, 1]
                b_target_val = torch.where(is_morph_pred, morph_float, b_target_val)
                b_target_err = torch.where(is_morph_pred, 0.0, b_target_err)
        
        prob_weights = self.data_probabilities[batch_idxs].to(self.compute_device, non_blocking=True)

        return (
            b_input,
            b_input_idx,
            b_output_token_idx, # [B, N]
            b_target_val,       # [B, N]
            b_target_err,       # [B, N]
            b_class_labels,
            b_morph_labels,
            prob_weights
        )
