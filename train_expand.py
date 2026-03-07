import os
import glob
from pathlib import Path
import torch
import json
import numpy as np
import pandas as pd
from typing import Optional
import sys


# ───────────────────────────────────────────────────────────────────
# Path configuration
# ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import OJALA

# ───────────────────────────────────────────────────────────────────
# 0. Global torch settings
# ───────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 64
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
memory_format = torch.channels_last

# ───────────────────────────────────────────────────────────────────
# 1. SETTINGS 


EPOCH_TO_LOAD = 2
# ===========================================



DATA_DIR = REPO_ROOT / "data"
MOCKS_DIR = DATA_DIR / "J-PAS_mock_data"
CATALOGUES_DIR = DATA_DIR / "catalogues"
PRETRAINED_MODEL_FOLDER = REPO_ROOT / "model_OJALA"
NEW_MODEL_FOLDER = REPO_ROOT / "model_OJALA_new"

folder_S = MOCKS_DIR 
U_h5_path = CATALOGUES_DIR / "JPAS_EDR_photometry.h5"

if not folder_S.is_dir():
    raise FileNotFoundError(f"Mock directory not found: {folder_S}")

if not U_h5_path.is_file():
    raise FileNotFoundError(f"Unsupervised HDF5 file not found: {U_h5_path}")


all_pseudobatch_files = sorted(str(p) for p in folder_S.glob("*.h5"))

num_pseudo_batches = "ALL"
selected_files = all_pseudobatch_files if num_pseudo_batches == "ALL" else all_pseudobatch_files[:num_pseudo_batches]

if len(all_pseudobatch_files) == 0:
    raise FileNotFoundError(f"No .h5 pseudobatch files found in: {folder_S}")


if not PRETRAINED_MODEL_FOLDER.is_dir():
    raise FileNotFoundError(f"Pretrained model directory not found: {PRETRAINED_MODEL_FOLDER}")


if len(all_pseudobatch_files) == 0:
    raise FileNotFoundError(f"No .h5 pseudobatch files found in: {folder_S}")




ckpt_folder_path = PRETRAINED_MODEL_FOLDER / "checkpoints" / f"epoch_{EPOCH_TO_LOAD}"
config_path_epoch = ckpt_folder_path / "config.json"
weights_path_epoch = ckpt_folder_path / "weights.pt"
print(f"\n📂 Archivos seleccionados:\n   Config: {config_path_epoch}\n   Weights: {weights_path_epoch}")

if not config_path_epoch.is_file():
    raise FileNotFoundError(f"Config file not found for epoch {EPOCH_TO_LOAD}: {config_path_epoch}")
if not weights_path_epoch.is_file():
    raise FileNotFoundError(f"Weights file not found for epoch {EPOCH_TO_LOAD}: {weights_path_epoch}")

with open(config_path_epoch, "r") as f:
    old_config = json.load(f)

tokens_names_OLD = old_config["tokenizer_config"]["vocabs"]
tokens_ids_OLD   = old_config["tokenizer_config"]["vocab_tokens"]

FilterJPAS_REF = [
    'J0378','J0390','J0400','J0410','J0420','J0430','J0440','J0450', 'J0460','J0470',
    'J0480','J0490','J0500','J0510','J0520','J0530','J0540','J0550','J0560','J0570',
    'J0580','J0590','J0600','J0610','J0620','J0630','J0640','J0650','J0660','J0670',
    'J0680','J0690','J0700','J0710','J0720','J0730','J0740','J0750','J0760','J0770',
    'J0780','J0790','J0800','J0810','J0820','J0830','J0840','J0850','J0860','J0870',
    'J0880','J0890','J0900','J0910'
]
Base_Obs_REF = FilterJPAS_REF + ['MAG_i','MAG_G','MAG_R','MAG_Z', 'MORPHTYPE','MAG_W1', 'MAG_W2']

Target_Vars_REF = (['LOGM'] + ['LOGG', 'TEFF', 'ALPHAFE', 'FEH'] + 
                   ['OIII_5007_EW','NII_6584_EW','HBETA_4861_EW','HALPHA_6562_EW'] + 
                   ['HBETA_CONT', 'HALPHA_CONT'] + ['Z_GAL','Z_QSO'] + ['SPECTYPE'] + ['LOGSFR'])

# 3. EDIT HERE


VARS_TO_REMOVE = ["HBETA_CONT"]
NEW_OBS_TO_ADD     = ["W3"]
NEW_TARGETS_TO_ADD = ["OII_3727_CONT"]

# ───────────────────────────────────────────────────────────────────
# 4. CONSTRUCCIÓN DEL NUEVO VOCABULARIO
# ───────────────────────────────────────────────────────────────────
def filter_list(source_list, remove_list):
    return [t for t in source_list if t not in remove_list]

tokens_names_NEW_base = [t for t in tokens_names_OLD if t not in VARS_TO_REMOVE]
tokens_names_NEW = tokens_names_NEW_base + NEW_OBS_TO_ADD + NEW_TARGETS_TO_ADD

Base_Obs_NEW_pool    = filter_list(Base_Obs_REF, VARS_TO_REMOVE) + NEW_OBS_TO_ADD
Target_Vars_NEW_pool = filter_list(Target_Vars_REF, VARS_TO_REMOVE) + NEW_TARGETS_TO_ADD

observations = [t for t in tokens_names_NEW if t in Base_Obs_NEW_pool]
target_var   = [t for t in tokens_names_NEW if t in Target_Vars_NEW_pool]

da_tokens    = observations 
context_length = len(tokens_names_NEW) - 1

print("\n" + "="*60)
print(f"📊 PRE-RUN DIAGNOSTICS:")
print(f"   - Vocabulario OLD (Config): {len(tokens_names_OLD)} tokens")
print(f"   - Vocabulario NEW (Objetivo): {len(tokens_names_NEW)} tokens")
print(f"   - Observations (Ordenados): {len(observations)}")
print(f"   - Targets (Ordenados): {len(target_var)}")
print("="*60 + "\n")

# ───────────────────────────────────────────────────────────────────
# 2. Helpers Data Loading
# ───────────────────────────────────────────────────────────────────
def _stack_columns(df: pd.DataFrame, names: list):
    N = len(df)
    vals, errs = [], []
    for n in names:
        vals.append(df[n].to_numpy() if n in df.columns else np.full(N, np.nan, dtype=float))
        en = f"{n}_ERR"
        errs.append(df[en].to_numpy() if en in df.columns else np.full(N, np.nan, dtype=float))
    return np.column_stack(vals), np.column_stack(errs)

def load_pseudobatch_S(fname, obs_names):
    with pd.HDFStore(fname, 'r') as store:
        df = store['data']
    training_labels, training_labels_err = _stack_columns(df, obs_names)
    class_label  = df['SPECTYPE'].to_numpy() if 'SPECTYPE' in df.columns else None
    morph_labels = df['MORPHTYPE'].to_numpy() if 'MORPHTYPE' in df.columns else None
    probs        = df['prob2'].to_numpy() if 'prob2' in df.columns else np.ones(len(df), dtype=float)
    return {
        "training_labels": training_labels, "training_labels_err": training_labels_err,
        "class_label": class_label, "morph_labels": morph_labels,
        "obs_names": np.atleast_2d(obs_names), "inputs_probabilities": probs
    }

def pseudo_batch_loader_S(file_list, obs_names):
    for fname in file_list:
        yield load_pseudobatch_S(fname, obs_names)

# ───────────────────────────────────────────────────────────────────
# 3. Model Build & Weight Surgery 
# ───────────────────────────────────────────────────────────────────
num_classes = 3
num_morph_classes = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mixed_precision = True

print(f"🚀 Creando modelo NUEVO en: {NEW_MODEL_FOLDER}")
nn_model = OJALA(
    vocabs=tokens_names_NEW,  
    context_length=context_length,
    embedding_dim=128,
    num_classes=num_classes, num_morph_classes=num_morph_classes, embedding_activation="gelu",
    encoder_head_num=16, encoder_dense_num=1024, encoder_dropout_rate=0.1, encoder_activation="gelu",
    decoder_head_num=16, decoder_dense_num=2048, decoder_dropout_rate=0.1, decoder_activation="gelu",
    device=device,
    mixed_precision=(device.type != "cpu" and mixed_precision),
    folder=str(NEW_MODEL_FOLDER)
)

print(f"🔄 Loading raw checkpoint weights...")
try:
    checkpoint = torch.load(weights_path_epoch, map_location=device, weights_only=False)
except TypeError:
    checkpoint = torch.load(weights_path_epoch, map_location=device)

raw_state_dict = checkpoint['model_state_dict']

def find_best_matching_tensor(target_name, raw_dict):
    """
    Busca en el raw_dict cualquier clave que coincida en nombre base.
    Devuelve la que tenga mayor dimensión 0 para embeddings.
    Maneja tensores escalares (0-d) sin crashear.
    """
    candidates = []
    for k, v in raw_dict.items():
        clean_k = k.replace("_orig_mod.", "")
        if clean_k == target_name:
            candidates.append((k, v))
            
    if not candidates:
        return None, None

    candidates.sort(
        key=lambda x: x[1].shape[0] if len(x[1].shape) > 0 else 0, 
        reverse=True
    )
    
    best_key, best_tensor = candidates[0]
    return best_key, best_tensor

# ===================================================

new_state_dict = nn_model.torch_model.state_dict()
final_state_dict = {}

vocab_map_OLD = dict(zip(tokens_names_OLD, tokens_ids_OLD))
vocab_map_NEW = {name: i + 1 for i, name in enumerate(tokens_names_NEW)} 

copied_tokens = 0
reinit_tokens = 0

print("\n🔧 Starting weight surgery:")

for key, new_param in new_state_dict.items():
    
    best_key_found, old_param = find_best_matching_tensor(key, raw_state_dict)
    
    if old_param is None:
        final_state_dict[key] = new_param
        continue
    is_main_embedding = "embedding_layer" in key and (key.endswith("embeddings") or key.endswith("bias"))
    is_meta_mlp = "meta_mlp" in key
    
    if is_main_embedding and not is_meta_mlp:
        transferred_param = new_param.clone()
        real_size = old_param.shape[0]
        
        print(f"   ► Capa '{key}': Cirugía (Origen: '{best_key_found}', Shape: {old_param.shape})")

        for token_name, new_idx in vocab_map_NEW.items():
            if token_name in vocab_map_OLD:
                old_idx = vocab_map_OLD[token_name]
                
                if old_idx < real_size:
                    transferred_param[new_idx] = old_param[old_idx]
                    copied_tokens += 1
                else:

                    if "embeddings" in key: 
                        print(f"     ⚠️ Token '{token_name}' (ID {old_idx}) > Matriz ({real_size}). Se reinicia.")
                    reinit_tokens += 1
            else:
                pass 
        
        if real_size > 0: 
            transferred_param[0] = old_param[0] # Padding

        final_state_dict[key] = transferred_param

    elif "token_metadata" in key: 
        continue
    elif old_param.shape == new_param.shape:
        final_state_dict[key] = old_param
    else:
        print(f"   ⚠️ Salto capa '{key}' (Shape mismatch: New {new_param.shape} vs Old {old_param.shape})")

nn_model.torch_model.load_state_dict(final_state_dict, strict=False)
print(f"\n✅ Transfer completed.")
print(f"   - Copied token assignments: {copied_tokens}")
print(f"   - Reinitialized token assignments: {reinit_tokens}")
print(f"   (Ready for training).\n")




print("\n" + "█"*80)
print("📋 SUMMARY")
print("█"*80)

print(f"\n1️⃣  OLD TOKENS   [Total: {len(tokens_names_OLD)}]:")
print(tokens_names_OLD)

print(f"\n2️⃣  NEW TOKENS  [Total: {len(tokens_names_NEW)}]:")
print(tokens_names_NEW)

print(f"\n3️⃣  OBSERVATIONS (Inputs for Encoder) [Total: {len(observations)}]:")
print(observations)

print(f"\n4️⃣  TARGETS (Outputs for Decoder) [Total: {len(target_var)}]:")
print(target_var)

print(f"\n5️⃣  DA TOKENS (Domain adaptation variables) [Total: {len(da_tokens)}]:")
print(da_tokens)

print(f"\n📏 CONTEXT LENGTH: {context_length}")
print("█"*80 + "\n")

# ───────────────────────────────────────────────────────────────────
# 4. Training
# ───────────────────────────────────────────────────────────────────
LEARNING_RATE_CONTINUE = 5e-5 
DA_LR_SCALE = 0.3

nn_model.optimizer = torch.optim.AdamW(
    [
        {"params": nn_model.torch_model.parameters(), "lr": LEARNING_RATE_CONTINUE},
        {"params": nn_model.domain_disc.parameters(), "lr": LEARNING_RATE_CONTINUE * DA_LR_SCALE},
        {"params": nn_model.reweighter.parameters(),  "lr": LEARNING_RATE_CONTINUE * DA_LR_SCALE},
    ],
    fused=(device.type=="cuda")
)
scheduler_factory = lambda opt: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000, T_mult=1, eta_min=1e-10)

nn_model.torch_model = nn_model.torch_model.to(device, memory_format=memory_format)
nn_model.domain_disc = nn_model.domain_disc.to(device)
nn_model.reweighter  = nn_model.reweighter.to(device)
nn_model.torch_model = torch.compile(nn_model.torch_model, mode="reduce-overhead")

pseudo_loader_callable_S = lambda: pseudo_batch_loader_S(selected_files, tokens_names_NEW)

print("▶️  Starting training...")


nn_model.fit(
    pseudo_batch_loader_S=pseudo_loader_callable_S,
    U_h5_path=str(U_h5_path),
    jpas_filter_names=FilterJPAS_REF, 
    batch_size_S=10000,
    batch_size_U=2000,
    epochs=2000,
    z_loss_mode="delta",
    z_var_reg=0.0,
    lambda_dom=0.0,
    lambda_mmd=0.2,
    entropy_coef=0.1,
    lr_scheduler_factory=scheduler_factory,
    checkpoint_every_n_epochs=1,
    terminate_on_nan=True,
    da_token_names=da_tokens,          
    apply_context_masking=True,
    observations_names=observations,   
    target_names=target_var,           
    keep_frac_min=0.0,
    keep_frac_max=1.0,
    keep_min=5,
)