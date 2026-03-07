import os
import glob
from pathlib import Path
import torch
import json
import numpy as np
import pandas as pd
from typing import Optional, List
import sys


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
# Path configuration
# ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
MOCKS_DIR = DATA_DIR / "mocks"
CATALOGUES_DIR = DATA_DIR / "catalogues"
MODEL_DIR = REPO_ROOT / "model_OJALA"


from src.model import OJALA

# ───────────────────────────────────────────────────────────────────
# 1. FOLDER CONFIGURATION
# ───────────────────────────────────────────────────────────────────
# THE MODEL FROM WHICH I WANT TO RESUME TRAINING
model_folder = MODEL_DIR
if not model_folder.is_dir():
    raise FileNotFoundError(f"Model directory not found: {model_folder}")

# ───────────────────────────────────────────────────────────────────
# 1. Settings for S + U
# ───────────────────────────────────────────────────────────────────
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

def _latest_checkpoint(folder: Path) -> Optional[int]:
    ckpt_root = folder / "checkpoints"
    if not ckpt_root.is_dir():
        return None
    epochs = [int(d.name.split("_")[1]) for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
    return max(epochs) if epochs else None

resume_epoch = _latest_checkpoint(model_folder)

#Reference list, order does not matter
REF_OBSERVATIONS = [
    'J0378','J0390','J0400','J0410','J0420','J0430','J0440','J0450','J0460','J0470',
    'J0480','J0490','J0500','J0510','J0520','J0530','J0540','J0550','J0560','J0570',
    'J0580','J0590','J0600','J0610','J0620','J0630','J0640','J0650','J0660','J0670',
    'J0680','J0690','J0700','J0710','J0720','J0730','J0740','J0750','J0760','J0770',
    'J0780','J0790','J0800','J0810','J0820','J0830','J0840','J0850','J0860','J0870',
    'J0880','J0890','J0900','J0910',
    'MAG_i','MAG_G','MAG_R','MAG_Z', 'MORPHTYPE',
    'MAG_W1', 'MAG_W2'
]

REF_TARGETS = [
    'LOGM', 'LOGG', 'TEFF', 'ALPHAFE', 'FEH',
   'OIII_5007_EW','NII_6584_EW','HBETA_4861_EW','HALPHA_6562_EW',
    'HBETA_CONT', 'HALPHA_CONT', 'Z_GAL','Z_QSO', 'SPECTYPE',
     'LOGSFR'
]

if resume_epoch is not None:
    print(f"📄 Checkpoint detectado (Epoch {resume_epoch}). Leyendo configuración...")
    config_path = model_folder / "checkpoints" / f"epoch_{resume_epoch}" / "config.json"    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tokens_names = config["tokenizer_config"]["vocabs"]
    print(tokens_names)
    print(f"✅ Vocabulario cargado del disco ({len(tokens_names)} tokens). Orden preservado.")


    observations = [t for t in tokens_names if t in REF_OBSERVATIONS]
    target_var   = [t for t in tokens_names if t in REF_TARGETS]
    
    # J-PAS filters for DA
    FilterJPAS = ['J0378','J0390','J0400','J0410','J0420','J0430','J0440','J0450', 'J0460','J0470',
    'J0480','J0490','J0500','J0510','J0520','J0530','J0540','J0550','J0560','J0570',
    'J0580','J0590','J0600','J0610','J0620','J0630','J0640','J0650','J0660','J0670',
    'J0680','J0690','J0700','J0710','J0720','J0730','J0740','J0750','J0760','J0770',
    'J0780','J0790','J0800','J0810','J0820','J0830','J0840','J0850','J0860','J0870',
    'J0880','J0890','J0900','J0910']

    
else:
    raise FileNotFoundError(
        f"❌ No checkpoint was found in {model_folder}. "
        "This script is only intended to resume existing trainings."
    )
da_tokens = observations 
context_length = len(tokens_names) - 1

print(f"📊 Resumen de Configuración Detectada:")
print(f"   - Context Length: {context_length}")
print(f"   - Inputs (Observations): {len(observations)}")
print(f"   - Targets: {len(target_var)}")

# ───────────────────────────────────────────────────────────────────
# 3. Helpers Data Loading
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
# 4. Resume Model
# ───────────────────────────────────────────────────────────────────
num_classes = 3
num_morph_classes = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mixed_precision = True

# LR Settings (Standard Resume)
LR_RESUME = 1e-5
DA_LR_SCALE = 0.3

def _scheduler_factory(opt):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=1e-9)

print(f"🔄  Cargando modelo completo desde epoch {resume_epoch} …")


nn_model = OJALA.load(
    folder_name       = str(model_folder),
    checkpoint_epoch  = resume_epoch,
    mixed_precision   = mixed_precision,
    device            = device,
)
nn_model.root_folder = str(model_folder.resolve())
# Move to device
nn_model.torch_model = nn_model.torch_model.to(device, memory_format=memory_format)
nn_model.domain_disc = nn_model.domain_disc.to(device)
nn_model.reweighter  = nn_model.reweighter.to(device)

# Force LR update (por si acaso el guardado tenía un LR muy bajo del final del ciclo)
for g in nn_model.optimizer.param_groups:
    g["lr"] = LR_RESUME

print(f"Total Trainable Parameters (M): {nn_model.get_parameters_sum() / 1e6:.2f}")

# Compile
nn_model.torch_model = torch.compile(nn_model.torch_model, mode="reduce-overhead")



# ───────────────────────────────────────────────────────────────────
# 5. INVERSE PROBABILITY TOKEN WEIGHTING (MONTE CARLO SIMULATION)
# ───────────────────────────────────────────────────────────────────
MAX_TOKEN_WEIGHT = 1000.0
CALIB_BATCH_SIZE = 10000 
N_ITER_CALIB = 10        



def calculate_token_weights(file_list, vocab_names, nn_model):
    # --- CAMBIO AQUÍ: Seleccionamos hasta 2 archivos ---
    files_to_use = file_list[:2]
    print(f"\n⚖️  Calculando pesos inversos mediante Simulación Monte Carlo...")
    print(f"   (Archivos usados: {len(files_to_use)} | Iteraciones: {N_ITER_CALIB} | BatchSize: {CALIB_BATCH_SIZE})")
    
    vals_list = []
    probs_list = []
    
    for fname in files_to_use:
        print(f"   -> Cargando: {os.path.basename(fname)}")
        batch_data = load_pseudobatch_S(fname, vocab_names)
        vals_list.append(batch_data["training_labels"])
        probs_list.append(batch_data["inputs_probabilities"])
    
    vals = np.vstack(vals_list)
    probs = np.concatenate(probs_list)
    
    if probs.sum() == 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / probs.sum()


    token_ids = nn_model.tokenize([vocab_names])[0].flatten().astype(int)
    

    valid_counts_accumulator = np.zeros(len(vocab_names), dtype=float)
    total_samples_seen = 0
    
    for _ in range(N_ITER_CALIB):

        indices = np.random.choice(len(vals), size=CALIB_BATCH_SIZE, p=probs, replace=True)
        
        sampled_vals = vals[indices]
        

        batch_valid_counts = np.isfinite(sampled_vals).sum(axis=0)
        
        valid_counts_accumulator += batch_valid_counts
        total_samples_seen += CALIB_BATCH_SIZE
        

    effective_freqs = valid_counts_accumulator / total_samples_seen
    
    max_id = int(max(nn_model.vocab_tokens))
    weight_vector = np.ones(max_id + 1, dtype=np.float32)
    
    report_data = []
    
    for i, col_name in enumerate(vocab_names):
        tid = token_ids[i]
        if tid == 0: continue 
            
        freq = effective_freqs[i]
        
        if freq > 0:
            w = 1.0 / freq
        else:
            w = 0.0 
            
        w_capped = min(w, MAX_TOKEN_WEIGHT)
        
        weight_vector[tid] = w_capped
        
        report_data.append({
            "Token": col_name,
            "ID": tid,
            "Freq": freq,
            "Weight": w_capped
        })
        
    df_rep = pd.DataFrame(report_data)
    df_rep = df_rep.sort_values(by="Weight", ascending=False)
    
    pd.set_option('display.max_rows', None)
    
    print("-" * 65)
    print(f"{'TOKEN':<20} | {'ID':<5} | {'FREQ (Eff)':<10} | {'WEIGHT':<8}")
    print("-" * 65)
    
    for _, r in df_rep.iterrows(): 
        print(f"{r['Token']:<20} | {int(r['ID']):<5} | {r['Freq']:.5f}    | {r['Weight']:.2f}")
        
    print("-" * 65)
    
    return torch.tensor(weight_vector, dtype=torch.float32)

token_balancing_weights = calculate_token_weights(selected_files, tokens_names, nn_model)
# ───────────────────────────────────────────────────────────────────
# 6. Continue Training
# ───────────────────────────────────────────────────────────────────
pseudo_loader_callable_S = lambda: pseudo_batch_loader_S(selected_files, tokens_names)


print("▶️  Resuming training...")

nn_model.fit(
    pseudo_batch_loader_S     = pseudo_loader_callable_S,
    U_h5_path                 = str(U_h5_path),
    jpas_filter_names         = FilterJPAS,
    batch_size_S              = CALIB_BATCH_SIZE,
    batch_size_U              = 2000,
    epochs                    = 2000, 
    z_loss_mode               = "delta",
    z_var_reg                 = 0.0,
    lambda_dom                = 0.0,
    lambda_mmd                = 0.2,
    entropy_coef              = 0.1,
    lr_scheduler_factory      = _scheduler_factory,
    checkpoint_every_n_epochs = 1,
    terminate_on_nan          = True,
    da_token_names            = da_tokens,
    apply_context_masking     = True,
    observations_names        = observations,
    target_names              = target_var,
    keep_frac_min             = 0.,
    keep_frac_max             = 1.0,
    keep_min                  = 5,
    ess_target_frac           = 0.7,
    token_weights             = token_balancing_weights,
    n_tokens_predict          = 5,
)
