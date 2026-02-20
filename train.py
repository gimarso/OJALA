import os
import glob
import torch
import json
import numpy as np
import pandas as pd
from typing import Optional, List

# ───────────────────────────────────────────────────────────────────
# 0. Global torch settings
# ───────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 64
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
memory_format = torch.channels_last

from src.model import OJALA

# ───────────────────────────────────────────────────────────────────
# 1. Settings for S + U
# ───────────────────────────────────────────────────────────────────
folder_S   = '/home/users/dae/gimarso/DESI/JPAS_mock/batches_noisev4/'
U_h5_path  = '/home/users/dae/gimarso/JPAS/JPAS-IDR202406/processed_files/JPAS_training_f50_filtered_APER_COR_3_0_ext.h5'

all_pseudobatch_files = sorted(glob.glob(os.path.join(folder_S, "*.h5")))
num_pseudo_batches = "ALL"
selected_files = all_pseudobatch_files if num_pseudo_batches == "ALL" else all_pseudobatch_files[:num_pseudo_batches]


FilterJPAS = ['J0378','J0390','J0400','J0410','J0420','J0430','J0440','J0450','J0460','J0470',
    'J0480','J0490','J0500','J0510','J0520','J0530','J0540','J0550','J0560','J0570',
    'J0580','J0590','J0600','J0610','J0620','J0630','J0640','J0650','J0660','J0670',
    'J0680','J0690','J0700','J0710','J0720','J0730','J0740','J0750','J0760','J0770',
    'J0780','J0790','J0800','J0810','J0820','J0830','J0840','J0850','J0860','J0870',
    'J0880','J0890','J0900','J0910']

REF_OBSERVATIONS = FilterJPAS + [
    'MAG_i','MAG_G','MAG_R','MAG_Z', 'MORPHTYPE',
    'MAG_W1', 'MAG_W2']

REF_TARGETS = [
    'LOGM', 'LOGG', 'TEFF', 'ALPHAFE', 'FEH',
    'OIII_5007_EW','NII_6584_EW','HBETA_4861_EW','HALPHA_6562_EW',
    'HBETA_CONT', 'HALPHA_CONT', 'Z_GAL','Z_QSO','SPECTYPE','LOGSFR'
]

TOKEN_NAMES = REF_OBSERVATIONS + REF_TARGETS
DA_TOKENS = REF_OBSERVATIONS 
CONTEXT_LENGTH = len(TOKEN_NAMES) - 1

print(f"📊 Resumen de Configuración Detectada:")
print(f"   - Context Length: {CONTEXT_LENGTH}")
print(f"   - Inputs (Observations): {len(REF_OBSERVATIONS)}")
print(f"   - Targets: {len(REF_TARGETS)}")



# ───────────────────────────────────────────────────────────────────
# 2. Helpers to load S pseudobatches (robust to missing columns)
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
    # column in the dataset that includes the sampling probability for eahc object
    return {
        "training_labels": training_labels, "training_labels_err": training_labels_err,
        "class_label": class_label, "morph_labels": morph_labels,
        "obs_names": np.atleast_2d(obs_names), "inputs_probabilities": probs
    }

def pseudo_batch_loader_S(file_list, obs_names):
    for fname in file_list:
        yield load_pseudobatch_S(fname, obs_names)



# ───────────────────────────────────────────────────────────────────
# 3. Build / Resume model
# ───────────────────────────────────────────────────────────────────
num_classes = 3
num_morph_classes = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mixed_precision = True

def _latest_checkpoint(folder: str) -> Optional[int]:
    ckpt_root = os.path.join(folder, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None
    epochs = [int(d.split("_")[1]) for d in os.listdir(ckpt_root) if d.startswith("epoch_")]
    return max(epochs) if epochs else None

model_folder = "./model_OJALA/"

resume_epoch = _latest_checkpoint(model_folder)

# LR
LR_RESUME = 1e-6
LR_FRESH  = 1e-4
DA_LR_SCALE = 0.3  # smaller LR for DA heads

# Scheduler factory
def _scheduler_factory(opt):
    if resume_epoch is None:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000, T_mult=1, eta_min=1e-10)
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=1e-9)

if resume_epoch is not None:
    print(f"🔄  Resuming from epoch {resume_epoch} …")
    nn_model = OJALA.load(
        folder_name       = model_folder,
        checkpoint_epoch  = resume_epoch,
        mixed_precision   = mixed_precision,
        device            = device,
    )
    nn_model.root_folder = os.path.abspath(model_folder)
    # force small LR for resume on all param groups
    for g in nn_model.optimizer.param_groups:
        g["lr"] = LR_RESUME
else:
    print("🚀  Starting a brand-new S+U training run …")
    nn_model = OJALA(
            vocabs               = TOKEN_NAMES,
        context_length       = CONTEXT_LENGTH,
        embedding_dim        = 128,
        num_classes          = num_classes,
        num_morph_classes    = num_morph_classes,
        embedding_activation = "gelu",
        encoder_head_num     = 16,
        encoder_dense_num    = 1024,
        encoder_dropout_rate = 0.1,
        encoder_activation   = "gelu",
        decoder_head_num     = 16,
        decoder_dense_num    = 2048,
        decoder_dropout_rate = 0.1,
        decoder_activation   = "gelu",
        device               = device,
        mixed_precision      = mixed_precision if device != "cpu" else False,
        folder               = model_folder,
    )



# Move to device
nn_model.torch_model = nn_model.torch_model.to(device, memory_format=memory_format)
nn_model.domain_disc = nn_model.domain_disc.to(device)
nn_model.reweighter  = nn_model.reweighter.to(device)

print(f"Total Trainable Parameters (M): {nn_model.get_parameters_sum() / 1e6:.2f}")


# --- Initialize Optimizer (FIX) ---
if nn_model.optimizer is None:
    print("⚙️  Inicializando optimizador (AdamW)...")
    main_params = list(nn_model.torch_model.parameters())
    da_params   = list(nn_model.domain_disc.parameters()) + list(nn_model.reweighter.parameters())
    
    nn_model.optimizer = torch.optim.AdamW([
        {'params': main_params, 'lr': LR_FRESH},
        {'params': da_params,   'lr': LR_FRESH * DA_LR_SCALE}
    ])


# Compile
nn_model.torch_model = torch.compile(nn_model.torch_model, mode="reduce-overhead")




# 5. INVERSE PROBABILITY TOKEN WEIGHTING (MONTE CARLO SIMULATION)

MAX_TOKEN_WEIGHT = 1000.0
CALIB_BATCH_SIZE = 10000 
N_ITER_CALIB = 10         



def calculate_token_weights(file_list, vocab_names, nn_model):
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
            w = 0.0 #
            
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

token_balancing_weights = calculate_token_weights(selected_files, TOKEN_NAMES, nn_model)


# ───────────────────────────────────────────────────────────────────
# 6. Continue Training
# ───────────────────────────────────────────────────────────────────
pseudo_loader_callable_S = lambda: pseudo_batch_loader_S(selected_files, TOKEN_NAMES)

print("▶️  Retomando entrenamiento...")
nn_model.fit(
    pseudo_batch_loader_S     = pseudo_loader_callable_S,
    U_h5_path                 = U_h5_path,
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
    da_token_names            = DA_TOKENS,
    apply_context_masking     = True,
    observations_names        = REF_OBSERVATIONS,
    target_names              = REF_TARGETS,
    keep_frac_min             = 0.,
    keep_frac_max             = 1.0,
    keep_min                  = 5,
    ess_target_frac           = 0.7,
    token_weights             = token_balancing_weights,
    n_tokens_predict          = 5,
)
