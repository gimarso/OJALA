# train_BH_mass_QSO.py

import os
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# --- 0. Configuración ---
warnings.filterwarnings("ignore", message="The given buffer is not writable", category=UserWarning)
torch.set_float32_matmul_precision('high')


from src.model import OJALA


# --- 1. Definición del Modelo y la Función de Pérdida
class Predictor(nn.Module):
    """Modelo para predecir múltiples líneas de emisión. Configurable para salida única (MSE) o dual (Heteroscedastic)."""
    def __init__(self, input_dim, output_dim, loss_type='heteroscedastic', hidden_dim1=128):
        super(Predictor, self).__init__()
        self.loss_type = loss_type
        self.base_net = nn.Sequential(
            nn.Linear(input_dim, 4*hidden_dim1), nn.GELU(), nn.LayerNorm(4*hidden_dim1), nn.Dropout(0.1),
            nn.Linear(4*hidden_dim1, 2*hidden_dim1), nn.GELU(), nn.LayerNorm(2*hidden_dim1), nn.Dropout(0.1),
        )
        self.mean_head = nn.Linear(2*hidden_dim1, output_dim)
        
        if self.loss_type == 'heteroscedastic':
            self.logvar_head = nn.Linear(2*hidden_dim1, output_dim)



    def forward(self, x):
        features = self.base_net(x)
        mean = self.mean_head(features)
        
        if self.loss_type == 'heteroscedastic':
            log_variance = self.logvar_head(features)
            return mean, log_variance
        else:
            return mean

def general_heteroscedastic_loss(y_true: torch.Tensor, y_pred: torch.Tensor, log_variance: torch.Tensor, y_err_true: torch.Tensor) -> torch.Tensor:
    """Pérdida heteroscedástica que incorpora el error de medida observacional."""
    LOGVAR_MIN, LOGVAR_MAX = -20.0, +10.0
    
    safe_logvar = log_variance.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
    model_variance = torch.exp(safe_logvar)
    observation_error_variance = torch.square(y_err_true)
    total_variance = model_variance + observation_error_variance + 1e-6
    
    term1 = torch.square(y_true - y_pred) / total_variance
    term2 = torch.log(total_variance)
    
    return 0.5 * torch.mean(term1 + term2)

# --- 2. Configuración y Parámetros ---
NUM_PSEUDO_BATCHES_TO_PROCESS = 19
FORCE_RECALCULATE_PROJECTIONS = False
FORCE_RECALCULATE_STATS = False


TARGET_VARIABLES = ['LOGMASS_DAS_PAN25'] 
FINETUNING_TASK_NAME = 'BH_model'
LOSS_TYPE = 'heteroscedastic' # Options: 'heteroscedastic', 'mse'

NORMALIZATION_METHOD = 'mean_std' # Options: 'mean_std', 'median_mad'

PRETRAINED_MODEL_PATH = './model_OJALA'
PSEUDO_BATCH_FOLDER = '/home/users/dae/gimarso/DESI/JPAS_mock/batches_noisev4/'


TARGET_ERRORS = [f"{var}_ERR" for var in TARGET_VARIABLES]

pretrained_model_name = os.path.basename(os.path.normpath(PRETRAINED_MODEL_PATH))
PROJECTIONS_DIR = f'./projections/projections_{pretrained_model_name}_task_{FINETUNING_TASK_NAME}/'

STATS_FILE_PATH = os.path.join(PROJECTIONS_DIR, f'normalization_stats_{NORMALIZATION_METHOD}.npz')

FINETUNED_MODEL_DIR = './finetuned_models/'
FINETUNED_MODEL_SUFFIX = f'_{FINETUNING_TASK_NAME}_predictor.pth'
FINETUNED_MODEL_SAVE_PATH = os.path.join(FINETUNED_MODEL_DIR, f"{pretrained_model_name}{FINETUNED_MODEL_SUFFIX}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 10000
EPOCHS = 300
LEARNING_RATE = 1e-4
TEST_SIZE = 0.02
PROCESSING_CHUNK_SIZE = 32768
NUM_WORKERS = min(os.cpu_count(), 4)

FilterJPAS = ['J0378','J0390','J0400','J0410','J0420','J0430','J0440',
'J0450','J0460','J0470','J0480','J0490','J0500','J0510','J0520','J0530',
'J0540','J0550','J0560','J0570','J0580','J0590','J0600','J0610','J0620',
'J0630','J0640','J0650','J0660','J0670','J0680','J0690','J0700','J0710',
'J0720','J0730','J0740','J0750','J0760','J0770','J0780','J0790','J0800',
'J0810','J0820','J0830','J0840','J0850','J0860','J0870','J0880','J0890',
'J0900','J0910']
Context = FilterJPAS + ['MAG_i','MAG_G','MAG_R','MAG_Z','MAG_W1','MAG_W2']

# --- 3. Extracción de Representaciones ---
def extract_and_save_projections(model, source_folder, dest_folder, num_to_process, force_recalculate):
    os.makedirs(dest_folder, exist_ok=True)
    all_source_files = sorted(glob.glob(os.path.join(source_folder, "*.h5")))
    if not all_source_files: raise ValueError(f"No se encontraron ficheros HDF5 en {source_folder}")
    
    files_to_process = all_source_files[:num_to_process] if num_to_process is not None else all_source_files
    
    expected_projections = [os.path.join(dest_folder, os.path.basename(f).replace('.h5', '.npz')) for f in files_to_process]
    if all(os.path.exists(p) for p in expected_projections) and not force_recalculate:
        print("✅ Las proyecciones requeridas ya existen. Saltando extracción.")
        return

    if force_recalculate:
        print(f"💥 Forzando recálculo: eliminando proyecciones existentes...")
        for p in expected_projections:
            if os.path.exists(p): os.remove(p)

    print(f"📂 Extrayendo proyecciones de {len(files_to_process)} fichero(s)...")
    for fname in files_to_process:
        output_path = os.path.join(dest_folder, os.path.basename(fname).replace('.h5', '.npz'))

        file_representations, file_targets, file_target_errors, file_probs = [], [], [], []
        
        with pd.HDFStore(fname, 'r') as store:
            
            storer = store.get_storer('data')
            is_table_format = storer is not None and getattr(storer, 'nrows', None) is not None

            if is_table_format:
                total_rows = storer.nrows
                chunk_iterator = store.select('data', chunksize=PROCESSING_CHUNK_SIZE)
            else:
                full_df = store['data']
                total_rows = len(full_df)
                
                # Generador para simular chunks
                def manual_chunker(df, chunk_size):
                    for start in range(0, len(df), chunk_size):
                        yield df.iloc[start:start + chunk_size]
                
                chunk_iterator = manual_chunker(full_df, PROCESSING_CHUNK_SIZE)

            with tqdm(total=total_rows, desc=f"Procesando {os.path.basename(fname)}") as pbar:
                for df_chunk in chunk_iterator:
                    
                    qso_mask = (
                        (df_chunk['SPECTYPE'] == 2) & 
                        (df_chunk[TARGET_VARIABLES].notna().all(axis=1)) & 
                        (df_chunk[TARGET_ERRORS].notna().all(axis=1)) & (df_chunk['prob'] > 0) & (df_chunk['prob2'] > 0)
                    )
                    df_filtered = df_chunk.loc[qso_mask] 

                    if not df_filtered.empty:
                        X_input = df_filtered[Context].values
                        with torch.no_grad():
                            rep_seq = model.get_encoder_representation(X_input, Context)
                            tok = model.tokenize(np.atleast_2d(Context), len(X_input))
                            _, tok_pad = model.padding(X_input, tok)
                            pad_mask = (tok_pad == 0)
                            rep = torch.from_numpy(rep_seq).to(DEVICE)
                            valid_mask = (~torch.from_numpy(pad_mask).to(DEVICE)).float().unsqueeze(-1)
                            pooled_rep = (rep * valid_mask).sum(1) / valid_mask.sum(1).clamp_min(1e-9)
                            
                            file_representations.append(pooled_rep.cpu().numpy())
                            file_targets.append(df_filtered[TARGET_VARIABLES].values)
                            file_target_errors.append(df_filtered[TARGET_ERRORS].values)
                            file_probs.append(df_filtered['prob'].values)

                    pbar.update(len(df_chunk))

        if file_representations:
            all_representations = np.concatenate(file_representations)
            all_targets = np.concatenate(file_targets)
            all_target_errors = np.concatenate(file_target_errors)
            all_probs = np.concatenate(file_probs)

            renormalized_probs = all_probs / all_probs.sum()

            np.savez_compressed(
                output_path, 
                representations=all_representations, 
                targets=all_targets,
                target_errors=all_target_errors,
                probabilities=renormalized_probs
            )
            print(f"✅ Fichero de proyecciones guardado en {output_path}")

class ProjectionDatasetInMemory(Dataset):
    def __init__(self, file_paths):
        all_representations, all_targets, all_target_errors, all_probs = [], [], [], []
        print("🧠 Cargando dataset en memoria (RAM)...")
        for path in tqdm(file_paths, desc="Cargando ficheros"):
            with np.load(path) as data:
                all_representations.append(data['representations'])
                all_targets.append(data['targets'])
                all_target_errors.append(data['target_errors'])
                all_probs.append(data['probabilities'])
        
        self.representations = torch.from_numpy(np.concatenate(all_representations, axis=0)).float()
        self.targets = torch.from_numpy(np.concatenate(all_targets, axis=0)).float()
        self.target_errors = torch.from_numpy(np.concatenate(all_target_errors, axis=0)).float()
        self.probabilities = torch.from_numpy(np.concatenate(all_probs, axis=0)).float()
        
        self.global_weights = self.probabilities / len(file_paths)
        
        self.is_standardized = False
        self.normalization_method = None
        self.targets_mean = None
        self.targets_std = None
        self.targets_median = None
        self.targets_mad = None

        print(f"✅ Dataset cargado. Total de muestras: {len(self.representations)}")

    def __len__(self): return len(self.representations)
    
    def __getitem__(self, idx): 
        return self.representations[idx], self.targets[idx], self.target_errors[idx]

    def standardize_targets(self, method: str, stat1: torch.Tensor, stat2: torch.Tensor):
        """Estandariza los targets y errores usando el método y estadísticas dadas."""
        if self.is_standardized:
            print("⚠️ Los targets ya estaban estandarizados. Saltando operación.")
            return
            
        self.normalization_method = method
        
        if self.normalization_method == 'mean_std':
            self.targets_mean = stat1.to(self.targets.device)
            self.targets_std = stat2.to(self.targets.device).clamp_min(1e-6) 
            
            self.targets = (self.targets - self.targets_mean) / self.targets_std
            self.target_errors = self.target_errors / self.targets_std
            print("✅ Targets y errores estandarizados (mean/std).")

        elif self.normalization_method == 'median_mad':
            self.targets_median = stat1.to(self.targets.device)
            self.targets_mad = stat2.to(self.targets.device).clamp_min(1e-6) 
            
            self.targets = (self.targets - self.targets_median) / self.targets_mad
            self.target_errors = self.target_errors / self.targets_mad
            print("✅ Targets y errores estandarizados (median/MAD).")
        
        else:
            raise ValueError(f"Método de normalización desconocido: {self.normalization_method}")

        self.is_standardized = True


if __name__ == '__main__':
    print(f"Usando dispositivo: {DEVICE} | Workers para datos: {NUM_WORKERS}")
    print(f"Usando Loss Type: {LOSS_TYPE}")
    
    print("\n--- Fase 1: Extracción de Proyecciones ---")
    nn_model = OJALA.load(PRETRAINED_MODEL_PATH, device=DEVICE)
    extract_and_save_projections(nn_model, PSEUDO_BATCH_FOLDER, PROJECTIONS_DIR, NUM_PSEUDO_BATCHES_TO_PROCESS, FORCE_RECALCULATE_PROJECTIONS)
    del nn_model; torch.cuda.empty_cache()

    print("\n--- Fase 2: Preparación del Dataset ---")
    all_projection_files = sorted(glob.glob(os.path.join(PROJECTIONS_DIR, "pseudo_batch_*.npz")))
    
    files_to_load = all_projection_files[:NUM_PSEUDO_BATCHES_TO_PROCESS] if NUM_PSEUDO_BATCHES_TO_PROCESS is not None else all_projection_files
    
    print(f"Se encontraron {len(all_projection_files)} proyecciones, se cargarán {len(files_to_load)} en memoria.")
    if not files_to_load:
        raise ValueError(f"La lista de ficheros a cargar está vacía. Revisa la ruta: {PROJECTIONS_DIR}")

    print("\n--- Fase 2b: Cálculo o Carga de Parámetros de Normalización ---")
    print(f"Usando método de normalización: {NORMALIZATION_METHOD}")

    if len(files_to_load) > 1:
        train_files, _ = train_test_split(files_to_load, test_size=TEST_SIZE, random_state=42)
    else:
        print("⚠️ Solo se encontró un fichero de proyecciones. Se usará entero para calcular las estadísticas.")
        train_files = files_to_load

    if os.path.exists(STATS_FILE_PATH) and not FORCE_RECALCULATE_STATS:
        print(f"✅ Cargando estadísticas de normalización desde {STATS_FILE_PATH}")
        stats = np.load(STATS_FILE_PATH)
        
        if NORMALIZATION_METHOD == 'mean_std':
            targets_stat1 = torch.from_numpy(stats['mean']).float()
            targets_stat2 = torch.from_numpy(stats['std']).float()
        elif NORMALIZATION_METHOD == 'median_mad':
            targets_stat1 = torch.from_numpy(stats['median']).float()
            targets_stat2 = torch.from_numpy(stats['mad']).float()
        else:
            raise ValueError(f"Método de normalización desconocido: {NORMALIZATION_METHOD}")

    else:
        if FORCE_RECALCULATE_STATS and os.path.exists(STATS_FILE_PATH):
            print("💥 Forzando recálculo: eliminando estadísticas existentes...")
            os.remove(STATS_FILE_PATH)

        if not train_files:
             raise ValueError("La lista de ficheros de entrenamiento para calcular estadísticas está vacía.")

        if NORMALIZATION_METHOD == 'mean_std':
            print("🧠 Calculando estadísticas de normalización (mean/std, modo seguro para memoria)...")
            
            n_samples = 0
            with np.load(train_files[0]) as data:
                n_features = data['targets'].shape[1] 
            
            sum_x = np.zeros(n_features, dtype=np.float64)
            sum_x_sq = np.zeros(n_features, dtype=np.float64)

            for file_path in tqdm(train_files, desc="Calculando Estadísticas (mean/std)"):
                with np.load(file_path) as data:
                    targets_chunk = data['targets'].astype(np.float64)
                    n_samples += len(targets_chunk)
                    sum_x += np.sum(targets_chunk, axis=0)
                    sum_x_sq += np.sum(np.square(targets_chunk), axis=0)

            if n_samples == 0:
                raise ValueError("No se encontraron muestras en los ficheros de entrenamiento para calcular estadísticas.")
            
            mean_np = sum_x / n_samples
            variance_np = (sum_x_sq / n_samples) - np.square(mean_np)
            std_np = np.sqrt(variance_np)
            std_np[std_np < 1e-6] = 1.0 

            targets_stat1 = torch.from_numpy(mean_np).float()
            targets_stat2 = torch.from_numpy(std_np).float()
            
            np.savez(STATS_FILE_PATH, mean=targets_stat1.numpy(), std=targets_stat2.numpy())
            print(f"✅ Estadísticas (mean/std) guardadas en {STATS_FILE_PATH}")

        elif NORMALIZATION_METHOD == 'median_mad':
            print("🧠 Calculando estadísticas de normalización (median/MAD)...")
            print("   ⚠️ Esto cargará todos los targets de 'train' en memoria.")
            
            all_train_targets = []
            for file_path in tqdm(train_files, desc="Cargando targets para (median/MAD)"):
                with np.load(file_path) as data:
                    all_train_targets.append(data['targets'])
            
            if not all_train_targets:
                raise ValueError("No se encontraron muestras en los ficheros de entrenamiento para calcular estadísticas.")

            targets_tensor = torch.from_numpy(np.concatenate(all_train_targets, axis=0)).float()
            
            median_torch = torch.median(targets_tensor, dim=0).values
            mad_torch = torch.median(torch.abs(targets_tensor - median_torch), dim=0).values
            mad_torch[mad_torch < 1e-6] = 1.0 

            targets_stat1 = median_torch
            targets_stat2 = mad_torch
            
            np.savez(STATS_FILE_PATH, median=targets_stat1.numpy(), mad=targets_stat2.numpy())
            print(f"✅ Estadísticas (median/MAD) guardadas en {STATS_FILE_PATH}")
        
        else:
            raise ValueError(f"Método de normalización desconocido: {NORMALIZATION_METHOD}")
    
    if NORMALIZATION_METHOD == 'mean_std':
        print(f"Media de los targets (train): {targets_stat1.numpy()}")
        print(f"Std de los targets (train):  {targets_stat2.numpy()}")
    elif NORMALIZATION_METHOD == 'median_mad':
        print(f"Mediana de los targets (train): {targets_stat1.numpy()}")
        print(f"MAD de los targets (train):     {targets_stat2.numpy()}")

    full_dataset = ProjectionDatasetInMemory(files_to_load)
    
    full_dataset.standardize_targets(NORMALIZATION_METHOD, targets_stat1, targets_stat2)
    
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=TEST_SIZE, random_state=42)
    
    # --- CAMBIO: Inverse Weighting Controlado ("Top-Hat con Alas Suaves") ---
    print("⚖️ Calculando pesos: Hybrid Inverse Weighting + Strong Decay...")
    
    # 1. Obtener targets y deshacer estandarización para trabajar en unidades físicas
    train_targets_norm = full_dataset.targets[train_indices].cpu().numpy().flatten()
    
    if NORMALIZATION_METHOD == 'mean_std':
        norm_shift = targets_stat1.item()
        norm_scale = targets_stat2.item()
    elif NORMALIZATION_METHOD == 'median_mad':
        norm_shift = targets_stat1.item()
        norm_scale = targets_stat2.item()
    
    train_values_real = (train_targets_norm * norm_scale) + norm_shift
    

    TARGET_MIN = 6.5
    TARGET_MAX = 9.5
    DECAY_SIGMA = 0.15 
    
    target_density = np.ones_like(train_values_real)
    mask_left = train_values_real < TARGET_MIN
    mask_right = train_values_real > TARGET_MAX
    
    target_density[mask_left] = np.exp(-0.5 * ((train_values_real[mask_left] - TARGET_MIN) / DECAY_SIGMA)**2)
    target_density[mask_right] = np.exp(-0.5 * ((train_values_real[mask_right] - TARGET_MAX) / DECAY_SIGMA)**2)
    

    counts, bin_edges = np.histogram(train_values_real, bins=100, density=True)
    
    bin_indices = np.digitize(train_values_real, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(counts) - 1)
    
    sample_densities = counts[bin_indices] + 1e-9

    weights_raw = target_density / sample_densities
    

    weights_raw = weights_raw / np.median(weights_raw) 
    weights_raw = np.clip(weights_raw, 0, 20.0)        

    train_weights = torch.from_numpy(weights_raw).float()
    print(f"   Pesos calculados (Strict Decay): Min {train_weights.min():.4e}, Max {train_weights.max():.4e}")
    
    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_indices), replacement=True)
    
    print("📊 Generando plot de verificación del weighting (PDF) con valores ORIGINALES...")
    
    num_check_samples = 10000
    check_indices_tensor = torch.multinomial(train_weights, num_check_samples, replacement=True)
    sampled_values_check = train_values_real[check_indices_tensor.numpy()]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(train_values_real, bins=50, density=True, alpha=0.5, 
            label='Original Distribution', color='cyan', histtype='stepfilled', linewidth=2)
    
    ax.hist(sampled_values_check, bins=50, density=True, alpha=0.5, 
            label=f'Target Profile (Flat {TARGET_MIN}-{TARGET_MAX}, sig={DECAY_SIGMA})', color='magenta', histtype='stepfilled', linewidth=2)
    
    ax.axvline(TARGET_MIN, color='white', linestyle='--', alpha=0.5, label='Flat Region Bound')
    ax.axvline(TARGET_MAX, color='white', linestyle='--', alpha=0.5)

    ax.set_xlabel(f'{TARGET_VARIABLES[0]} (Original Units)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(f'Target Sampling Profile: Flat + Hard Decay', fontsize=16)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(False)
    ax.legend(loc='lower left', fontsize='large')
    
    plot_filename = "check_weighting_distribution.pdf"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot de verificación guardado en: {plot_filename}")

    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    print("\n--- Fase 3: Entrenamiento del Modelo ---")
    sample_repr, _, _ = full_dataset[0]
    input_dim = sample_repr.shape[0]
    output_dim = len(TARGET_VARIABLES) 
    print(f"Input dim: {input_dim} | Output dim: {output_dim} (Target: {TARGET_VARIABLES})")

    model = Predictor(input_dim=input_dim, output_dim=output_dim, loss_type=LOSS_TYPE)
    
    model_loaded = False
    if os.path.exists(FINETUNED_MODEL_SAVE_PATH):
        print(f"🔍 Verificando compatibilidad del modelo guardado: {FINETUNED_MODEL_SAVE_PATH}")
        try:
            checkpoint = torch.load(FINETUNED_MODEL_SAVE_PATH, map_location='cpu')
            
            ckpt_input_dim = checkpoint.get('base_net.0.weight', torch.empty(0, 0)).shape[1]
            ckpt_output_dim = checkpoint.get('mean_head.weight', torch.empty(0, 0)).shape[0]
            
            is_heteroscedastic_ckpt = 'logvar_head.weight' in checkpoint
            is_heteroscedastic_current = (LOSS_TYPE == 'heteroscedastic')

            if input_dim == ckpt_input_dim and output_dim == ckpt_output_dim and (is_heteroscedastic_ckpt == is_heteroscedastic_current):
                print("✅ Modelo compatible. Reanudando entrenamiento...")
                model.load_state_dict(checkpoint)
                model_loaded = True
            else:
                print("⚠️ ¡Incompatibilidad de arquitectura detectada!")
                print(f"   - Dimensión de entrada esperada: {input_dim}, en checkpoint: {ckpt_input_dim}")
                print(f"   - Dimensión de salida esperada: {output_dim}, en checkpoint: {ckpt_output_dim}")
                print(f"   - Loss Type compatible: {is_heteroscedastic_ckpt == is_heteroscedastic_current} (Checkpoint Heteroscedastic: {is_heteroscedastic_ckpt}, Current: {is_heteroscedastic_current})")
                print("   - Se ignorará el fichero del modelo y se entrenará uno nuevo desde cero.")
        
        except Exception as e:
            print(f"🚨 No se pudo cargar o verificar el modelo guardado. Error: {e}. Se entrenará uno nuevo.")

    if not model_loaded:
        print(f"🚀 Creando un nuevo modelo {type(model).__name__} desde cero (Loss: {LOSS_TYPE}).")
    
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=5, verbose=True)
    
    os.makedirs(FINETUNED_MODEL_DIR, exist_ok=True)
    
    best_val_loss = float('inf')
    mse_loss_fn = nn.MSELoss()

    if model_loaded:
        print("    Calculando la 'best_val_loss' actual del modelo cargado...")
        model.eval()
        initial_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, target_errors in val_loader:
                inputs, targets, target_errors = inputs.to(DEVICE), targets.to(DEVICE), target_errors.to(DEVICE)
                
                if LOSS_TYPE == 'heteroscedastic':
                    outputs_mean, outputs_logvar = model(inputs)
                    loss = general_heteroscedastic_loss(targets, outputs_mean, outputs_logvar, target_errors)
                else: # mse
                    outputs_mean = model(inputs)
                    loss = mse_loss_fn(outputs_mean, targets)
                
                initial_val_loss += loss.item() * inputs.size(0)
        
        if val_indices:
            best_val_loss = initial_val_loss / len(val_indices)
        
        print(f"    Pérdida de validación inicial: {best_val_loss:.6f}")

    print(f"🚀 Iniciando fine-tuning para {EPOCHS} épocas...")
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets, target_errors in pbar:
            inputs, targets, target_errors = inputs.to(DEVICE), targets.to(DEVICE), target_errors.to(DEVICE)
            optimizer.zero_grad()
            
            if LOSS_TYPE == 'heteroscedastic':
                outputs_mean, outputs_logvar = model(inputs)
                loss = general_heteroscedastic_loss(targets, outputs_mean, outputs_logvar, target_errors)
            else: 
                outputs_mean = model(inputs)
                loss = mse_loss_fn(outputs_mean, targets)

            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, target_errors in val_loader:
                inputs, targets, target_errors = inputs.to(DEVICE), targets.to(DEVICE), target_errors.to(DEVICE)
                
                if LOSS_TYPE == 'heteroscedastic':
                    outputs_mean, outputs_logvar = model(inputs)
                    loss = general_heteroscedastic_loss(targets, outputs_mean, outputs_logvar, target_errors)
                else:
                    outputs_mean = model(inputs)
                    loss = mse_loss_fn(outputs_mean, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        if val_indices:
            val_loss /= len(val_indices)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Val Loss: {val_loss:.6f}")
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), FINETUNED_MODEL_SAVE_PATH)
            print(f"-> ✅ Modelo guardado en {FINETUNED_MODEL_SAVE_PATH} (Mejor Val Loss: {best_val_loss:.6f})")

    print("\n🎉 Entrenamiento finalizado.")