#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import gc
import argparse
import warnings
import numpy as np
import pandas as pd

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# ---------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
CATALOGUE_DIR = DATA_DIR / "catalogues"
MODEL_DIR = REPO_ROOT / "model_OJALA"

DEFAULT_INPUT_FILE = CATALOGUE_DIR / "JPAS_EDR_photometry.csv"
DEFAULT_OUTPUT_DIR = CATALOGUE_DIR
DEFAULT_MODEL_PATH = MODEL_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import OJALA  

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# FILTERS / FEATURES
# ---------------------------------------------------------------------
FilterJPAS = [
    'J0378', 'J0390', 'J0400', 'J0410', 'J0420', 'J0430', 'J0440', 'J0450', 'J0460', 'J0470',
    'J0480', 'J0490', 'J0500', 'J0510', 'J0520', 'J0530', 'J0540', 'J0550', 'J0560', 'J0570',
    'J0580', 'J0590', 'J0600', 'J0610', 'J0620', 'J0630', 'J0640', 'J0650', 'J0660', 'J0670',
    'J0680', 'J0690', 'J0700', 'J0710', 'J0720', 'J0730', 'J0740', 'J0750', 'J0760', 'J0770',
    'J0780', 'J0790', 'J0800', 'J0810', 'J0820', 'J0830', 'J0840', 'J0850', 'J0860', 'J0870',
    'J0880', 'J0890', 'J0900', 'J0910'
]

MAG_SED = ['MAG_i', 'MAG_G', 'MAG_R', 'MAG_Z', 'MAG_W1', 'MAG_W2']

MODEL_REQUEST_PARAMS = [
    'HALPHA_CONT',
    'HBETA_CONT',
    'Z_GAL',
    'Z_QSO',
    'HBETA_4861_EW',
    'HALPHA_6562_EW',
    'OIII_5007_EW',
    'NII_6584_EW',
    'LOGM',
    'LOGSFR',
    'TEFF',
    'LOGG',
    'ALPHAFE',
    'FEH'
]

CLASS_NAMES = ['GALAXY', 'STAR', 'QSO']

EXTRA_PHYSICAL_PARAMS = ['LOGM', 'LOGSFR', 'TEFF', 'LOGG', 'ALPHAFE', 'FEH']
# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def clean_array(arr):
    arr = np.asarray(arr, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_context_columns(jpas_prefix):
    """
    Build the input column names from the chosen JPAS photometry prefix.

    Returns
    -------
    jpas_input_cols : list
        Real column names in the CSV, e.g. AUTO_J0660 or APER_COR_3_0_J0660.
    model_context_cols : list
        Vocabulary expected by OJALA, e.g. J0660, ..., MAG_i, ...
    rename_map : dict
        Mapping from CSV column names to OJALA vocabulary names.
    """
    jpas_input_cols = [f"{jpas_prefix}_{filt}" for filt in FilterJPAS]
    model_context_cols = FilterJPAS + MAG_SED
    rename_map = {f"{jpas_prefix}_{filt}": filt for filt in FilterJPAS}
    return jpas_input_cols, model_context_cols, rename_map


def validate_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in the input CSV:\n" + "\n".join(missing)
        )

def batch_predict(model, feats_df, model_context_cols, batch_size=20000):
    """
    Run OJALA predictions in batches.

    Parameters
    ----------
    model : OJALA
        Loaded OJALA model.
    feats_df : pandas.DataFrame
        Feature matrix whose columns already match the OJALA vocabulary.
    model_context_cols : list
        Vocabulary expected by the model, e.g. J0378, ..., MAG_i, ...
    batch_size : int
        Batch size for prediction.

    Returns
    -------
    preds_df : pandas.DataFrame
        Regression predictions.
    errs_df : pandas.DataFrame
        Regression uncertainties.
    cls_prob_df : pandas.DataFrame
        Class probabilities.
    cls_label_series : pandas.Series
        Final class label from argmax over class probabilities.
    """
    n = len(feats_df)
    preds_list, errs_list = [], []
    cls_probs_list = []

    all_X = feats_df[model_context_cols].values

    for i in range(0, n, batch_size):
        print(f"Processing batch {i:,} - {min(i + batch_size, n):,} of {n:,}")
        X = all_X[i:i + batch_size]

        model.perceive(X, model_context_cols)
        out = model.request(['SPECTYPE'] + MODEL_REQUEST_PARAMS)

        preds_list.append(np.array(out['regression_preds'], copy=True))
        errs_list.append(np.array(out['regression_preds_err'], copy=True))
        cls_probs_list.append(np.array(out['main_class_probs'], copy=True))

        del out, X
        gc.collect()

    preds = np.concatenate(preds_list, axis=0)
    errs = np.concatenate(errs_list, axis=0)
    cls_probs = np.concatenate(cls_probs_list, axis=0)
    cls_probs = np.asarray(cls_probs)

    # Convert possible shapes such as (N, 1, 3) into (N, 3)
    if cls_probs.ndim == 3 and cls_probs.shape[1] == 1:
        cls_probs = cls_probs[:, 0, :]
    elif cls_probs.ndim == 1:
        cls_probs = cls_probs.reshape(-1, len(CLASS_NAMES))
    elif cls_probs.ndim != 2:
        raise ValueError(
            f"Unexpected shape for main_class_probs: {cls_probs.shape}"
        )

    if cls_probs.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Unexpected shape for main_class_probs after reshaping: {cls_probs.shape}. "
            f"Expected second dimension = {len(CLASS_NAMES)}."
        )

    preds_df = pd.DataFrame(preds, columns=MODEL_REQUEST_PARAMS, index=feats_df.index)
    errs_df = pd.DataFrame(
        errs,
        columns=[f"{c}_ERR" for c in MODEL_REQUEST_PARAMS],
        index=feats_df.index
    )

    cls_prob_df = pd.DataFrame(
        cls_probs,
        columns=[f"P_{c}" for c in CLASS_NAMES],
        index=feats_df.index
    )

    cls_idx = np.argmax(cls_probs, axis=1)
    cls_label_series = pd.Series(
        [CLASS_NAMES[k] for k in cls_idx],
        index=feats_df.index,
        name="CLASS"
    )

    return preds_df, errs_df, cls_prob_df, cls_label_series
# ---------------------------------------------------------------------
# SFR FUNCTIONS
# ---------------------------------------------------------------------

def calculate_sfr(ha_flux_obs, ha_hb_ratio, redshift, cosmo_model):
    """
    Compute log10 SFR from observed Halpha flux using:
    - Balmer decrement dust correction with the Calzetti law
    - Kennicutt (1998) conversion: SFR = 7.9e-42 * L_Halpha
    This keeps the Salpeter IMF convention used by Kennicutt (1998).

    Parameters
    ----------
    ha_flux_obs : array-like
        Observed Halpha flux in erg / s / cm^2.
    ha_hb_ratio : array-like
        Observed Halpha/Hbeta flux ratio.
    redshift : array-like
        Redshift.
    cosmo_model : astropy cosmology
        Cosmology instance.

    Returns
    -------
    log_sfr : ndarray
        log10(SFR / Msun yr^-1).
    """
    BALMER_INTRINSIC = 2.86
    KENNICUTT_FACTOR = 7.9e-42

    # Calzetti law values at Halpha and Hbeta
    k_ha = 3.326
    k_hb = 4.598
    dust_exp = k_ha / (k_hb - k_ha)

    ha_flux_obs = np.asarray(ha_flux_obs, dtype=float)
    ha_hb_ratio = np.asarray(ha_hb_ratio, dtype=float)
    redshift = np.asarray(redshift, dtype=float)

    log_sfr = np.full_like(ha_flux_obs, np.nan, dtype=float)

    valid = (
        np.isfinite(ha_flux_obs) & (ha_flux_obs > 0) &
        np.isfinite(ha_hb_ratio) & (ha_hb_ratio > 0) &
        np.isfinite(redshift) & (redshift >= 0)
    )
    if not np.any(valid):
        return log_sfr

    ratio = np.maximum(ha_hb_ratio[valid], BALMER_INTRINSIC)
    ha_flux_corr = ha_flux_obs[valid] * (ratio / BALMER_INTRINSIC) ** dust_exp

    dl_cm = cosmo_model.luminosity_distance(redshift[valid]).to(u.cm).value
    lum_ha = 4.0 * np.pi * dl_cm**2 * ha_flux_corr
    sfr = KENNICUTT_FACTOR * lum_ha

    with np.errstate(divide="ignore", invalid="ignore"):
        log_sfr_valid = np.log10(sfr)

    log_sfr[valid] = log_sfr_valid
    log_sfr[~np.isfinite(log_sfr)] = np.nan

    return log_sfr


def calculate_sfr_error(
    ha_flux_cont,
    ha_ew,
    hb_flux_cont,
    hb_ew,
    z,
    ha_flux_cont_err,
    ha_ew_err,
    hb_flux_cont_err,
    hb_ew_err,
    z_err,
    cosmo_model,
):
    """
    Estimate the uncertainty in log10 SFR(Halpha) assuming:
    - F_Ha_obs ∝ ha_flux_cont * ha_ew
    - F_Hb_obs ∝ hb_flux_cont * hb_ew
    - Dust correction from Balmer decrement with Calzetti law
    - L_Ha ∝ F_Ha_corr * d_L^2

    Notes
    -----
    This returns the propagated uncertainty in dex.
    The common absolute photometric normalization cancels out in the Balmer ratio
    and is not included here.

    Parameters
    ----------
    ha_flux_cont, hb_flux_cont : array-like
        Continuum terms used to reconstruct line fluxes.
    ha_ew, hb_ew : array-like
        Rest-frame equivalent widths in Angstrom.
    z : array-like
        Redshift.
    *_err : array-like
        Corresponding 1-sigma uncertainties.
    cosmo_model : astropy cosmology
        Cosmology instance.

    Returns
    -------
    sigma_log10 : ndarray
        Uncertainty in log10(SFR), in dex.
    """
    BALMER_INTRINSIC = 2.86

    # Calzetti law values at Halpha and Hbeta
    k_ha = 3.326
    k_hb = 4.598
    dust_exp = k_ha / (k_hb - k_ha)

    def clean(arr):
        arr = np.asarray(arr, dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def rel_err(val, err):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.abs(err / val)
        out[~np.isfinite(out)] = 0.0
        return out

    ha_c = clean(ha_flux_cont)
    ha_ew_val = clean(ha_ew)
    hb_c = clean(hb_flux_cont)
    hb_ew_val = clean(hb_ew)
    z_val = clean(z)

    ha_c_err = clean(ha_flux_cont_err)
    ha_ew_err_val = clean(ha_ew_err)
    hb_c_err = clean(hb_flux_cont_err)
    hb_ew_err_val = clean(hb_ew_err)
    z_err_val = clean(z_err)

    sigma_log10 = np.full_like(z_val, np.nan, dtype=float)

    # Relative errors of reconstructed observed line fluxes
    rel_fha = np.sqrt(rel_err(ha_c, ha_c_err) ** 2 + rel_err(ha_ew_val, ha_ew_err_val) ** 2)
    rel_fhb = np.sqrt(rel_err(hb_c, hb_c_err) ** 2 + rel_err(hb_ew_val, hb_ew_err_val) ** 2)

    # Observed Balmer ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (ha_c * ha_ew_val) / (hb_c * hb_ew_val)
    ratio[~np.isfinite(ratio)] = np.nan

    valid = (
        np.isfinite(ratio) & (ratio > 0) &
        np.isfinite(z_val) & (z_val >= 0)
    )
    if not np.any(valid):
        return sigma_log10

    # Only ratios above the intrinsic value contribute to dust-correction uncertainty
    active_dust = valid & (ratio > BALMER_INTRINSIC)

    # Relative uncertainty of the Balmer ratio
    rel_ratio = np.sqrt(rel_fha**2 + rel_fhb**2)

    # Relative uncertainty of the corrected Halpha flux:
    # F_Ha_corr = F_Ha_obs * (ratio / 2.86)^dust_exp
    rel_fha_corr = rel_fha.copy()
    rel_fha_corr[active_dust] = np.sqrt(
        rel_fha[active_dust] ** 2 +
        (dust_exp * rel_ratio[active_dust]) ** 2 +
        2.0 * dust_exp * rel_fha[active_dust] ** 2
    )
    # The covariance term above appears because ratio contains F_Ha_obs itself:
    # ln(Fcorr) = (1 + dust_exp) ln(FHa) - dust_exp ln(FHb) + const
    # so equivalently:
    rel_fha_corr[active_dust] = np.sqrt(
        ((1.0 + dust_exp) * rel_fha[active_dust]) ** 2 +
        (dust_exp * rel_fhb[active_dust]) ** 2
    )

    # Cosmological distance uncertainty term
    eps = 1e-4
    z_safe = np.maximum(z_val, 0.0)

    dl = cosmo_model.luminosity_distance(z_safe).to(u.cm).value
    dl_plus = cosmo_model.luminosity_distance(z_safe + eps).to(u.cm).value

    with np.errstate(divide="ignore", invalid="ignore"):
        dln_dl_dz = (dl_plus - dl) / (eps * dl)

    dln_dl_dz[~np.isfinite(dln_dl_dz)] = 0.0

    rel_lha_from_z = 2.0 * dln_dl_dz * z_err_val

    total_rel = np.sqrt(rel_fha_corr**2 + rel_lha_from_z**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_log10_valid = total_rel[valid] / np.log(10.0)

    sigma_log10[valid] = sigma_log10_valid
    sigma_log10[~np.isfinite(sigma_log10)] = np.nan

    return sigma_log10


def calculate_ha_luminosity_corrected(ha_flux_obs, ha_hb_ratio, redshift, cosmo_model):
    """
    Return the dust-corrected Halpha luminosity using:
    - Balmer decrement
    - Calzetti attenuation law

    Parameters
    ----------
    ha_flux_obs : array-like
        Observed Halpha flux in erg / s / cm^2.
    ha_hb_ratio : array-like
        Observed Halpha/Hbeta flux ratio.
    redshift : array-like
        Redshift.
    cosmo_model : astropy cosmology
        Cosmology instance.

    Returns
    -------
    lum_ha : ndarray
        Dust-corrected Halpha luminosity in erg / s.
    """
    BALMER_INTRINSIC = 2.86

    # Calzetti law values at Halpha and Hbeta
    k_ha = 3.326
    k_hb = 4.598
    dust_exp = k_ha / (k_hb - k_ha)

    ha_flux_obs = np.asarray(ha_flux_obs, dtype=float)
    ha_hb_ratio = np.asarray(ha_hb_ratio, dtype=float)
    redshift = np.asarray(redshift, dtype=float)

    lum_ha = np.full_like(ha_flux_obs, np.nan, dtype=float)

    valid = (
        np.isfinite(ha_flux_obs) & (ha_flux_obs > 0) &
        np.isfinite(ha_hb_ratio) & (ha_hb_ratio > 0) &
        np.isfinite(redshift) & (redshift >= 0)
    )
    if not np.any(valid):
        return lum_ha

    ratio = np.maximum(ha_hb_ratio[valid], BALMER_INTRINSIC)
    ha_flux_corr = ha_flux_obs[valid] * (ratio / BALMER_INTRINSIC) ** dust_exp

    dl_cm = cosmo_model.luminosity_distance(redshift[valid]).to(u.cm).value
    lum_ha[valid] = 4.0 * np.pi * dl_cm**2 * ha_flux_corr
    lum_ha[~np.isfinite(lum_ha)] = np.nan

    return lum_ha






def drop_unwanted_input_columns(df):
    """
    Drop unwanted input columns if they exist.

    Rules
    -----
    Drop any column whose name:
    - starts with 'MASK_FLAGS_'
    - starts with 'FLAGS_'
    - starts with 'FLUX_APER_COR_6_0'
    - starts with 'MAG_I'
    """
    prefixes_to_drop = [
        "MASK_FLAGS_",
        "FLAGS_",
        "FLUX_APER_COR_6_0",
        "MAG_I",
    ]

    cols_to_drop = [
        col for col in df.columns
        if any(col.startswith(prefix) for prefix in prefixes_to_drop)
    ]

    if cols_to_drop:
        print(f"\nDropping {len(cols_to_drop)} unwanted input columns...")
        df = df.drop(columns=cols_to_drop, errors="ignore")

    return df
    

def main():
    parser = argparse.ArgumentParser(description="Generate an OJALA catalogue from a JPAS CSV file.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_FILE), help="Input CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to the OJALA checkpoint")
    parser.add_argument("--batch-size", type=int, default=20000, help="Prediction batch size")
    parser.add_argument(
        "--jpas-photometry",
        type=str,
        default="APER_COR_3_0",
        help="JPAS photometry prefix, e.g. AUTO or APER_COR_3_0"
    )
    parser.add_argument(
        "--isdss-col",
        type=str,
        default="AUTO_iSDSS",
        help="Column used as iSDSS photometry for line flux calculations"
    )
    args = parser.parse_args()

    input_csv = Path(args.input).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path is not None else None

    if args.output is None:
        base = input_csv.stem
        output_csv = input_csv.parent / f"{base}_{args.jpas_photometry}_OJALA_catalog.csv"
    else:
        output_csv = Path(args.output).expanduser().resolve()


    print("\nReading input catalogue...")
    df = pd.read_csv(input_csv)
    print(f"Number of objects before sampling: {len(df):,}")
    print(f"Number of columns before cleaning: {len(df.columns):,}")

    df = drop_unwanted_input_columns(df)

    print(f"Number of columns after cleaning: {len(df.columns):,}")

        # Keep the 1% test sample used in your current script version.
    #df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    #print(f"Number of objects after sampling: {len(df):,}")

    jpas_input_cols, model_context_cols, rename_map = build_context_columns(args.jpas_photometry)

    required_cols = jpas_input_cols + MAG_SED + [args.isdss_col]
    validate_columns(df, required_cols)

    print("\nBuilding feature matrix...")
    feats_df = df[jpas_input_cols + MAG_SED].copy()

    # Convert to numeric without filtering NaNs.
    for c in feats_df.columns:
        feats_df[c] = pd.to_numeric(feats_df[c], errors="coerce")

    # Rename JPAS photometry columns to the vocabulary expected by OJALA.
    feats_df = feats_df.rename(columns=rename_map)

    # Reorder explicitly to match the model vocabulary.
    feats_df = feats_df[model_context_cols]

    phot_i = pd.to_numeric(df[args.isdss_col], errors="coerce").values

    print("\nLoading OJALA model...")
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = OJALA.load(str(model_path), device=device)
    print("\nRunning batched prediction...")
    preds_df, errs_df, cls_prob_df, cls_label = batch_predict(
        model=model,
        feats_df=feats_df,
        model_context_cols=model_context_cols,
        batch_size=args.batch_size
    )

    # Keep this commented for debugging when needed.
    # import pdb; pdb.set_trace()

    print("\nComputing derived quantities...")

    z_gal = clean_array(preds_df['Z_GAL'].values)
    z_gal_err = clean_array(errs_df['Z_GAL_ERR'].values)

    halpha_cont = clean_array(preds_df['HALPHA_CONT'].values)
    halpha_cont_err = clean_array(errs_df['HALPHA_CONT_ERR'].values)

    hbeta_cont = clean_array(preds_df['HBETA_CONT'].values)
    hbeta_cont_err = clean_array(errs_df['HBETA_CONT_ERR'].values)

    halpha_ew = clean_array(preds_df['HALPHA_6562_EW'].values)
    halpha_ew_err = clean_array(errs_df['HALPHA_6562_EW_ERR'].values)

    hbeta_ew = clean_array(preds_df['HBETA_4861_EW'].values)
    hbeta_ew_err = clean_array(errs_df['HBETA_4861_EW_ERR'].values)

    oiii_ew = clean_array(preds_df['OIII_5007_EW'].values)
    oiii_ew_err = clean_array(errs_df['OIII_5007_EW_ERR'].values)

    nii_ew = clean_array(preds_df['NII_6584_EW'].values)
    nii_ew_err = clean_array(errs_df['NII_6584_EW_ERR'].values)

    phot_i_clean = clean_array(phot_i)
    one_plus_z = 1.0 + np.maximum(z_gal, 0.0)

    # Observed line fluxes.
    flux_halpha = halpha_cont * halpha_ew * one_plus_z * phot_i_clean * 1e-19
    flux_hbeta = hbeta_cont * hbeta_ew * one_plus_z * phot_i_clean * 1e-19
    flux_oiii5007 = hbeta_cont * oiii_ew * one_plus_z * phot_i_clean * 1e-19 
    flux_nii6584 = halpha_cont * nii_ew * one_plus_z * phot_i_clean * 1e-19

    # Simple relative error propagation.
    def relative_error(val, err):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.abs(err / val)
        r[~np.isfinite(r)] = 0.0
        return r

    rel_halpha = np.sqrt(
        relative_error(halpha_cont, halpha_cont_err) ** 2 +
        relative_error(halpha_ew, halpha_ew_err) ** 2 +
        relative_error(one_plus_z, z_gal_err) ** 2
    )

    rel_hbeta = np.sqrt(
        relative_error(hbeta_cont, hbeta_cont_err) ** 2 +
        relative_error(hbeta_ew, hbeta_ew_err) ** 2 +
        relative_error(one_plus_z, z_gal_err) ** 2
    )

    rel_oiii = np.sqrt(
        relative_error(hbeta_cont, hbeta_cont_err) ** 2 +
        relative_error(oiii_ew, oiii_ew_err) ** 2 +
        relative_error(one_plus_z, z_gal_err) ** 2
    )

    rel_nii = np.sqrt(
        relative_error(halpha_cont, halpha_cont_err) ** 2 +
        relative_error(nii_ew, nii_ew_err) ** 2 +
        relative_error(one_plus_z, z_gal_err) ** 2
    )

    flux_halpha_err = np.abs(flux_halpha) * rel_halpha
    flux_hbeta_err = np.abs(flux_hbeta) * rel_hbeta
    flux_oiii5007_err = np.abs(flux_oiii5007) * rel_oiii
    flux_nii6584_err = np.abs(flux_nii6584) * rel_nii

    with np.errstate(divide='ignore', invalid='ignore'):
        ha_hb_ratio = flux_halpha / flux_hbeta
    ha_hb_ratio[~np.isfinite(ha_hb_ratio)] = np.nan

    lum_ha_corr = calculate_ha_luminosity_corrected(
        ha_flux_obs=flux_halpha,
        ha_hb_ratio=ha_hb_ratio,
        redshift=z_gal,
        cosmo_model=cosmo
    )

    log_sfr_ha = calculate_sfr(
        ha_flux_obs=flux_halpha,
        ha_hb_ratio=ha_hb_ratio,
        redshift=z_gal,
        cosmo_model=cosmo
    )

    log_sfr_ha_err = calculate_sfr_error(
        ha_flux_cont=halpha_cont,
        ha_ew=halpha_ew,
        hb_flux_cont=hbeta_cont,
        hb_ew=hbeta_ew,
        z=z_gal,
        ha_flux_cont_err=halpha_cont_err,
        ha_ew_err=halpha_ew_err,
        hb_flux_cont_err=hbeta_cont_err,
        hb_ew_err=hbeta_ew_err,
        z_err=z_gal_err,
        cosmo_model=cosmo
    )

    print("\nBuilding final catalogue...")


    derived_df = pd.DataFrame(index=df.index)

    # Store key physical parameters explicitly in the final catalogue.
    for col in EXTRA_PHYSICAL_PARAMS:
        derived_df[col] = preds_df[col]
        err_col = f"{col}_ERR"
        if err_col in errs_df.columns:
            derived_df[err_col] = errs_df[err_col]


    derived_df['CLASS'] = cls_label
    derived_df['P_GALAXY'] = cls_prob_df['P_GALAXY']
    derived_df['P_STAR'] = cls_prob_df['P_STAR']
    derived_df['P_QSO'] = cls_prob_df['P_QSO']

    derived_df['FLUX_HALPHA'] = flux_halpha
    derived_df['FLUX_HALPHA_ERR'] = flux_halpha_err

    derived_df['FLUX_HBETA'] = flux_hbeta
    derived_df['FLUX_HBETA_ERR'] = flux_hbeta_err

    derived_df['FLUX_OIII5007'] = flux_oiii5007
    derived_df['FLUX_OIII5007_ERR'] = flux_oiii5007_err

    derived_df['FLUX_NII6584'] = flux_nii6584
    derived_df['FLUX_NII6584_ERR'] = flux_nii6584_err

    derived_df['HA_HB_RATIO'] = ha_hb_ratio
    derived_df['L_HALPHA_CORR'] = lum_ha_corr
    derived_df['LOGSFR_HA'] = log_sfr_ha
    derived_df['LOGSFR_HA_ERR'] = log_sfr_ha_err


    # Keep all original CSV columns and append predictions and derived quantities.
    # Avoid duplicating the parameters that are already copied explicitly to derived_df.
    preds_df_to_save = preds_df.drop(columns=EXTRA_PHYSICAL_PARAMS, errors="ignore")
    errs_df_to_save = errs_df.copy()

    final_catalog = pd.concat(
        [df, preds_df_to_save, errs_df_to_save, cls_prob_df, derived_df],
        axis=1
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving catalogue to:\n{output_csv}")
    final_catalog.to_csv(output_csv, index=False)

    print("\nDone.")
    print(f"Processed objects: {len(final_catalog):,}")
    print(f"Saved file: {output_csv}")


if __name__ == "__main__":
    main()