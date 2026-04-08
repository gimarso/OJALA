# OJALA: Optimizing J-PAS Astronomy for Large-scale Analysis 

![Esquema de OJALA](JPAS.png)

> J-PAS multi-band image representation of galaxy observations

This repository contains the official implementation, trained model weights, and training pipelines for **OJALA**, a transformer-based autoregressive foundation model for the spectral energy distribution of galaxies, QSOs, and stars. More details are presented in the paper  *"OJALA: Optimizing J-PAS Astronomy for Large-scale Analysis"*. 

## 🧠 About the Model

OJALA operates analogously to Large Language Models but is tailored for astronomical data. Instead of predicting missing words, it predicts missing observations or physical properties by treating the SED and physical parameters as a flexible sequence of tokens. 

Key features of the architecture include:
* **Lightweight & Efficient:** With approximately 4.6 million parameters, OJALA is highly optimized, allowing for the rapid inference of millions of objects on standard consumer hardware.
* **Robust to Missing Data:** Thanks to its autoregressive attention mechanism, the model naturally handles incomplete input contexts without requiring imputation. It can perform robust inference even if certain photometric bands or physical labels are missing.
* **Highly Modular:** The pre-trained embeddings serve as powerful feature extractors. The model can be easily fine-tuned via lightweight regression heads to predict new physical properties (e.g., Black Hole masses) without retraining the core network.

## 📊 Training Data & Capabilities

OJALA is designed to process **54-band narrow-band photometry** from the J-PAS survey, combined with **broad-band photometry** from the DESI Legacy Imaging Surveys (g, r, z), WISE (W1, W2), and morphological parameters.

The foundation model was trained on a massive dataset of **~20 million synthetic SEDs** generated from high-quality **DESI DR1 spectra**. 

It is capable of simultaneously performing a wide range of tasks:
* **Spectral Classification:** Accurately separates Stars, Galaxies, and QSOs (achieving a weighted F1-score of ~0.9 for sources with i < 21).
* **Photometric Redshifts:** Delivers high-precision photo-z estimates for galaxies and high-redshift QSOs.
* **Galaxy Physical Properties:** Estimates stellar masses, Star Formation Rates (SFR), and equivalent widths (EW) for major optical emission lines (Hα, Hβ, [OIII], [NII]).
* **Stellar Parameters:** Infers fundamental atmospheric parameters such as effective temperature (T<sub>eff</sub>), surface gravity (log *g*), metallicity ([Fe/H]), and alpha-enhancement ([α/Fe]).

---

## 📂 Repository Structure

* `src/`: Core model architecture files, data loaders, and necessary utilities to build and load the transformer model.
* `model_OJALA/`: Directory containing the trained weights of the fitted model.
* `data/`: Directory reserved for the synthetic mocks and training catalogs. 

## 🚀 Usage & Scripts

We provide several standalone scripts at the root level to manage different stages of the model's lifecycle:

* **`train.py`**: Trains the OJALA model from scratch using the datasets located in the `data/` folder.
* **`train_resume.py`**: Resumes the training process from a previously saved checkpoint.
* **`train_expand.py`**: Modifies the model's vocabulary, allowing you to easily add new physical/observational variables or remove existing ones.
* **`finetuned_OJALA.py`**: Fine-tunes the pre-trained embeddings for new, specialized downstream tasks (e.g., predicting Black Hole masses) without retraining the core model from scratch.




## 📥 Data Access

The catalogues and mocks required to run OJALA are hosted on the cloud of the Instituto de Astrofísica de Andalucía (IAA-CSIC). The `data/README_data.md` file explains how to download:

* the **J-PAS input catalogue** used for inference,
* the **OJALA predicted catalogue** generated from the J-PAS EDR sample,
* and the **synthetic mocks** used for training and evaluation.

In particular, the catalogue used by the selection notebooks is:

* `JPAS_EDR_photometry_APER_COR_3_0_OJALA_catalog.csv`

For detailed download instructions and the suggested directory layout, please read:

* `data/README_data.md`

---

## 📒 Producing Science Catalogues with Jupyter Notebooks

OJALA provides three Jupyter notebooks to build science-ready subcatalogues for the main source classes:

* **`select_galaxies.ipynb`** → galaxy catalogue
* **`select_stars.ipynb`** → stellar catalogue
* **`select_QSOs.ipynb`** → QSO catalogue

These notebooks start from the global OJALA predicted catalogue and apply object-specific quality cuts, parameter-based selections, visualization steps, and final export of the selected sample. The predicted catalogue is the same one described in `data/README_data.md`. Each notebook writes a final CSV file containing the selected sample. Typical outputs are:

```text
galaxy_catalog.csv
star_catalog.csv
qso_catalog.csv
```

These files are intended to provide clean science-ready samples for downstream analysis.

## 🗂️ Output Catalogue: Columns and Physical Meaning

The final OJALA catalogue keeps **all columns from the original input CSV** and appends the model predictions, their uncertainties, class probabilities, and a set of physically derived quantities.

In practice, the output catalogue contains three types of information:

### 1. Original input columns
These are preserved exactly as they appear in the input catalogue, including:
- the original J-PAS photometry used as model input (e.g. `APER_COR_3_0_J0378` ... `APER_COR_3_0_J0910`, depending on the selected photometry prefix),
- broad-band magnitudes (`MAG_i`, `MAG_G`, `MAG_R`, `MAG_Z`, `MAG_W1`, `MAG_W2`),
- the reference `iSDSS` photometric column used to rescale line fluxes,
- and any other metadata already present in the input table.

This means the OJALA output is a **value-added catalogue (VAC)** built on top of the original J-PAS photometric table.

---

### 2. Direct model outputs
These columns are predicted directly by the OJALA network.

#### 2.1 Main spectral classification
- `CLASS`  
  Final spectral class assigned by the model. Possible values are:
  - `GALAXY`
  - `STAR`
  - `QSO`

- `P_GALAXY`, `P_STAR`, `P_QSO`  
  Predicted class probabilities from the classification head.  
  `CLASS` is obtained from the maximum of these probabilities.

#### 2.2 Direct regression outputs
OJALA predicts the following physical quantities directly:

- `Z_GAL`  
  Photometric redshift optimized for galaxies.

- `Z_QSO`  
  Photometric redshift optimized for QSOs.

- `HALPHA_CONT`  
  Predicted continuum level around Hα.

- `HBETA_CONT`  
  Predicted continuum level around Hβ.

- `HALPHA_6562_EW`  
  Rest-frame equivalent width of Hα.

- `HBETA_4861_EW`  
  Rest-frame equivalent width of Hβ.

- `OIII_5007_EW`  
  Rest-frame equivalent width of [OIII] λ5007.

- `NII_6584_EW`  
  Rest-frame equivalent width of [NII] λ6584.

- `LOGM`  
  Predicted stellar mass, in log10 solar units.

- `LOGSFR`  
  Predicted star formation rate, in log10(M⊙ yr⁻¹).

- `TEFF`  
  Stellar effective temperature.

- `LOGG`  
  Stellar surface gravity.

- `ALPHAFE`  
  Alpha-element enhancement, [α/Fe].

- `FEH`  
  Stellar metallicity, [Fe/H].

For every direct regression output above, the catalogue also includes a corresponding uncertainty column:
- `<NAME>_ERR`

For example:
- `Z_GAL_ERR`
- `LOGM_ERR`
- `HALPHA_6562_EW_ERR`
- `TEFF_ERR`

These uncertainties are the heteroscedastic predictive uncertainties returned by the model.

> **Important note:** not all parameters are physically meaningful for all source classes.  
> For example, stellar parameters (`TEFF`, `LOGG`, `FEH`, `ALPHAFE`) are intended for sources classified as stars, while nebular and galaxy parameters are mainly intended for galaxy-like sources.

---

### 3. Post-processed / derived quantities
These columns are **not predicted directly** by the network.  
Instead, they are computed from the direct model outputs plus the input photometry.

#### 3.1 Emission-line fluxes
The observed emission-line fluxes are reconstructed from the predicted continuum, predicted equivalent width, the galaxy redshift term `(1 + z)`, and the input `iSDSS` normalization:

- `FLUX_HALPHA`
- `FLUX_HBETA`
- `FLUX_OIII5007`
- `FLUX_NII6584`

with corresponding propagated uncertainties:

- `FLUX_HALPHA_ERR`
- `FLUX_HBETA_ERR`
- `FLUX_OIII5007_ERR`
- `FLUX_NII6584_ERR`

In the current implementation, the fluxes are computed as:

- `FLUX_HALPHA ∝ HALPHA_CONT × HALPHA_6562_EW × (1 + Z_GAL) × iSDSS`
- `FLUX_HBETA ∝ HBETA_CONT × HBETA_4861_EW × (1 + Z_GAL) × iSDSS`
- `FLUX_OIII5007 ∝ HBETA_CONT × OIII_5007_EW × (1 + Z_GAL) × iSDSS`
- `FLUX_NII6584 ∝ HALPHA_CONT × NII_6584_EW × (1 + Z_GAL) × iSDSS`

A constant scale factor is applied in the script to place the fluxes in physical units.

#### 3.2 Balmer decrement
- `HA_HB_RATIO`  
  Ratio between the reconstructed Hα and Hβ fluxes:
  `FLUX_HALPHA / FLUX_HBETA`

This is used as a proxy for nebular attenuation.

#### 3.3 Dust-corrected Hα luminosity
- `L_HALPHA_CORR`  
  Extinction-corrected Hα luminosity derived from:
  - the reconstructed Hα flux,
  - the Hα/Hβ Balmer decrement,
  - the galaxy redshift through the luminosity distance.

The dust correction assumes:
- an intrinsic Case B ratio `Hα/Hβ = 2.86`,
- and a Calzetti attenuation law.

This quantity is therefore a **derived post-processing product**, not a native network output.

#### 3.4 Hα-based star formation rate
- `LOGSFR_HA`  
  Star formation rate derived from the dust-corrected Hα luminosity using the Kennicutt (1998) conversion:
  `SFR = 7.9 × 10^-42 × L_Hα`

- `LOGSFR_HA_ERR`  
  Propagated uncertainty on `LOGSFR_HA`, based on the uncertainties in:
  - Hα continuum,
  - Hβ continuum,
  - Hα EW,
  - Hβ EW,
  - and `Z_GAL`.

As with `L_HALPHA_CORR`, this quantity is produced in post-processing and should be interpreted as a secondary derived estimator.

---

## ✅ Summary: direct model outputs vs post-processing

### Direct OJALA outputs
- `CLASS`
- `P_GALAXY`, `P_STAR`, `P_QSO`
- `Z_GAL`, `Z_QSO`
- `HALPHA_CONT`, `HBETA_CONT`
- `HALPHA_6562_EW`, `HBETA_4861_EW`, `OIII_5007_EW`, `NII_6584_EW`
- `LOGM`, `LOGSFR`
- `TEFF`, `LOGG`, `ALPHAFE`, `FEH`
- all corresponding `*_ERR` columns

### Post-processed derived quantities
- `FLUX_HALPHA`, `FLUX_HBETA`, `FLUX_OIII5007`, `FLUX_NII6584`
- `FLUX_HALPHA_ERR`, `FLUX_HBETA_ERR`, `FLUX_OIII5007_ERR`, `FLUX_NII6584_ERR`
- `HA_HB_RATIO`
- `L_HALPHA_CORR`
- `LOGSFR_HA`
- `LOGSFR_HA_ERR`

---

## ⚠️ Recommended use
Users should interpret the catalogue in a class-aware way:
- use `TEFF`, `LOGG`, `FEH`, and `ALPHAFE` primarily for objects with high `P_STAR`,
- use `Z_GAL`, nebular EWs, line fluxes, `L_HALPHA_CORR`, `LOGM`, and `LOGSFR` primarily for objects with high `P_GALAXY`,
- use `Z_QSO` primarily for objects with high `P_QSO`.

For ambiguous objects, the class probabilities should be inspected before using class-specific physical parameters.



## 📖 Citation

If you use OJALA, the trained models, or the generated catalogues in your research, please cite the official paper:

**[OJALÁ: Optimizing J-PAS Astronomy for Large-scale Analysis. A foundation model for the SED of galaxies, QSOs and stars](https://arxiv.org/abs/2604.00661)**

```bibtex
@ARTICLE{2026arXiv260400661M,
       author = {{Mart{\'\i}nez-Solaeche}, G. and {Gonz{\'a}lez Delgado}, R.~M. and {Garc{\'\i}a-Benito}, R. and {Hern{\'a}n-Caballero}, A. and {P{\'e}rez-R{\`a}fols}, I. and {D{\'\i}az-Garc{\'\i}a}, L.~A. and {Abramo}, L. Raul and {Rodr{\'\i}guez-Mart{\'\i}n}, J.~E. and {Conrado}, A.~M. and {Breda}, I. and {Dom{\'\i}nguez S{\'a}nchez}, H. and {M{\'a}rquez}, I. and {Pieri}, M. and {L{\'o}pez-Cano}, D. and {Placco}, V.~M. and {Nakazono}, L. and {del Pino}, A. and {Marra}, V. and {Alcaniz}, J. and {Benitez}, N. and {Bonoli}, S. and {Carneiro}, S. and {Cenarro}, A.~J. and {Crist{\'o}bal-Hornillos}, D. and {Daflon}, S. and {Dupke}, R.~A. and {Ederoclite}, A. and {Hern{\'a}ndez-Monteagudo}, C. and {Liu}, J. and {L{\'o}pez-Sanjuan}, C. and {Mar{\'\i}n-Franch}, A. and {Mendes de Oliveira}, C. and {Moles}, M. and {Roig}, F. and {Sodr{\'e}}, L. and {Taylor}, K. and {Varela}, J. and {V{\'a}zquez Rami{\'o}}, H. and {V{\'\i}lchez}, J.~M. and {Zaragoza-Cardiel}, J.},
        title = "{OJAL{\'A}: Optimizing J-PAS Astronomy for Large-scale Analysis. A foundation model for the SED of galaxies, QSOs and stars}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies, Instrumentation and Methods for Astrophysics},
         year = 2026,
        month = apr,
          eid = {arXiv:2604.00661},
        pages = {arXiv:2604.00661},
          doi = {10.48550/arXiv.2604.00661},
archivePrefix = {arXiv},
       eprint = {2604.00661},
 primaryClass = {astro-ph.GA},
       adsurl = {[https://ui.adsabs.harvard.edu/abs/2026arXiv260400661M](https://ui.adsabs.harvard.edu/abs/2026arXiv260400661M)},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

