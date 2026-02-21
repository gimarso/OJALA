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

## 💾 Data Access

The Value Added Catalogs (VACs) for the J-PAS Early Data Release (EDR) will be available soon. 

The J-PAS synthetic mocks based on DESI DR1 are **available for download now**. 
* **[Direct Download Link (ZIP)](https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download)**
* For advanced download methods (Terminal/Wget or Python script), please read the instructions in the `data/README_data.md` file.

## 📖 Citation

The paper presenting OJALA is currently **in preparation**. We will update this section with the official reference, citation format, and arXiv link as soon as it is publicly available.