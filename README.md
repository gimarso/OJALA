# OJALA: Optimizing J-PAS Astronomy for Large-scale Analysis 🌌

![Esquema de OJALA](JPAS.png)

> A transformer-based autoregressive foundation model for the spectral energy distribution (SED) of galaxies, QSOs, and stars.

This repository contains the official implementation, pre-trained model weights, and training pipelines for **OJALA**, as presented in the paper *"OJALA: Optimizing J-PAS Astronomy for Large-scale Analysis"*. 

OJALA is designed to simultaneously classify astronomical objects and infer their physical parameters using 54-band narrow-band photometry from the J-PAS survey, combined with broad-band photometry from the DESI Legacy Imaging Surveys and WISE.

## 📂 Repository Structure

* `src/`: Core model architecture files, data loaders, and necessary utilities to build and load the transformer model.
* `model_OJALA/`: Directory containing the pre-trained weights of the fitted model.
* `data/`: Directory reserved for the synthetic mocks and training catalogs. 

## 🚀 Usage & Scripts

We provide several standalone scripts at the root level to manage different stages of the model's lifecycle:

* **`train.py`**: Trains the OJALA model from scratch using the datasets located in the `data/` folder.
* **`train_resume.py`**: Resumes the training process from a previously saved checkpoint.
* **`train_expand.py`**: Modifies the model's vocabulary, allowing you to easily add new physical variables or remove existing ones from the prediction targets.
* **`finetuned_OJALA.py`**: Fine-tunes the pre-trained embeddings for new, specialized downstream tasks (e.g., predicting Black Hole masses) without retraining the core model from scratch.

## 💾 Data Access

The Value Added Catalogs (VACs) for the J-PAS Early Data Release (EDR) will be available soon. 

The J-PAS synthetic mocks based on DESI DR1 are **available for download now**. Please follow the instructions provided in the `data/` directory to access and set them up.

## 📖 Citation

The paper presenting OJALA is currently **in preparation**. We will update this section with the official reference, citation format, and arXiv link as soon as it is publicly available.