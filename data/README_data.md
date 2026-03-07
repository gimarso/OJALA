# Data Availability

The data products used by **OJALA** are hosted on the cloud of the Instituto de Astrofísica de Andalucía (IAA-CSIC).

This directory contains instructions to download both the input J-PAS catalogue used for inference and the OJALA catalogue with model predictions.

---

## 1. J-PAS input catalogue

This is the input catalogue used to run OJALA inference on J-PAS EDR photometry.

**File:**
- `JPAS_EDR_photometry.csv`

### Direct download
Use the following public link to download the file:

```text
https://cloud.iaa.es/index.php/s/Qg9ZL6BGkwrsaP3/download?path=%2F&files=JPAS_EDR_photometry.csv
```

### Command line
```bash
wget -O JPAS_EDR_photometry.csv "https://cloud.iaa.es/index.php/s/Qg9ZL6BGkwrsaP3/download?path=%2F&files=JPAS_EDR_photometry.csv"
```

---

## 2. OJALA predicted catalogue

This is the OJALA catalogue generated from the J-PAS EDR input sample using the `APER_COR_3_0` photometry configuration.

**File:**
- `JPAS_EDR_photometry_APER_COR_3_0_OJALA_catalog.csv`

### Direct download
Use the following public link to download the file:

```text
https://cloud.iaa.es/index.php/s/Qg9ZL6BGkwrsaP3/download?path=%2F&files=JPAS_EDR_photometry_APER_COR_3_0_OJALA_catalog.csv
```

### Command line
```bash
wget -O JPAS_EDR_photometry_APER_COR_3_0_OJALA_catalog.csv "https://cloud.iaa.es/index.php/s/Qg9ZL6BGkwrsaP3/download?path=%2F&files=JPAS_EDR_photometry_APER_COR_3_0_OJALA_catalog.csv"
```

---

## 3. Download the full folder

If you prefer, you can also access the full public folder in the IAA cloud:

```text
https://cloud.iaa.es/index.php/s/Qg9ZL6BGkwrsaP3
```

From there, you can download individual files or the complete folder through your browser.

---

## 4. J-PAS mocks used for training and evaluation

The synthetic data (mocks) used to train and evaluate the OJALA model are also hosted on the IAA-CSIC cloud.

### Direct download
```text
https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download
```

### Command line
```bash
wget -O JPAS_mocks_DESI_DR1.zip "https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download"
unzip JPAS_mocks_DESI_DR1.zip -d ./JPAS_mock_data/
```

---

## 5. Suggested directory structure

After downloading the files, a convenient structure is:

```text
data/
├── README_data.md
├── catalogues/
│   ├── JPAS_EDR_photometry.csv
│   └── JPAS_EDR_photometry_APER_COR_3_0_OJALA_catalog.csv
└── mocks/
    └── JPAS_mock_data/
```

This layout is compatible with the catalogue production and validation scripts included in the repository.
