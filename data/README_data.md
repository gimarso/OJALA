# Downloading the J-PAS Mocks (DESI DR1)

The synthetic data (mocks) used to train and evaluate the OJALA model are hosted on the cloud of the Instituto de Astrofísica de Andalucía (IAA-CSIC).

To download the files to your local machine, you can use any of the following methods.

---

## Option 1: Direct Download (Browser)
Click on the following public link to download the complete folder as a ZIP file through your web browser:

[https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download](https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download)

---

## Option 2: Command Line (Terminal)
If you are working on a remote server or prefer using the terminal, you can use `wget` to download and extract the ZIP file directly. 

```bash
# Download the file
wget -O JPAS_mocks_DESI_DR1.zip "[https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download](https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download)"

# Extract the data into the current directory
unzip JPAS_mocks_DESI_DR1.zip -d ./JPAS_mock_data/