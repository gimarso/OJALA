# Descarga de los J-PAS Mocks (DESI DR1)

Los datos sintéticos (mocks) utilizados para entrenar y evaluar el modelo OJALA están alojados en la nube del Instituto de Astrofísica de Andalucía (IAA-CSIC).

Para descargar los archivos a tu máquina local, puedes utilizar cualquiera de los siguientes métodos.

---

## Opción 1: Descarga directa (Navegador)
Haz clic en el siguiente enlace público para descargar la carpeta completa en formato ZIP a través de tu navegador:

[https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download](https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download)

---

## Opción 2: Línea de comandos (Terminal)
Si estás trabajando en un servidor remoto o prefieres usar la terminal, puedes utilizar `wget` para descargar y extraer el archivo ZIP directamente. 

```bash
# Descarga el archivo
wget -O JPAS_mocks_DESI_DR1.zip "[https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download](https://cloud.iaa.es/index.php/s/8y42nN6XHHaSoKw/download)"

# Descomprime los datos en la carpeta actual
unzip JPAS_mocks_DESI_DR1.zip -d ./JPAS_mock_data/
