# OJALA Dual License Agreement

Copyright (c) 2026 G. Martínez-Solaeche, R. M. González Delgado, et al. (The J-PAS Collaboration)

This repository and its associated contents (software, model weights, and data catalogs) are distributed under a dual-licensing model to accommodate the different nature of the code and the scientific data. By using, downloading, or adapting any part of this repository, you agree to the following terms:

---

## 1. Software License (Source Code)

All source code contained in this repository, including but not limited to the `src/` directory, training pipelines (`train.py`, `train_resume.py`, `train_expand.py`, `finetuned_OJALA.py`), and Jupyter notebooks, is distributed under the **MIT License**.

**MIT License**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 2. Data and Model Weights License

All trained model weights (located in the `model_OJALA/` directory), synthetic mock catalogs, predicted catalogs, and any associated datasets hosted by the Instituto de Astrofísica de Andalucía (IAA-CSIC) are distributed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

**Summary of CC BY 4.0 Terms:**
* **Share:** You are free to copy and redistribute the material in any medium or format.
* **Adapt:** You are free to remix, transform, and build upon the material for any purpose, even commercially.

**Attribution Requirement (Credit):**
You must give appropriate credit, provide a link to the license, and indicate if changes were made. If you use the model weights, data catalogs, or software in your scientific research or any derived products, **you must cite the following foundational paper**:

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
