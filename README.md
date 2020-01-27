G StereoGNN

Understands molecules in terms of stereoisomers.
Prediction targets
- solubility
- protein bioactivity
- ...

## Introduction

Stereoisomer is one of the important factor causing fails in drug development.
However, most of current deep learning models understanding chemical compounds does rarely address the issue.
Even He doesnt neither....

## Demo

### Prerequisites
Gene expression data of TCGA is required.
The data can be downloaded in ICGC data portal.
https://dcc.icgc.org/

### Training Variational Autoencoder
Run :
```
python vae_run.py
```


### Training and evaluating VAECox
Run :
```
python main.py
```

## Authors

* **Sunkyu Kim**  
* **Keonwoo Kim** 
* **Junseok Choi**
* **Inggeol Lee** 
* **Jaewoo Kang** - *corresponding author ... * 


## References
