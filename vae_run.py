import os

#os.system('CUDA_VISIBLE_DEVICES="0" python3 vae_main.py -mt vae -ol mRNA@ -hn 4096 -lr 0.001 -dr 0 -wd 1e-5 -dv cuda -cd 0 -sn vae_pretrained -fc None -mx 500 -af Tanh -mo Adam -bs 0 -vd ember_libfm_200115 -sm')
os.system('CUDA_VISIBLE_DEVICES="0" python3 vae_main.py -mt vae -ol mRNA@ -hn 4096 -lr 0.001 -dr 0 -wd 1e-5 -dv cuda -cd 0 -sn vae_pretrained -fc None -mx 500 -af Tanh -mo Adam -bs 0 -vd toyforVAE -sm')
