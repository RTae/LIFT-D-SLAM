# LIFT-D-SLAM (Name will come later)

## Introduction

## Install package

1. Install python environment control such as conda, pipenv and etc.
```
conda create --name WORKSPACE python=3.8
conda activate WORKSAPCE
```

2. Install python depency
```
pip install -r requirements.txt
```

3. Install pre-build package which is show below (I build in arm64, so if you use another system please built it again by yourself)
 - [g2oPy](https://github.com/uoip/g2opy)
 - [Pangolin](https://github.com/uoip/pangolin)

 3.1 You need to copy the file by using command below
 ```
 cp deployment/packages/build_lib/* [path-to-python-lib]/lib/python{python-version-which-you-use}/site-package
 ```

## How to use it


## Ref