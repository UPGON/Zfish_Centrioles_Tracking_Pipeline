### Create a virtual environment <a name="intro_conda"></a>
Omero requires zero-ice installation befpre it can be used. This package can be downloaded in this [website](https://github.com/glencoesoftware/zeroc-ice-py-win-x86_64/releases)

On the cluster:
```bash
conda create -y -n omero-env python=3.12
conda activate stardist-env
pip install zeroc_ice-3.6.5-cp312-cp312-win_amd64.whl
pip install omero-py
```