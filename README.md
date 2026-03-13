### Create a virtual environment <a name="intro_conda"></a>

On the cluster:
```bash
conda create -y -n omero-env python=3.12
conda activate stardist-env
pip install {ice_truc_location}
pip install omero-py
```