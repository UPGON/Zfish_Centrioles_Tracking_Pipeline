# 1. Setup and installation
This document describes all the steps required for setting up the project and running its scripts. All required installations will also be discussed in this document.
This procedure can be performed on local computers or on an HPC cluster.

## 1.1 Clone the repository
To run the project scripts locally, you must first clone the repository to your own computer. <br>
Cloning the repository can be done by following these instructions: 
1. Open the folder manager on your computer 
2. Navigate to the location where you want to clone the repository
3. Open a terminal <br>
        **On Windows**: Right click inside the folder and select `Open in Terminal` <br>
        **On Mac**: Press `Command + Option + P`
4. Run the command `git clone https://github.com/UPGON/Zfish_Centrioles_Tracking_Pipeline.git` <br>
    Verify that Git is correctly installed on your computer. If it is not the case, follow the instructions on this [website](https://git-scm.com/install/windows)
5. Verify that the folders from the project are now present on your computer 


## 1.2 Install Anaconda
To create and manage Python environments, we recommend installing Miniconda or Anaconda. <br>
Please follow the instructions corresponding to your OS on this [website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

## 1.3 Create the Python environment
To run the project with all the associated packages, a Python environment needs to be created and activated every time one wants to run scripts. <br>
To do so, open the Miniconda/Anaconda terminal and run the following commands:

```bash
conda create -n zf_pipeline-env python=3.10
conda activate zf_pipeline-env 
```
After the environment has been created and activated, navigate in the Anaconda terminal to the project root location (under ZFISH_CENTRIOLES_TRACKING_PIPELINE) and run the command:
```bash
pip install -r .\requirements.txt
```

## 1.4 Connect to Jed cluster 
To connect to a cluster, follow the instructions in the [Scitas documentation](https://scitas-doc.epfl.ch/user-guide/using-clusters/connecting-to-the-clusters/). The connection can be established by running the following command in the Windows command prompt terminal:
```bash
ssh voland@jed.hpc.epfl.ch
```
If you have trouble connecting, you probably do not have access to Jed. You can verify this by checking if you have something related to Jed or HPC on this [website](https://groups.epfl.ch/#/home/member-groups?query=&status=all&owner=&member=325144&admin=&pageindex=0&pagesize=10&sortcolumn=name&sortdirection=asc&mode=member). As a master's student, you should have something like: hpc-masters - Master's students having access to SCITAS managed HPC resources for project work.

To handle files and folders stored in the session, the easiest way is to use WinSCP.
To connect, enter voland@jed.hpc.epfl.ch as the address.