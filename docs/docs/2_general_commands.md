# General commands

## 2.1 Running scripts
Running project scripts must be done from the Anaconda terminal, with the proper environment activated.
The general instructions to run any script are described below; for running specific scripts, please refer to the corresponding documentation:
1. Open the folder manager on your computer
2. Navigate in the folder manager to the folder where you want to save the script output
3. Create a `.bat` file. This file will contain all the command lines required to run your script (e.g. `detection_cs.bat`, `coloc_ana.bat`)
4. Open the `.bat` file and write the command required for your script. The command should have the following global structure:
```bash
@echo off
python path_to_the_script.py script_parameter 
```
For the script parameters, please refer to the corresponding documentation.
5. Open the Anaconda terminal
6. Activate your conda environment with the command:
```bash
conda activate zf_pipeline-env
```
Please adapt the conda environment name if you want to use another environment.
7. Navigate in the Anaconda terminal to the location where the `.bat` script is located
8. Run your script by calling your `.bat` command file
```bash
.\my_file_name.bat
```