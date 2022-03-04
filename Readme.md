# Basics of Deep Learning in Computer Vision

## Table of Contents

- [Basics of Deep Learning in Computer Vision](#basics-of-deep-learning-in-computer-vision)
  - [Table of Contents](#table-of-contents)
  - [1. Set up a local development environment](#1-set-up-a-local-development-environment)
    - [1.1. Prerequisites](#11-prerequisites)
      - [1.1.1. Windows](#111-windows)
      - [1.1.2. MacOS](#112-macos)
    - [1.2. Set up the local environment](#12-set-up-the-local-environment)
  - [2. Set up a remote development environment](#2-set-up-a-remote-development-environment)
    - [2.1. First time](#21-first-time)
    - [2.2 Afterwards (login to the remote project)](#22-afterwards-login-to-the-remote-project)
  - [3. Set up VSCode](#3-set-up-vscode)
    - [3.1. Open the workspace](#31-open-the-workspace)
    - [3.2. Test local Python](#32-test-local-python)
    - [3.3. Set up remote Jupyter](#33-set-up-remote-jupyter)
    - [3.4. Test remote Jupyter](#34-test-remote-jupyter)
  - [4. Update project files from Github](#4-update-project-files-from-github)
    - [4.1. Overwrite old files and add new files](#41-overwrite-old-files-and-add-new-files)
  - [5. MLflow](#5-mlflow)
    - [5.1. Launch MLflow UI](#51-launch-mlflow-ui)
    - [5.2. Tracking URI](#52-tracking-uri)
    - [5.3. Run experiments](#53-run-experiments)
  - [6. Tensorboard](#6-tensorboard)
  - [7. Load trained weights](#7-load-trained-weights)
    - [7.1 `pl.LightningModule`](#71-pllightningmodule)
    - [7.2. from `pl.LightningModule` to `torch.nn.Module`](#72-from-pllightningmodule-to-torchnnmodule)

## 1. Set up a local development environment

### 1.1. Prerequisites

#### 1.1.1. Windows

1. Install [VSCode](https://code.visualstudio.com/download)

2. Install VSCode extensions from PowerShell

    ```bash
    code --install-extension ms-python.python
    code --install-extension ms-python.vscode-pylance
    code --install-extension ms-toolsai.jupyter
    code --install-extension ms-toolsai.jupyter-renderers
    code --install-extension ms-vscode-remote.vscode-remote-extensionpack
    ```

3. Install [WSL(Ubuntu)](https://docs.microsoft.com/en-us/windows/wsl/setup/environment) from an adminstrator PowerShell

    ```bash
    wsl --install
    ```

    You need to reboot your computer during the installation, and then set a username and password for WSL.

4. Install packages in Ubuntu

    Right click mouse to paste contents in WSL.

    ```bash
    sudo apt-get update
    sudo apt-get install wget ca-certificates curl
    ```

    You would be asked for a passowrd.

5. Install pyenv in Ubuntu

    ```bash
    curl https://pyenv.run | bash
    ```

6. Add environment variables

    ```bash
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    exec $SHELL
    ```

#### 1.1.2. MacOS

1. Lauch Terminal
2. Install [Homebrew](https://brew.sh/)

    ```bash
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    ```

3. Install packages

    ```bash
    brew install wget git pyenv pyenv-virtualenv
    ```

4. Add environment variables

    ```bash
    echo 'eval "$(pyenv init --path)"' >> ~/.zprofile
    echo 'eval "$(pyenv init -)"' >> ~/.zprofile
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zprofile
    exec $SHELL
    ```

    If your MacOS < 10.15, replace `.zprofile` with `.bash_profile`.

5. Install VSCode and extensions.

    ```bash
    brew install --cask visual-studio-code
    code --install-extension ms-python.python
    code --install-extension ms-python.vscode-pylance
    code --install-extension ms-toolsai.jupyter
    code --install-extension ms-toolsai.jupyter-renderers
    code --install-extension ms-vscode-remote.vscode-remote-extensionpack
    ```

### 1.2. Set up the local environment

1. Create the project folder

    ```bash
    mkdir -p ~/Projects
    cd ~/Projects
    ```

2. Clone the lecture project

    ```bash
    git clone https://github.com/hinson/DLCVLecture
    cd DLCVLecture
    ```

3. Create the python environment

    ```bash
    pyenv install mambaforge
    pyenv local mambaforge
    mamba env create -f env-cpu.yml
    ```

4. Activate the python environment (named `dlcvl`)

    ```bash
    conda activate dlcvl
    ```

    If you want to deactivate, use

    ```bash
    conda deactivate
    ```

5. Open the project with VSCode

    ```bash
    code DLCVLecture.code-workspace
    ```

    Please trust the author😊

## 2. Set up a remote development environment

### 2.1. First time

1. Create your RSA key pair from an adminstrator PowerShell in Windows or Terminal in MacOS

    ```bash
    ssh-keygen
    ```

    Press `enter` if you are prompted.

2. Copy your public key

    ```bash
    cat ~/.ssh/id_rsa.pub     # copy the entire output
    ```

3. Login to a remote GPU server

    ```bash
    ssh {username}@{server IP}     # replace the placeholders
    ```

    Input your password. Input `yes` if the machine ask for your confirmation.

    If you want to change the password, use

    ```bash
    passwd
    ```

4. Paste the public key to the remote server

    ```bash
    mkdir -p ~/.ssh && vim ~/.ssh/authorized_keys
    ```

    Paste the copied content to the end of the file and save (press `shift` + `g`, press `o`, paste, press `esc`, input `:wq`).

5. Login the remote server without password

    ```bash
    exit
    ssh {username}@{server IP}    #
    ```

    If the server does not ask for a password, then all is well.

6. Set up the project

    ```bash
    curl https://pyenv.run | bash
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    exec $SHELL
    ```

    ```bash
    mkdir -p ~/Projects
    cd ~/Projects     #
    ```

    ```bash
    git clone https://github.com/hinson/DLCVLecture
    cd DLCVLecture    #
    ```

7. Setup

    ```bash
    pyenv install mambaforge
    pyenv local mambaforge
    mamba env create -f env-gpu.yml
    conda activate dlcvl      #
    ```

8. Launch a Jupyter server

    ```bash
    jupyter notebook --no-browser --ip={server IP} --port={your port}     #
    ```

    The output will be like this:

    ```bash
    ...
    http://{server IP}:{your port}/?token={your token}
    ...
    ```

    Open this url in your browser.
    `{your token}` will change every time you lanch jupyter.

### 2.2 Afterwards (login to the remote project)

The above commands with `#` are required every time you re-login to the remote server. Summarize as

```bash
ssh {username}@{server IP} 
cd ~/Projects/DLCVLecture
conda activate dlcvl
```

## 3. Set up VSCode

### 3.1. Open the workspace

1. From menu,  `File` -> `Open Workspace from File`, select `{your home}/Projects/DLCVLecture/DLCVLecture.code-workspace`
2. Check if the project files are shown in `VSCode Explorer` on the left.

### 3.2. Test local Python

1. From `VSCode Explorer`, open `DLCVLecture/python/test.py`
2. click `Python {version}` in the lower left corner
3. Select `Select at workspace level`
4. Select the version with `(dlcvl)`
5. Prest `ctrl`+`F5` to run `test.py`.

### 3.3. Set up remote Jupyter

1. Launch VSCode
2. Press `F1`, input `jusplr`, select `Jupyter: Specify local or remote Jupyter server for connections`
3. Select `Existing`
4. Paste the above jupyter url and press `enter`

See [Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) for more details.

### 3.4. Test remote Jupyter

1. From `VSCode Explorer`, open `DLCVLecture/notebooks/test.ipynb`
2. Click `Select Kernel` in the upper right corner
3. Select the one with `/home/{username}/...(Remote)...`
4. Click the right Play button or use `shift` + `enter` to execute the code in the first cell.

If you want to close remote Jupyter, press `ctrl` + `c` in the remote session.

## 4. Update project files from Github

### 4.1. Overwrite old files and add new files

In both your `local` and `remote` project folders, run the following command to update the materials.

```bash
cd ~/Projects/DLCVLecture
git fetch && git reset --hard origin/master
```

If you want to overwrite a specific file:

```bash
cd ~/Projects/DLCVLecture
git fetch && git checkout origin/master {relative path} #  eg. git checkout origin/master Readme.md
```

## 5. MLflow

### 5.1. Launch MLflow UI

Follow [2.2](#22-afterwards-login-to-the-remote-project) to login to the remote server, then

```bash
cd python/example
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port {your port} 
```

There will be a new `sqlite` db file `mlflow.db` to store experimental data in `~/Projects/DLCVLecture/python/example`. If you want to use other databases, see [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html).

Then visit `{server IP}:{your port}` in a browser.

### 5.2. Tracking URI

Open a new terminal and repeat [2.2](#22-afterwards-login-to-the-remote-project), then

```bash
export MLFLOW_TRACKING_URI=http://{server IP}:{your port}
```

If you do not export the above evironment variable, the following `mlflow` commands will use the default tracking URI `http://localhost:5000`

### 5.3. Run experiments

```bash
cd python/example
```

- To run the default entry point `main`:

```bash
mlflow run . --no-conda --experiment-name MNIST   
```

The `--no-conda` flag is to avoid creating `dlcvl` environment repeatedly.

According to the `MLproject` file, we can set `max_epochs` and `batch_size` like:

```bash
mlflow run . --no-conda --experiment-name MNIST -P max_epochs=5 -P batch_size=128  
```

- To run the `grid search` entry point:

```bash
mlflow run . --no-conda --experiment-name MNIST_gs -e grid
```

## 6. Tensorboard

```bash
tensorboard --host 0.0.0.0 --port {your port} --logdir {lightning log path}
```

Then visit `{server IP}:{your port}` in a browser.

## 7. Load trained weights

### 7.1 `pl.LightningModule`

```python
model = MNISTClassifier.load_from_checkpoint({some pl. checkpoint path})
```

See [Saveing and Loading Weights (Lightning)](https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html) for details.

### 7.2. from `pl.LightningModule` to `torch.nn.Module`

Because `pl.LightningModule` is a subclass of `torch.nn.Module`, you can also load a state dict saved by `pl.LightningModule` to a `torch.nn.Module` model when they _have the same layer definitions_.

For example, you can use the `CNN` class defined in `notebook/process` 3.1:

```python
model = CNN()
ckpt = torch.load({some pl. checkpoint path})
model.load_state_dict(ckpt["state_dict"])
```

More about `state dict`, see [Saving and Loading Models (PyTorch)](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
