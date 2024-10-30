| :zap:        This is xander's workspace. Please follow this document to run! The `rslsl` folder is implemented for now.|
| ------------------------------------------------------------------------------------------------ |

## Installation ##
1. Create a new Python virtual env or conda environment with Python 3.6, 3.7, or 3.8 (3.8 recommended)
    ```
    conda create -n mqe python=3.8
    ```
2. Install PyTorch and Isaac Gym.
    - Install appropriate PyTorch version from https://pytorch.org/. [The latest version is all right in fact]
        ```
        pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
        ```
    - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym. Ubuntu20.04 and Python3.8 recommended.
        ```
        tar -xf IsaacGym_Preview_4_Package.tar.gz
        cd isaacgym/python && pip install -e .
        ```
3. Check Isaac Gym is available by running
    - `cd examples && python 1080_balls_of_solitude.py`
4. Install MQE. Move to the directory of this repository and run
    - `pip install -e .`
5. Check MQE is available by running
    - `python ./test.py`
6. Move to `xander_ws/rslrl/rsl_rl` and run
    - `pip install -e .`
   
   to install rsl_rl. 


## Usage ##
1. Try different tasks

    - `python ./test.py`

    - Task could be specified in `./test.py`

2. Run PPO 

    - `cd xander_ws/rslrl && python train.py`

3. Play your trained model
   
   - `cd xander_ws/rslrl && python play.py`
