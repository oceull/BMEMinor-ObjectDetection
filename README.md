# Installation

### Dataset
    M2CAI16-tool dataset (http://camma.u-strasbg.fr/datasets), download and add to <local_dir>/mnt/data/m2cai
        Linux machine: drop in mnt/data/m2cai
### Dependencies
    Prerequisites: - Python ( > 3.9.10, 64-bit recommended)
                   - pip (https://www.geeksforgeeks.org/how-to-install-pip-on-windows/, https://www.tecmint.com/install-pip-in-linux/)
    Open terminal in <local_dir>
        python3 -m venv venv/  -> Creates virtual environment venv
        Linux/MacOS(?): source venv/bin/activate
        Windows: ./.venv/Scripts/Activate
        pip install Requirements.txt -> install project dependencies
        pip install torch===1.10.1+cu113 torchvision===0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html -> torch cuda for GPU optimization

# Configuration
    A file with global variables for the project wasn't made.
    By default, all files used at Runtime are located in curr_dir + mnt/data/m2cai.

    For the following files, make sure to update curr_dir variable (absolute path of project directory)
    get_dense_feature.py                      preprocess/video_preprocess.py
    generate_video_features.py                          /data_list_generate.py
    python train_video_graph_cls.py
    main.py 
    
    Furthermore, these files contain optimization flags: gpu_num, batch_size (batch_size 1,2,4,8,16,32,64 adjust depending on GPU available memory)
    Other optimization are present, scan through each file ran via terminal for more details.

    Get_dense_features and train_video_graph_cls are configured to save in a pkl format the trained weights of the CNNs, for continuous training, it can be modified to load an existing train model and train it only with the new data.

    Runtime Error: torch CUDA not located -> make sure torch, GPU version was installed.
    Runtime Error: GPU out of memory -> reduce batch size.
    Runtime [OS] Error: Too many opened files -> preprocess/data_list_generate -> reduce max_num_to_dump

# Training

## Video preprocess
    -- activate virtual env --
    -- cd main --
    python preprocess/video_preprocess.py
    python preprocess/data_list_generate.py

## Feature extraction
    -- activate virtual env --
    -- cd main --
    python get_dense_feature.py  / Train DenseNet121, this step can be skipped if you want to use Pretrained version (lower accuracy but no training time)
    python generate_video_features.py
Possible Errors: page fault, increase maximum virtual paging from os settings to around 100 GB for M2CAI16-Tool

## Training CNN
    Graph Convolution Net is trained using features extracted by Densenet121
    -- activate virtual env --
    -- cd main --
    python train_video_graph_cls.py

# Real-time
    -- activate virtual env --
    -- cd main --
    python main.py 
    Camera is setted for local camera (cv2.VideoCapture(0) ), it can be configured for url, saved videos, etc.