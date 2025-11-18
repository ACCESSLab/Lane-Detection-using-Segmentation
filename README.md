## Realtime Semantic Segmentation-Based Lane Detection

### Demo Videos
  :new: Testing TensorRT model on Jetson Orin Nano Super and RTX A4000 GPU
   - <a href="https://ncaandt-my.sharepoint.com/:v:/g/personal/tagetahun_ncat_edu/EUKvM8wB2qhMiThRu1SZiaABa-REYMeviw1XAnWUbKjelw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=IJktTU"> Testing on Jetson Orin Nano Super </a>
   - <a href="https://ncaandt-my.sharepoint.com/:v:/g/personal/tagetahun_ncat_edu/EZZbYlWpZWxLrrvOgNei5NMBebqeyQ4I-U8BkohgicXAtQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=8ALG7c"> Testing on RTX A4000 GPU </a>

  Practical Test using Lincoln-MKZ vehicle for lane following
   - <a href="https://ncaandt-my.sharepoint.com/:v:/g/personal/tagetahun_ncat_edu/ERGLKlEc3FZElOkbGbgSAwoBE8rOamfPws7On8taXWP3sw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=73Pk23"> 4.5Km test drive </a>
  
 Test on a video by our self-driving car platform - Lincoln MKZ
  - <a href="https://ncaandt-my.sharepoint.com/:v:/g/personal/tagetahun_ncat_edu/EaESHuMcMm1LsHKHCcAGZkgBSxu7w6pSqFN0R2wTofEuYw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=4HNvzl"> Rural road driving - Brown Summit - NC </a>
 
 Test on other videos from YouTube - several driving scenarios and challenges
   <!--- <a href=""> Winding Road, strong shadows, and tunnel </a>-->
   - <a href="https://ncaandt-my.sharepoint.com/:v:/g/personal/tagetahun_ncat_edu/ESieUMnEZ3JPiE0WNe_r_3sBRO0fRnRU9ra0B5R2pMCF6A?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=BmVbFN"> Wet Road, urban traffic </a>
   - <a href="https://ncaandt-my.sharepoint.com/:v:/g/personal/tagetahun_ncat_edu/EcuTbxwpg5RMiSBXOosV-lcBbhxXC3PGjkdYkWWeM2PQFQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=rBYWTP"> Long drive on mostly rural area </a>

## Block-diagram
 <img src="/images/block_diagram.png" width="400" />


## Experimental Results
Lane Detection results for some challenging scenarios
 <img src="/images/sample_0.png" width="700" />

## Pre-Trained Models
 The pre-trained model is available in the ```models/``` folder and includes the following:
  - A PyTorch model saved with epochs, state dictionary, and optimizer state dictionary
  - A serialized and optimized model for inference using ```toch.jit.script()```
  - The model in ONNX format
    
### Testing the serialized model
We tested the serialized model in Ubuntu 22.04 with a conda environment created as follows:
```Shell
# Create conda environment:
conda create -n test_env python==3.10

#Activate the environment
conda activate test_env

# Install pytorch. If gpu is available in your system.
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# if your system doesn't have gpu, conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 cpuonly -c pytorch

#Install opencv:
pip install opencv-python

# Clone this repo:
git clone https://github.com/ACCESSLab/Lane-Detection-using-Segmentation
cd Lane-Detection-using-Segmentation

# Run the Python script:
python test_model.py
```
You can modify the <code> config.py </code> to 
- change prediction threshold
- provide test image path/directory
- provide a path to save the inference results
- change target device (cuda or cpu)

<hr>









