## Realtime Semantic Segmentation-Based Lane Detection

### Demo Videos
  :new: Testing TensorRT model on Jetson Orin Nano Super and RTX A4000 GPU
   - <a href="https://drive.google.com/file/d/1rV1CCNVzmVbafmkCB4vTFUKVRGQUsnAU/view?usp=drive_link"> Testing on Jetson Orin Nano Super </a>
   - <a href="https://drive.google.com/file/d/1P_1gD50oi6cWZgGyRwyPo8gdO-ROA3zO/view?usp=sharing"> Testing on RTX A4000 GPU </a>

  Practical Test using Lincoln-MKZ vehicle for lane following
   - <a href="https://drive.google.com/file/d/17PR47yzuDoqEjB3BVDdnxwGd1Zl-9wun/view?usp=sharing"> 4.5Km test drive </a>
  
 Test on a video by our self-driving car platform - Lincoln MKZ
  - <a href="https://drive.google.com/file/d/1UVaQ9m5bIdyIH1S_aAp4uc74WZMZ6l6x/view?usp=sharing"> Rural road driving - Brown Summit - NC </a>
 
 Test on other videos from YouTube - several driving scenarios and challenges
   - <a href=""> Winding Road, strong shadows, and tunnel </a>
   - <a href="https://drive.google.com/file/d/1OOU7RTUBEW8-I-DKV36T-yIZDx92nvtD/view?usp=sharing"> Wet Road, urban traffic </a>
   - <a href=""> Long drive on mostly rural area </a>

## Block-diagram
 <img src="/images/block_diagram.png" width="400" />


## Experimental Results
Lane Detection results for some challenging scenarios
 <img src="/images/sample_0.png" width="700" />
 
 <!-- <img src="/images/sample_1.png" width="700" /> -->

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

# Run the python script:
python test_model.py
```
You can modify the <code> config.py </code> to 
- change prediction threshold
- provide test image path/directory
- provide a path to save the inference results
- change target device (cuda or cpu)

<hr>





