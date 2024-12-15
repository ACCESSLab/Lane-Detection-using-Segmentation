## Realtime Semantic segmentation-based Lane Detection
 ----
### Architecture

 > <img src="/images/block_diagram.png" width="400" />


### Experimental Results
> Lane Detection results in some challenging scenarios
 <img src="/images/sample_0.png" width="700" />
 <hr>
 <img src="/images/sample_1.png" width="700" /> 


### Demo Videos
 > Practical Test using Lincoln-MKZ vehicle for lane following

   [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/E3Gwv1mPJ2E/0.jpg)](https://youtu.be/E3Gwv1mPJ2E)
  
 > Test on videos by our self-driving car platform - Lincoln MKZ
  - <a href="https://youtu.be/h-Oo3QAGmfI"> Rural road driving - Brown Summit - NC </a>
  - <a href="https://youtu.be/mtoy8UmIjJo"> Highway driving - I-40 Greensboro - NC </a>
 
 > Test on other videos from YouTube - several driving scenarios and challenges
   - <a href="https://youtu.be/HR-Y1Pi0aFM"> Winding Road, strong shadows, and tunnel </a>
   - <a href="https://youtu.be/j5-JM3bYv-8"> Wet Road, urban traffic </a>
   - <a href="https://youtu.be/5uSY_c71Rfc"> Long drive on mostly rural area </a>
<hr>

### Pre-Trained Models
 > The pre-trained model is available in the ```models/``` folder and includes the following:
  - A pytorch model saved with epochs, state dictionary, and optimizer state dictionary
  - A serialized and optimized model for inference using ```toch.jit.script()```
  - The model in ONNX format
    
### Testing the serialized model
> We tested the serialized model in Ubuntu 22.04 with a conda environment created as follows:
```Shell
# Create conda environment:
conda create -n test_env python==3.10

#Activate the environment
conda activate test_env

# Install pytorch:
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

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
