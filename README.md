## Real-time Semantic segmentation-based Lane Detection
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
  
 > Videos by our self-driving platform - Lincoln MKZ
  - <a href="https://youtu.be/h-Oo3QAGmfI"> Rural road driving - Brown Summit - NC </a>
  - <a href="https://youtu.be/mtoy8UmIjJo"> Highway driving - I-40 Greensboro - NC </a>
 
 > On other videos from YouTube - several driving scenarios and challenges
   - <a href="https://youtu.be/HR-Y1Pi0aFM"> Winding Road, strong shadows, and tunnel </a>
   - <a href="https://youtu.be/j5-JM3bYv-8"> Wet Road, urban traffic </a>
   - <a href="https://youtu.be/5uSY_c71Rfc"> Long drive on mostly rural area </a>
<hr>

### Trained Models
 > Pre-trained model is available in the 'models/' folder which contains
  - the pytorch model saved with epochs, state_dictionary, and optimizer state_dict
  - the serialized and optimized for inference model using toch.jit.script()
  - the model in onnx format
