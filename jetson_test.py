import os
import time
import numpy as np
import torch
import torch_tensorrt
import jetson.utils

from config import param
#############################
if __name__=='__main__':
    
    device = 'cuda' 
    image_size = param['image_size']

    model_path = param['trt_model_path']
    threshold = param['inference_threshold']

    #load tensort model 
    seg_trt_model = torch.jit.load(model_path).cuda() 

    ######## test video #########  
    # video_dir = param['test_video_dir']
    # video_path = os.path.join(video_dir,'utah_vid.mp4') 
    video_path = 'test_videos/udacity_video.mp4'

    camera = jetson.utils.videoSource(video_path)
    display = jetson.utils.videoOutput("display://0")

    resized_img = jetson.utils.cudaAllocMapped(width=640,height=384,format='rgb8')
    
    while True:
        test_img = camera.Capture()
        if test_img is None:
            break
       
        t0 = time.time() # start timer for total latency measurement
        
        ##(1) Preprocessing
        tst_prep = time.time()
        jetson.utils.cudaResize(test_img,resized_img)  ## resize
        ## convert to torch tensor
        tensor_img = torch.as_tensor(resized_img,device='cuda')
        img = tensor_img.to(torch.float32).div_(255.0)
        img = img.permute(2,0,1).contiguous().unsqueeze(0)
        tend_prep = time.time()
        
        ###(2) Inference/prediction
        tst_infer = time.time()
        lane_pred = seg_trt_model(img)
        tend_infer = time.time()
        
        ##(3) Postprocess - mostly for visualization
        tst_post = time.time()
        output = (lane_pred.squeeze()>threshold).to(dtype=torch.uint8)*255
        #combine orginal image and prediction
        result = torch.zeros((384,640,3),dtype=torch.uint8,device='cuda')
        tensor_img = (tensor_img).to(torch.uint8) 
        result[...,1] = output 
        result = result.to(torch.float32)*1.0 +tensor_img.to(torch.float32)*0.7
        result = result.clamp_(0,255).to(torch.uint8)
        tend_post = time.time()

        ## convert torch tensor to cudaImage form for visualization
        result_cuda = jetson.utils.cudaImage(ptr=result.data_ptr(),width=640,height=384,format='rgb8')
           
        tf = time.time()

        ## display the combined result
        display.Render(result_cuda)

        ## Latency estimation
        t_prep  = np.round((tend_prep-tst_prep)*1000,2) 
        t_infer = np.round((tend_infer-tst_infer)*1000,2)  
        t_postp = np.round((tend_post-tst_post)*1000,2)          
        t_delta = np.round((tf-t0)*1000,2) 
        print(f'Latency:')
        print(f' -- prepr: {t_prep} ms')
        print(f' -- infer: {t_infer} ms')
        print(f' -- postp: {t_postp} ms')
        print(f' -- total: {t_delta} ms \n --------------')

        if not camera.IsStreaming() or not display.IsStreaming():
            break

    print('Done.')
