import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
from config import param

#####################################
def preprocess(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img = np.array(img,np.float32)/255.0 # normalize
    img = np.transpose(img,[2,0,1])      # to get CHW form
    img = torch.from_numpy(img)          # shape [3,384,640]
    img = img.unsqueeze(0)               # shape [1,3,384,640],NCHW form
    return img

def postprocess(pred,thresh=0.30):
    msk = (pred>thresh)    
    output = np.zeros_like(pred[:,:], dtype=np.uint8)
    output[msk] = 255
    return output

def infer(img,model,device):
    img = img.to(device)
    pred = model(img) 
    ## model is trained with torch.nn.BCEWithLogitsLoss() => sigmoid is required to get probability   
    pred = F.sigmoid(pred)  
    pred = pred.squeeze(0).squeeze(0)
    pred = pred.detach().cpu().numpy()
    return pred

#############################
if __name__=='__main__':

    ## get parameters from config.py file
    device = param['target_device']
    image_size = param['image_size']

    model_path = param['model_path']
    threshold = param['inference_threshold']

    ## load serialized model
    seg_model = torch.jit.load(model_path).eval()
    seg_model = seg_model.to(device)

    ######## test images #########
    imgs_path = param['test_image_dir']

    ## save results if needed
    save_path = param['result_save_dir']

    imgs = os.listdir(imgs_path)
    for img_name in imgs:#range(len(imgs)):
        test_img_path = os.path.join(imgs_path,img_name)
        
        test_img = cv2.imread(test_img_path)
        test_img = cv2.resize(test_img,image_size)

        ### preprocess
        img = preprocess(test_img)
    
        ### get inference/prediction
        tst = time.time()
        lane_pred = infer(img,seg_model,device)    
        tend = time.time()
        t_delta = np.round((tend-tst)*1000,2)
        print(f'lane pred took: {t_delta} ms')

        # postprocess
        lane = postprocess(lane_pred, threshold)  #shape [384,640]
        
        ## merge prediction with original image
        zero = np.zeros_like(lane_pred,dtype=np.uint8)
        lane = cv2.merge((zero,lane,zero))  #shape (384,640,3)
        result = cv2.addWeighted(test_img,1.0,lane,1.0,0.0)

        ## save result
        cv2.imwrite(os.path.join(save_path,img_name),result)

        ##display result
        cv2.imshow('test result',result)
        cv2.waitKey(0)


