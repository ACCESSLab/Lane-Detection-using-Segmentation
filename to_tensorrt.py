import torch
import torch_tensorrt

## load serialized model
model_path='models/lane_seg_model_torchjit_serialized.pth'

seg_model = torch.jit.load(model_path).eval().cuda()

dummy_inputs = [torch.randn((1,3,384,640)).cuda()] # test input

trt_model = torch_tensorrt.compile(seg_model,  inputs=dummy_inputs) 

## save trt model
torch.jit.save(trt_model,'models/trt_model.ts')

print("Compiled model to Tensorrt and saved.")


