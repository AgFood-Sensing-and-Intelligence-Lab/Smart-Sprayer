import torch
import tensorrt as trt

# Step 1: Convert PyTorch model to ONNX
input_shape = (1, 3, 224, 224)
model = torch.load("/media/agfoodsensinglab/512ssd/WeedGUIProject/DCW-main/YOLOv5/bestS.pt")
model.eval()
dummy_input = torch.randn(input_shape)
torch.onnx.export(model, dummy_input, 'bestSRT.onnx')