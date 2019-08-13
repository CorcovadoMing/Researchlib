import torch
import onnxruntime
import os
from torch2trt import torch2trt


class _Export:
    def __init__(self, runner):
        self.runner = runner

    def save_onnx(self, name, shape):
        '''
            Export the model to onnx format
            
            @name: model save name `{name}.onnx`
            @shape: input size, e.g., (3, 32, 32) for Cifar10
        '''
        dummy_input = torch.randn(1, *shape)
        torch.onnx.export(self.runner.model.cpu().eval(), dummy_input.cpu(),
                          name + '.onnx')

    def optimized(self, name, max_batch_size=1024):
        '''
            Cache the optimized TensorRT model for best performance
            
            According to `https://github.com/microsoft/onnxruntime/blob/1c5b15c2b89432f03877c8845f2d854387543aea/onnxruntime/core/providers/tensorrt/tensorrt_execution_provider.cc#L332` the TensorRT max_batch_size is set by environment variable `ORT_TENSORRT_MAX_BATCH_SIZE`, so we need to set it during optimized stage
        '''

        os.environ['ORT_TENSORRT_MAX_BATCH_SIZE'] = str(max_batch_size)
        self.session = onnxruntime.InferenceSession(name + '.onnx')
        self.input_name = self.session.get_inputs()[0].name

    def inference(self, data):
        '''
            Inference using the optimzied model
            The data should feed in numpy array
        '''
        return self.session.run(None, {self.input_name: data})

    def weak_optimized(self, shape):
        '''
            Only limited OP sets supported
        '''
        dummy_input = torch.randn(1, *shape).cuda()
        self.trt_model = torch2trt(self.runner.model.cuda().eval(), [dummy_input])
        
    def weak_inference(self, data):
        '''
            Only limited OP sets supported
        '''
        return self.trt_model(data)
    