import torch
import onnxruntime

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
        self.shape_check = dummy_input.shape
        torch.onnx.export(self.runner.model.cpu().eval(), dummy_input.cpu(), name+'.onnx')
    
    
    def optimized(self, name):
        '''
            Cache the optimized TensorRT model for best performance
        '''
        self.session = onnxruntime.InferenceSession(name+'.onnx')
        self.input_name = self.session.get_inputs()[0].name
    
    
    def inference(self, data):
        '''
            Inference using the optimzied model
            The data should feed in numpy array
        '''
        assert data.shape == self.shape_check, 'Current the TensorRT inference only support batch_size 1 in onnxruntime'
        return self.session.run(None, {self.input_name: data})