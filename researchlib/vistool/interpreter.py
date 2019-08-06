import torch
from torch import nn
from torch.autograd import Function, Variable, grad
from torch._thnn import type2backend
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import cv2
import types

class EBLinear(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias=None):
        ctx.save_for_backward(inp, weight, bias)
        output = inp.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors

        wplus = weight.clone().clamp(min=0)

        output = inp.matmul(wplus.t())
        normalized_grad_output = grad_output / (output + 1e-10)
        normalized_grad_output = normalized_grad_output * (output > 0).float()

        grad_inp = normalized_grad_output.matmul(wplus)
        grad_inp = grad_inp * inp

        return grad_inp, None, None


def _output_size(inp, weight, pad, dilation, stride):
    pad = pad[0]
    dilation = dilation[0]
    stride = stride[0]

    channels = weight.size(0)
    output_size = (inp.size(0), channels)
    for d in range(inp.dim() - 2):
        in_size = inp.size(d + 2)
        kernel = dilation * (weight.size(d + 2) - 1) + 1
        output_size += ((in_size + (2 * pad) - kernel) // stride + 1,)
    if not all(map(lambda s: s > 0, output_size)):
        raise ValueError("convolution inp is too small (output would be {})".format(
            'x'.join(map(str, output_size))))
    return output_size


class EBConv2d(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(inp, weight, bias)

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        kH, kW = weight.size(2), weight.size(3)

        output_size = _output_size(inp, weight, padding, dilation, stride)
        output = inp.new(*output_size)
        columns = inp.new(*output_size)
        ones = inp.new(*output_size)

        backend = type2backend[inp.type()]
        f = getattr(backend, 'SpatialConvolutionMM_updateOutput')
        f(backend.library_state, inp, output, weight, bias, columns, ones,
          kH, kW, ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        output_size = _output_size(inp, weight, padding, dilation, stride)
        kH, kW = weight.size(2), weight.size(3)

        wplus = weight.clone().clamp(min=0)
        new_output = inp.new(*output_size)
        columns = inp.new(*output_size)
        ones = inp.new(*output_size)

        backend = type2backend[inp.type()]
        f = getattr(backend, 'SpatialConvolutionMM_updateOutput')
        f(backend.library_state, inp, new_output, wplus, None, columns, ones,
          kH, kW, ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1])

        normalized_grad_output = grad_output.data / (new_output + 1e-10)
        normalized_grad_output = normalized_grad_output * (new_output > 0).float()

        grad_inp = inp.new()
        grad_inp.resize_as_(inp)

        g = getattr(backend, 'SpatialConvolutionMM_updateGradInput')
        g(backend.library_state, inp, normalized_grad_output, grad_inp, wplus, columns, ones,
          kH, kW, ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1])

        grad_inp = grad_inp * inp

        return Variable(grad_inp), None, None, None, None, None, None


class EBAvgPool2d(Function):
    @staticmethod
    def forward(ctx, inp, kernel_size, stride=None, padding=0,
                ceil_mode=False, count_include_pad=True):
        ctx.kernel_size = (kernel_size, kernel_size)
        stride = stride if stride is not None else kernel_size
        ctx.stride = (stride, stride)
        ctx.padding = (padding, padding)
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad
        backend = type2backend[type(inp)]
        output = inp.new()

        backend.SpatialAveragePooling_updateOutput(
            backend.library_state,
            inp, output,
            ctx.kernel_size[1], ctx.kernel_size[0],
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.ceil_mode, ctx.count_include_pad)

        ctx.save_for_backward(inp, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        backend = type2backend[type(grad_output.data)]
        inp, output = ctx.saved_tensors

        normalized_grad_output = grad_output.data / (output + 1e-10)
        normalized_grad_output = normalized_grad_output * (output > 0).float()

        grad_inp = inp.new()

        backend.SpatialAveragePooling_updateGradInput(
            backend.library_state,
            inp, normalized_grad_output, grad_inp,
            ctx.kernel_size[1], ctx.kernel_size[0],
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.ceil_mode, ctx.count_include_pad)

        grad_inp = grad_inp * inp

        return Variable(grad_inp), None, None, None, None, None

    

class _Interpreter:
    def __init__(self):
        pass

    def saliency_map(self, model, img, label, plot=True):
        '''
            Deep inside convolutional networks: Visualising image classification models and saliency maps
            https://arxiv.org/abs/1312.6034

            @input: classification model built from builder
            @output: salience
        '''

        # Deep copy the model
        model = pickle.loads(pickle.dumps(model))
        model.nnlist = model.nnlist[:-1]
        var_img = Variable(img.data, requires_grad=True)
        output = model(var_img[None, :, :, :])
        loss = output[0, label]
        saliencymap = grad(outputs=loss, inputs=var_img, create_graph=True)[0]
        saliencymap = saliencymap.abs()
        saliencymap, _ = saliencymap.max(dim=0)
        # Normalize
        saliencymap -= saliencymap.min()
        saliencymap /= saliencymap.max()

        if plot:
            plot_img = img - img.min()
            plot_img /= plot_img.max()
            _, arr = plt.subplots(1, 3)
            arr[0].imshow(plot_img.detach().numpy().transpose(1,2,0))
            arr[0].set_title('Input')
            arr[0].axis('off')
            arr[1].imshow(saliencymap.detach().numpy(), cmap='gray')
            arr[1].set_title('Saliency Map')
            arr[1].axis('off')
            arr[2].imshow(plot_img.detach().numpy().transpose(1,2,0))
            arr[2].imshow(saliencymap.detach().numpy(), cmap='hot', alpha=0.3)
            arr[2].set_title('Saliency map on input')
            arr[2].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            return saliencymap
    
    
    def grad_cam(self, model, img, label, plot=True):
        '''
            Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization
            https://arxiv.org/pdf/1610.02391v1

            @input: classification model built from builder
            @output: gradcam map
        '''

        hook_module = []
        hook_forward_buffer = []
        hook_backward_buffer = []

        def backward_hook_fn(module, input, output):
            hook_backward_buffer.append(input)

        def forward_hook_fn(module, input, output):
            hook_forward_buffer.append(output)

        def _register_conv2d(m):
            if type(m) == nn.Conv2d:
                if 1 not in m.kernel_size: # avoid 1x1 conv
                    b_handler = m.register_backward_hook(backward_hook_fn)
                    f_handler = m.register_forward_hook(forward_hook_fn)
                    hook_module.append(b_handler)
                    hook_module.append(f_handler)

        # Deep copy the model
        model = pickle.loads(pickle.dumps(model))
        # Find last convolution
        model.apply(_register_conv2d)
        output = model(img[None, :, :, :])
        if type(model.nnlist[-1]) == nn.Sigmoid:
            loss = F.binary_cross_entropy(output, label)
        else:
            loss = -output[0, label].log()
        loss.backward()

        target_forward = hook_forward_buffer[-1]
        for i in hook_backward_buffer[0]:
            try:
                shape = i.shape
            except:
                continue
            if target_forward.shape[:2] == shape[:2]:
                target_gradient = i

        target_gradient = F.adaptive_avg_pool2d(target_gradient, 1).squeeze()
        target_forward = target_forward.squeeze()
        target_output = target_forward * target_gradient[:, None, None]
        target_output = F.relu(target_output)
        cam = target_output.sum(0)
        cam -= cam.min()
        cam /= cam.max()
        cam = cv2.resize(cam.detach().numpy(), img.shape[1:])

        if plot:
            plot_img = img - img.min()
            plot_img /= plot_img.max()
            plot_img = plot_img.detach().numpy().transpose(1,2,0)
            _, arr = plt.subplots(1, 3)
            if plot_img.shape[-1] == 1:
                arr[0].imshow(plot_img[:, :, 0], cmap='gray')
            else:
                arr[0].imshow(plot_img)
            arr[0].set_title('Input')
            arr[0].axis('off')
            arr[1].imshow(cam, cmap='gray')
            arr[1].set_title('Grad-CAM')
            arr[1].axis('off')
            if plot_img.shape[-1] == 1:
                arr[2].imshow(plot_img[:, :, 0], cmap='gray')
            else:
                arr[2].imshow(plot_img)
            arr[2].imshow(cam, cmap='hot', alpha=0.3)
            arr[2].set_title('Grad-CAM on input')
            arr[2].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            return torch.from_numpy(cam) # We just want the output still a torch tensor


    def excitation_backprop(self, model, img, label, contrastive=False, plot=True):
        '''
            Top-down Neural Attention by Excitation Backprop
            https://arxiv.org/abs/1608.00507

            Non-Contrastive: P_0 * P_1 * P_2 * ... * P_{N-1}
            Contrstive: P_0 * (P_1 - P_~1) * P-2 * ... * P_{N-1} where P_~1 is with the negative weight of P_1

        '''

        def new_linear(self, x):
            return EBLinear.apply(x, self.weight, self.bias)
        def new_conv2d(self, x):
            return EBConv2d.apply(x, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)
        def new_avgpool2d(self, x):
            return EBAvgPool2d.apply(x, self.kernel_size, self.stride,
                                     self.padding, self.ceil_mode, self.count_include_pad)
        def _replace(m):
            name = m.__class__.__name__
            if name == 'Linear':
                m.forward = types.MethodType(new_linear, m)
            elif name == 'Conv2d':
                m.forward = types.MethodType(new_conv2d, m)
            elif name == 'AvgPool2d':
                m.forward = types.MethodType(new_avgpool2d, m)

        hook_module = []
        hook_forward_buffer = []

        def forward_hook_fn(module, input, output):
            hook_forward_buffer.append(output)

        def _register_conv2d(m):
            name = m.__class__.__name__
            if name == 'Conv2d':
                if 1 not in m.kernel_size: # avoid 1x1 conv
                    f_handler = m.register_forward_hook(forward_hook_fn)
                    hook_module.append(f_handler)

        model = pickle.loads(pickle.dumps(model))
        model.nnlist = model.nnlist[:-1]
        model.apply(_replace)
        model.apply(_register_conv2d)

        output = model(img[None, :, :, :])
        grad_out = output.clone()
        grad_out.fill_(0.)
        grad_out[0, label] = 1.

        if contrastive:
            model.nnlist[-1].weight.data *= -1.0
            neg_map = grad(output, hook_forward_buffer[-1], grad_out, create_graph=True)[0]

            model.nnlist[-1].weight.data *= -1.0
            pos_map = grad(output, hook_forward_buffer[-1], grad_out, create_graph=True)[0]

            diff = pos_map - neg_map
            attmap = grad(hook_forward_buffer[-1], hook_forward_buffer[0], diff, create_graph=True)[0]
        else:
            attmap = grad(output, hook_forward_buffer[0], grad_out, create_graph=True)[0]

        attmap = attmap.sum(1).squeeze()
        attmap -= attmap.min()
        attmap /= attmap.max()
        attmap = cv2.resize(attmap.detach().numpy(), img.shape[1:])

        if plot:
            _, arr = plt.subplots(1, 3)
            plot_img = img - img.min()
            plot_img /= plot_img.max()
            arr[0].imshow(plot_img.detach().numpy().transpose(1,2,0))
            arr[0].set_title('Input')
            arr[0].axis('off')
            arr[1].imshow(attmap, cmap='gray')
            arr[1].set_title('EB')
            arr[1].axis('off')
            arr[2].imshow(plot_img.detach().numpy().transpose(1,2,0))
            arr[2].imshow(attmap, cmap='hot', alpha=0.3)
            arr[2].set_title('EB on input')
            arr[2].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            return torch.from_numpy(attmap)