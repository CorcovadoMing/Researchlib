import torch
from torch import nn
from torch.autograd import Variable, grad
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import cv2

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
        loss = -output[0, label].log()
        loss.backward()

        target_forward = hook_forward_buffer[-1]
        for i in hook_backward_buffer[0]:
            try:
                shape = i.shape
            except:
                continue

            if target_forward.shape == shape:
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
            _, arr = plt.subplots(1, 3)
            arr[0].imshow(plot_img.detach().numpy().transpose(1,2,0))
            arr[0].set_title('Input')
            arr[0].axis('off')
            arr[1].imshow(cam, cmap='gray')
            arr[1].set_title('Grad-CAM')
            arr[1].axis('off')
            arr[2].imshow(plot_img.detach().numpy().transpose(1,2,0))
            arr[2].imshow(cam, cmap='hot', alpha=0.3)
            arr[2].set_title('Grad-CAM on input')
            arr[2].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            return torch.from_numpy(cam) # We just want the output still a torch tensor