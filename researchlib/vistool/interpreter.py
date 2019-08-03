import torch
from torch.autograd import Variable, grad
import pickle
import matplotlib.pyplot as plt

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
            arr[2].imshow(saliencymap.detach().numpy(), cmap='hot', alpha=0.5)
            arr[2].set_title('Saliency map on input')
            arr[2].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            return saliencymap