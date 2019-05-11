import torch.nn.functional as F
from ..utils import _register_method

__methods__ = []
register_method = _register_method(__methods__)

@register_method
def cam(self, vx, final_layer, out_filters, classes):
    if not self.cam_model:
        self.cam_model = nn.Sequential(
                *list(self.model.children())[:final_layer+1], 
                nn.Conv2d(out_filters, classes, 1), 
                AdaptiveConcatPool2d(1),
                nn.Conv2d(classes*2, classes, 1),
                Flatten(),
                nn.LogSoftmax(1)
            ).cuda()
        self.cam_feature_layer = -5
        module_trainable(self.cam_model[:self.cam_feature_layer], False)
        #self.fit_onecycle()
        r = Runner(self.cam_model, self.train_loader, self.test_loader, 'adam', 'nll', fp16=False)
        r.fit(10, 1e-3)

    self.cam_feature = SaveFeatures(self.cam_model[self.cam_feature_layer])
    py = self.cam_model(vx.cuda().float())
    py = F.softmax(py, dim=-1)
    py = py.detach().cpu().numpy()[0]
    feat = self.cam_feature.features[0].detach().cpu().numpy()
    feat = np.maximum(0, feat)
    f2 = np.dot(np.rollaxis(feat,0,3), py)
    f2 -= f2.min()
    f2 /= f2.max()
    dx = vx.cpu().numpy().transpose(0,2,3,1)[0]
    #import skimage
    plt.axis('off')
    #plt.imshow(dx)
    #ss = skimage.transform.resize(f2, dx.shape[:2])
    plt.imshow(f2, alpha=0.5, cmap='hot')
    module_trainable(self.model, True)
    self.cam_feature.remove()
