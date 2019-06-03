from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np

class _Image:
    def __init__(self):
        pass
    
    def _browser_ui(self, index=0):
        data = self.data[index].numpy()
        
        if data.ndim < 3: gray = True
        else: data = data.transpose(1, 2, 0)
        
        print('Mean', data.mean())
        print('Std', data.std())
        print('Min', data.min())
        print('Max', data.max())
        
        fig, arr = plt.subplots(1, 2)
        if gray: arr[0].imshow(data, cmap='gray')
        else: arr[0].imshow(data)
        arr[1].hist(data)
        asp = np.diff(arr[1].get_xlim())[0] / np.diff(arr[1].get_ylim())[0]
        asp /= np.abs(np.diff(arr[0].get_xlim())[0] / np.diff(arr[0].get_ylim())[0])
        arr[1].set_aspect(asp)
        plt.tight_layout()
        plt.show()
        
    def browser(self, loader):
        self.data = loader.dataset.data
        _ = interact(self._browser_ui, index=range(min(100, len(self.data))), continuous_update=False)
        
        