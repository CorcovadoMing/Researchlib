#import seaborn as sns
#sns.set()
import matplotlib.pyplot as plt
import numpy as np

def plot_utils(s, u=None):
    plt.xscale('log')
    if u:
        plt.plot(u, s)
    else:
        plt.plot(s)
    plt.show()
    
def plot_montage(s, row, col):
    try:
        s = s.numpy()
    except:
        pass
    
    if len(s[0].shape) < 3:
        s = s[:, :, :, None]
        
    target_shape = s[0].shape
    result = np.zeros((target_shape[0]*row, target_shape[1]*col, target_shape[2]))
    
    if s.shape[-1] == 1:
        gray_scale = True
    else:
        gray_scale = False
    
    count = 0
    for i in range(row):
        for j in range(col):
            if count < len(s):
                result[target_shape[0]*i:target_shape[0]*(i+1), target_shape[1]*j:target_shape[1]*(j+1), :] = s[count]
                count += 1

    plt.figure(figsize=(20,20))
    if gray_scale:
        plt.imshow(result[:, :, 0], cmap='gray')
    else:
        plt.imshow(result)
    plt.axis('off')
    plt.show()