import numbers
import random
import numpy as np
import torch
from scipy import interpolate


class NonlinearJitter(object):
    '''
        https://github.com/deepmind/multidim-image-augmentation/blob/master/doc/color_augmentation_colab.md
       
        This is a experimental augmentation!
        The performance of this augmentation is being verifying
    '''
    def __init__(self, white_point=0.1, black_point=0.1, slope=0.5):
        self.white_point = self._check_input(white_point, 'white_point')
        self.black_point = self._check_input(black_point, 'black_point', center=0, bound=(-1, 1), clip_first_on_zero=False)
        self.slope = self._check_input(slope, 'slope')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def control_point_between_neighbours(left_neighbour, right_neighbour, slope):
        """Insert a new control point between left_neighbour and right_neighbour"""

        middle_x = (right_neighbour[0] - left_neighbour[0]) / 2
        
        min_slope = slope[0]
        max_slope = slope[1]

        max_y = min(left_neighbour[1]  + max_slope * middle_x, 
                    right_neighbour[1] - min_slope * middle_x)
        
        min_y = max(left_neighbour[1]  + min_slope * middle_x, 
                    right_neighbour[1] - max_slope * middle_x)
        
        y = random.uniform(min_y, max_y)
        x = left_neighbour[0] + middle_x
        
        return torch.tensor([x, y])

    @classmethod
    def create_lookup_table(cls, black_point, white_point, slope):
        black_point = torch.tensor([0, random.uniform(black_point[0], black_point[1])])
        white_point = torch.tensor([1, random.uniform(white_point[0], white_point[1])])

        middle = cls.control_point_between_neighbours(black_point, white_point, slope)
        quarter = cls.control_point_between_neighbours(black_point, middle, slope)
        three_quarter = cls.control_point_between_neighbours(middle, white_point, slope)

        vector = torch.stack([black_point, quarter, middle, three_quarter, white_point]) 

        x = vector[:,0]
        y = vector[:,1]

        f = interpolate.interp1d(x, y, kind='quadratic')

        xnew = torch.arange(0, 1, 1/256)
        ynew = f(xnew)
        
        return ynew

    @classmethod
    def get_params(cls, white_point=0.1, black_point=0.1, slope=0.5):
        r_new = cls.create_lookup_table(black_point, white_point, slope)
        g_new = cls.create_lookup_table(black_point, white_point, slope)
        b_new = cls.create_lookup_table(black_point, white_point, slope)
        return [r_new, g_new, b_new]

    
    def __call__(self, img, choice):
        if choice:
            curves = self.get_params(self.white_point, self.black_point, self.slope)

            orig_min, orig_max = img.min(), img.max()
            ratio = (img - orig_min) / (orig_max - orig_min)
            img = np.clip((ratio * 255), 0, 255)

            for i in range(img.shape[0]):
                new_curve = curves[i]
                new_values = new_curve[img[i, ...].astype(np.int)]
                img[i, ...] = new_values

            new_min, new_max = img.min(), img.max()
            ratio = (img - new_min) / (new_max - new_min)
            img = (ratio * (orig_max - orig_min)) + orig_min
        return img
    
    def options(self):
        return [{'choice': b} for b in [True, False]]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'white_point={0}'.format(self.white_point)
        format_string += ', black_point={0}'.format(self.black_point)
        format_string += ', slope={0}'.format(self.slope)
        return format_string