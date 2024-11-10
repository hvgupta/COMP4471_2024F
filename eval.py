from torch import Tensor
from piqa import MS_SSIM

class MSSSIM_Loss(MS_SSIM):
    def __init__(self,window_size: int = 11,
        sigma: float = 1.5,
        n_channels: int = 3,
        weights: Tensor = None,
        reduction: str = 'mean',
        padding: bool = False,
        value_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03):
        kwargs = {
            'padding': padding,
            'value_range': value_range,
            'k1': k1,
            'k2': k2,
        }
        super(MSSSIM_Loss, self).__init__(window_size, sigma, n_channels, weights, reduction, **kwargs)

    def forward(self, x, y):
        return 1 - super(MSSSIM_Loss, self).forward(x, y)