import torch
import numpy as np
import torch.nn.functional as Func
import math

def ceildiv(a, b):
    return -(-a // b)


# Decimated Laplacian pyramid
class lpyr_dec():

    def __init__(self, W, H, ppd, device):
        self.device = device
        self.ppd = ppd
        self.min_freq = 0.1
        self.W = W
        self.H = H

        max_levels = int(np.floor(np.log2(min(self.H, self.W)))) - 1

        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0, 14.0)) * 0.3228], 0) * self.ppd / 2.0

        # print(max_levels)
        # print(bands)
        # sys.exit(0)

        invalid_bands = np.array(np.nonzero(bands <= self.min_freq))  # we want to find first non0, length is index+1

        if invalid_bands.shape[-2] == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        # max_band+1 below converts index into count
        self.height = np.clip(max_band + 1, 0, max_levels)  # int(np.clip(max(np.ceil(np.log2(ppd)), 1.0)))
        self.band_freqs = np.array([1.0] + [0.3228 * 2.0 ** (-f) for f in range(self.height)]) * self.ppd / 2.0

        self.pyr_shape = self.height * [None]  # shape (W,H) of each level of the pyramid
        self.pyr_ind = self.height * [None]  # index to the elements at each level

        cH = H
        cW = W
        for ll in range(self.height):
            self.pyr_shape[ll] = (cH, cW)
            cH = ceildiv(H, 2)
            cW = ceildiv(W, 2)

    def get_freqs(self):
        return self.band_freqs

    def get_band_count(self):
        return self.height + 1

    def get_band(self, bands, band):
        if band == 0 or band == (len(bands) - 1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        return bands[band] * band_mul

    def set_band(self, bands, band, data):
        if band == 0 or band == (len(bands) - 1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        bands[band] = data / band_mul

    def get_gband(self, gbands, band):
        return gbands[band]

    # def get_gband_count(self):
    #     return self.height #len(gbands)

    # def clear(self):
    #     for pyramid in self.P:
    #         for level in pyramid:
    #             # print ("deleting " + str(level))
    #             del level

    def decompose(self, image):
        # assert len(image.shape)==4, "NCHW (C==1) is expected, got " + str(image.shape)
        # assert image.shape[-2] == self.H
        # assert image.shape[-1] == self.W

        # self.image = image

        return self.laplacian_pyramid_dec(image, self.height + 1)

    def reconstruct(self, bands):
        img = bands[-1]

        for i in reversed(range(0, len(bands) - 1)):
            img = self.gausspyr_expand(img, [bands[i].shape[-2], bands[i].shape[-1]])
            img += bands[i]

        return img

    def laplacian_pyramid_dec(self, image, levels=-1, kernel_a=0.4):
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        for i in range(height - 1):
            layer = gpyr[i] - self.gausspyr_expand(gpyr[i + 1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
            lpyr.append(layer)

        lpyr.append(gpyr[height - 1])

        # print("laplacian pyramid summary:")
        # print("self.height = %d" % self.height)
        # print("height      = %d" % height)
        # print("len(lpyr)   = %d" % len(lpyr))
        # print("len(gpyr)   = %d" % len(gpyr))
        # sys.exit(0)

        return lpyr, gpyr

    def interleave_zeros_and_pad(self, x, exp_size, dim):
        new_shape = [*x.shape]
        new_shape[dim] = exp_size[dim] + 4
        z = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
        odd_no = (exp_size[dim] % 2)
        if dim == -2:
            z[:, :, 2:-2:2, :] = x
            z[:, :, 0, :] = x[:, :, 0, :]
            z[:, :, -2 + odd_no, :] = x[:, :, -1, :]
        elif dim == -1:
            z[:, :, :, 2:-2:2] = x
            z[:, :, :, 0] = x[:, :, :, 0]
            z[:, :, :, -2 + odd_no] = x[:, :, :, -1]
        else:
            assert False, "Wrong dimension"

        return z

    def gaussian_pyramid_dec(self, image, levels=-1, kernel_a=0.4):

        default_levels = int(np.floor(np.log2(min(image.shape[-2], image.shape[-1]))))

        if levels == -1:
            levels = default_levels
        if levels > default_levels:
            raise Exception("Too many levels (%d) requested. Max is %d for %s" % (levels, default_levels, image.shape))

        res = [image]

        for i in range(1, levels):
            res.append(self.gausspyr_reduce(res[i - 1], kernel_a))

        return res

    def sympad(self, x, padding, axis):
        if padding == 0:
            return x
        else:
            beg = torch.flip(torch.narrow(x, axis, 0, padding), [axis])
            end = torch.flip(torch.narrow(x, axis, -padding, padding), [axis])

            return torch.cat((beg, x, end), axis)

    def get_kernels(self, im, kernel_a=0.4):

        ch_dim = len(im.shape) - 2
        if hasattr(self, "K_horiz") and ch_dim == self.K_ch_dim:
            return self.K_vert, self.K_horiz

        K = torch.tensor([0.25 - kernel_a / 2.0, 0.25, kernel_a, 0.25, 0.25 - kernel_a / 2.0], device=im.device,
                         dtype=im.dtype)
        self.K_vert = torch.reshape(K, (1,) * ch_dim + (K.shape[0], 1))
        self.K_horiz = torch.reshape(K, (1,) * ch_dim + (1, K.shape[0]))
        self.K_ch_dim = ch_dim
        return self.K_vert, self.K_horiz

    def gausspyr_reduce(self, x, kernel_a=0.4):

        K_vert, K_horiz = self.get_kernels(x, kernel_a)

        B, C, H, W = x.shape
        y_a = Func.conv2d(x.reshape(-1, 1, H, W), K_vert, stride=(2, 1), padding=(2, 0)).reshape(B, C, -1, W)

        # Symmetric padding
        y_a[:, :, 0, :] += x[:, :, 0, :] * K_vert[0, 0, 1, 0] + x[:, :, 1, :] * K_vert[0, 0, 0, 0]
        if (x.shape[-2] % 2) == 1:  # odd number of rows
            y_a[:, :, -1, :] += x[:, :, -1, :] * K_vert[0, 0, 3, 0] + x[:, :, -2, :] * K_vert[0, 0, 4, 0]
        else:  # even number of rows
            y_a[:, :, -1, :] += x[:, :, -1, :] * K_vert[0, 0, 4, 0]

        H = y_a.shape[-2]
        y = Func.conv2d(y_a.reshape(-1, 1, H, W), K_horiz, stride=(1, 2), padding=(0, 2)).reshape(B, C, H, -1)

        # Symmetric padding
        y[:, :, :, 0] += y_a[:, :, :, 0] * K_horiz[0, 0, 0, 1] + y_a[:, :, :, 1] * K_horiz[0, 0, 0, 0]
        if (x.shape[-2] % 2) == 1:  # odd number of columns
            y[:, :, :, -1] += y_a[:, :, :, -1] * K_horiz[0, 0, 0, 3] + y_a[:, :, :, -2] * K_horiz[0, 0, 0, 4]
        else:  # even number of columns
            y[:, :, :, -1] += y_a[:, :, :, -1] * K_horiz[0, 0, 0, 4]

        return y

    def gausspyr_expand_pad(self, x, padding, axis):
        if padding == 0:
            return x
        else:
            beg = torch.narrow(x, axis, 0, padding)
            end = torch.narrow(x, axis, -padding, padding)

            return torch.cat((beg, x, end), axis)

    # This function is (a bit) faster
    def gausspyr_expand(self, x, sz=None, kernel_a=0.4):
        if sz is None:
            sz = [x.shape[-2] * 2, x.shape[-1] * 2]

        K_vert, K_horiz = self.get_kernels(x, kernel_a)

        y_a = self.interleave_zeros_and_pad(x, dim=-2, exp_size=sz)

        B, C, H, W = y_a.shape
        y_a = Func.conv2d(y_a.reshape(-1, 1, H, W), K_vert * 2).reshape(B, C, -1, W)

        y = self.interleave_zeros_and_pad(y_a, dim=-1, exp_size=sz)
        B, C, H, W = y.shape

        y = Func.conv2d(y.reshape(-1, 1, H, W), K_horiz * 2).reshape(B, C, H, -1)

        return y

    def interleave_zeros(self, x, dim):
        z = torch.zeros_like(x, device=self.device)
        if dim == 2:
            return torch.cat([x, z], dim=3).reshape(x.shape[0], x.shape[1], 2 * x.shape[2], x.shape[3])
        elif dim == 3:
            return torch.cat([x.permute(0, 1, 3, 2), z.permute(0, 1, 3, 2)], dim=3).reshape(x.shape[0], x.shape[1],
                                                                                         2 * x.shape[3],
                                                                                         x.shape[2]).permute(0, 1, 3, 2)

# Decimated Laplacian pyramid with a bit better interface - stores all bands within the object
class lpyr_dec_2(lpyr_dec):

    def __init__(self, W, H, ppd, device, keep_gaussian=False):
        self.device = device
        self.ppd = ppd #110?
        self.min_freq = 0.2
        self.W = W
        self.H = H
        self.keep_gaussian=keep_gaussian

        max_levels = int(np.floor(np.log2(min(self.H, self.W))))-1

        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0,14.0)) * 0.3228], 0) * self.ppd/2.0

        # print(max_levels)
        # print(bands)
        # sys.exit(0)

        invalid_bands = np.array(np.nonzero(bands <= self.min_freq)) # we want to find first non0, length is index+1

        if invalid_bands.shape[-2] == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        # max_band+1 below converts index into count
        self.height = np.clip(max_band+1, 0, max_levels) # int(np.clip(max(np.ceil(np.log2(ppd)), 1.0)))
        self.band_freqs = np.array([1.0] + [0.3228 * 2.0 **(-f) for f in range(self.height)]) * self.ppd/2.0

        self.pyr_shape = self.height * [None] # shape (W,H) of each level of the pyramid
        self.pyr_ind = self.height * [None]   # index to the elements at each level

        cH = H
        cW = W
        for ll in range(self.height):
            self.pyr_shape[ll] = (cH, cW)
            cH = ceildiv(H,2)
            cW = ceildiv(W,2)

        self.lbands = [None] * (self.height+1) # Laplacian pyramid bands
        if self.keep_gaussian:
            self.gbands = [None] * (self.height+1) # Gaussian pyramid bands

    def get_freqs(self):
        return self.band_freqs

    def get_band_count(self):
        return self.height+1

    def get_lband(self, band):
        if band == 0 or band == (len(self.lbands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        return self.lbands[band] * band_mul

    def set_lband(self, band, data):
        if band == 0 or band == (len(self.lbands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        self.lbands[band] = data / band_mul

    def get_gband(self, band):
        return self.gbands[band]

    # def clear(self):
    #     for pyramid in self.P:
    #         for level in pyramid:
    #             # print ("deleting " + str(level))
    #             del level

    def decompose(self, image):
        return self.laplacian_pyramid_dec(image, self.height+1)

    def reconstruct(self):
        img = self.lbands[-1]

        for i in reversed(range(0, len(self.lbands)-1)):
            img = self.gausspyr_expand(img, [self.lbands[i].shape[-2], self.lbands[i].shape[-1]])
            img += self.lbands[i]

        return img

    def laplacian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return

        lpyr = []
        for i in range(height-1):
            layer = gpyr[i] - self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
            lpyr.append(layer)

        lpyr.append(gpyr[height-1])
        self.lbands = lpyr

        if self.keep_gaussian:
            self.gbands = gpyr

class weber_contrast_pyr(lpyr_dec):

    def __init__(self, W, H, ppd, device, contrast):
        super().__init__(W, H, ppd, device)
        self.contrast = contrast

    def decompose(self, image):
        levels = self.height+1
        kernel_a = 0.4
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        L_bkg_pyr = []
        for i in range(height):
            is_baseband = (i==(height-1))

            if is_baseband:
                layer = gpyr[i]
                if self.contrast.endswith('ref'):
                    L_bkg = torch.clamp(gpyr[i][...,1:2,:,:,:], min=0.01)
                else:
                    L_bkg = torch.clamp(gpyr[i][...,0:2,:,:,:], min=0.01)
                    # The sustained channels use the mean over the image as the background. Otherwise, they would be divided by itself and the contrast would be 1.
                    L_bkg_mean = torch.mean(L_bkg, dim=[-1, -2], keepdim=True)
                    L_bkg = L_bkg.repeat([int(image.shape[-4]/2), 1, 1, 1])
                    L_bkg[0:2,:,:,:] = L_bkg_mean
            else:
                glayer_ex = self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
                layer = gpyr[i] - glayer_ex

                # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
                # L_bkg is set to ref-sustained
                if self.contrast == 'weber_g1_ref':
                    L_bkg = torch.clamp(glayer_ex[...,1:2,:,:,:], min=0.01)
                elif self.contrast == 'weber_g1':
                    L_bkg = torch.clamp(glayer_ex[...,0:2,:,:,:], min=0.01)
                elif self.contrast == 'weber_g0_ref':
                    L_bkg = torch.clamp(gpyr[i][...,1:2,:,:,:], min=0.01)
                else:
                    raise RuntimeError( f"Contrast {self.contrast} not supported")

            if L_bkg.shape[-4]==2: # If L_bkg NOT identical for the test and reference images
                contrast = torch.empty_like(layer)
                contrast[...,0::2,:,:,:] = torch.clamp(torch.div(layer[...,0::2,:,:,:], L_bkg[...,0,:,:,:]), max=1000.0)
                contrast[...,1::2,:,:,:] = torch.clamp(torch.div(layer[...,1::2,:,:,:], L_bkg[...,1,:,:,:]), max=1000.0)
            else:
                contrast = torch.clamp(torch.div(layer, L_bkg), max=1000.0)

            lpyr.append(contrast)
            L_bkg_pyr.append(torch.log10(L_bkg))

        return lpyr, L_bkg_pyr

class laplacian_pyramid_simple(lpyr_dec):
    def __init__(self, W, H, ppd, device):
        super().__init__(W, H, ppd, device)

    def decompose(self, image, levels=None):
        """
        Args:
            image: Tensor of shape [B, C, H, W]
        Returns:
            lpyr: list of Laplacian pyramid layers, each of shape [B, C, H_i, W_i]
        """
        if levels is None:
            levels = self.height + 1
        kernel_a = 0.4

        # Build Gaussian pyramid
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        # Compute Laplacian pyramid
        lpyr = []
        for i in range(len(gpyr)):
            is_baseband = (i == len(gpyr) - 1)

            if is_baseband:
                layer = gpyr[i]
            else:
                glayer_ex = self.gausspyr_expand(gpyr[i + 1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
                layer = gpyr[i] - glayer_ex

            lpyr.append(layer)

        return gpyr, lpyr

class laplacian_pyramid_simple_contrast(lpyr_dec):
    def __init__(self, W, H, ppd, device, contrast):
        super().__init__(W, H, ppd, device)
        self.contrast = contrast

    def decompose(self, image, levels=None):
        if levels is None:
            levels = self.height + 1
        kernel_a = 0.4
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        # Compute Laplacian pyramid
        # lpyr = []
        lpyr_contrast = []
        L_bkg_pyr = []
        for i in range(height):
            is_baseband = (i==(height-1))
            if is_baseband:
                layer = gpyr[i]
                L_bkg = torch.clamp(gpyr[i][..., 0:1, :, :], min=0.01)
                # The sustained channels use the mean over the image as the background. Otherwise, they would be divided by itself and the contrast would be 1.
                L_bkg_mean = torch.mean(L_bkg, dim=[-1, -2], keepdim=True)
                L_bkg = L_bkg.repeat([1, 3, 1, 1])
                L_bkg[:, 0:1, :, :] = L_bkg_mean
            else:
                glayer_ex = self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
                layer = gpyr[i] - glayer_ex

                # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
                # L_bkg is set to ref-sustained
                if self.contrast == 'weber_g1':
                    L_bkg = torch.clamp(glayer_ex[...,0:1,:,:], min=0.01)
                else:
                    raise RuntimeError( f"Contrast {self.contrast} not supported")

            contrast = torch.clamp(torch.div(layer, L_bkg), max=1000.0)
            # lpyr.append(layer)
            lpyr_contrast.append(contrast)
            L_bkg_pyr.append(torch.log10(L_bkg))

        return lpyr_contrast, L_bkg_pyr
