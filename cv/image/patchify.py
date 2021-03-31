import copy
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor

__all__ = ["ImageSplitter"]


class ImageSplitter:
    """ Patchify and unpatchify an image for super-resolution. 

        The input image will be splitted into patches with overlapping paddings
        to avoid edge discontinuity. You can specify a padding mode from 
        ('reflection', 'replication', 'zero', 'const'), to let the program doing 
        your desired padding operation. 
        `patchify` should be used with `unpatchify` in pair for each image.

    Args:
        seg_size: The size of each patch. Patches are squares which have size (seg_size, seg_size)
        scale_factor: The upscaling scale you are going to use.
        channel_num: The input channel number. For RGB, it is 3.
        boarder_pad_size: The padding size for each patch. Default is 3.
        pad_mode: The padding method you are going to use. Select from (reflection|replication|zero|const)
        padding_value: If you choose const padding, the const value.

    From:
        https://github.com/yu45020/Waifu2x/blob/master/utils/prepare_images.py
        https://github.com/nagadomi/waifu2x/issues/238

    Example Usage:
        img = np.random.rand(250, 250, 3)
        up_module = nn.Upsample(scale_factor=2, mode='bilinear')
        img_splitter = ImageSplitter(seg_size=64, scale_factor=scale, boarder_pad_size=3, channel_num=31)
        img_patches = img_splitter.patchify(img, img_pad=0)
        with torch.no_grad():
            out = [up_module(i) for i in img_patches]
        img_upscale = img_splitter.unpatchify(out)
    """
    # key points:
    # Boarder padding and over-lapping img splitting to avoid the instability of edge value
    # Thanks Waifu2x's autorh nagadomi for suggestions (https://github.com/nagadomi/waifu2x/issues/238)

    def __init__(self, seg_size=48, scale_factor=2, channel_num=3, boarder_pad_size=3, pad_mode='reflection', padding_value=0):
        self.seg_size = seg_size
        self.scale_factor = scale_factor
        self.pad_size = boarder_pad_size
        self.height = 0
        self.width = 0
        self.channel_num = channel_num
        self.pad_mode = pad_mode
        self.padding_value = padding_value
        self.init_padder()

    def init_padder(self):
        """ Initialize the padder.
        """
        self.padder = None
        assert self.pad_mode in ('reflection', 'replication', 'zero', 'const')
        if self.pad_mode is 'reflection':
            self.padder = nn.ReflectionPad2d(self.pad_size)
        elif self.pad_mode is 'replication':
            self.padder = nn.ReplicationPad2d(self.pad_size)
        elif self.pad_mode is 'zero':
            self.padder = nn.ZeroPad2d(self.pad_size)
        else:
            self.padder = nn.ConstantPad2d(self.pad_size, self.padding_value)

    def patchify(self, img):
        """ Patchify an image. Split an image into patches.
        
        Args:
            img: The input image which has the shape of (H, W, C):np.ndarray.
                or, (B, C, H, W): torch.Tensor
        """
        if not isinstance(img, torch.Tensor):
            # Resize image and convert them into tensor
            img_tensor = to_tensor(img).unsqueeze(0)
        else:
            img_tensor = img
        img_tensor = self.padder(img_tensor)
        _, _, height, width = img_tensor.size()
        self.height = height
        self.width = width

        patch_box = []
        # Avoid the residual part is smaller than the padded size
        if height % self.seg_size < self.pad_size or width % self.seg_size < self.pad_size:
            self.seg_size += self.scale_factor * self.pad_size

        # Split image into over-lapping pieces
        for i in range(self.pad_size, height, self.seg_size):
            for j in range(self.pad_size, width, self.seg_size):
                part = img_tensor[:, :,
                                  (i - self.pad_size):min(i + self.pad_size + self.seg_size, height),
                                  (j - self.pad_size):min(j + self.pad_size + self.seg_size, width)]
                patch_box.append(part)
        return patch_box

    def unpatchify(self, list_img_tensor):
        """ Unpatchify an image. Merge parts together.

        Args:
            list_img_tensor: A list of upsampled patches each of which has the shape of
                (B, C, H*s, W*s), where B means batch, here it should be 1, C means the
                channel number, s means the upsampling scale.
        """
        out = torch.zeros((1, self.channel_num, self.height *
                           self.scale_factor, self.width * self.scale_factor))
        img_tensors = copy.copy(list_img_tensor)
        rem = self.pad_size * self.scale_factor

        pad_size = self.scale_factor * self.pad_size
        seg_size = self.scale_factor * self.seg_size
        height = self.scale_factor * self.height
        width = self.scale_factor * self.width
        for i in range(pad_size, height, seg_size):
            for j in range(pad_size, width, seg_size):
                part = img_tensors.pop(0)
                part = part[:, :, rem:-rem, rem:-rem]
                if len(part.size()) > 3:
                    _, _, p_h, p_w = part.size()
                    out[:, :, i:i + p_h, j:j + p_w] = part
        out = out[:, :, rem:-rem, rem:-rem]
        return out
