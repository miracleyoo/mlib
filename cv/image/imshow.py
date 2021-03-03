import numpy as np
import importlib
# import matplotlib.pyplot as plt

__all__ = ["imshow", "batch_show"]

def imshow(img, bands=(3,2,1), multiply=1, scale=1, method='plt', cmap='gray', ret=False, show=True):
    """ Show an image with various common type, bits.
        This function support numpy.ndarry, torch.Tensor as input.
        The channel dimension can be the first or last.
        If it has more than 3 dimension, will take the first element
        for the first several dimensions.
    Args:
        img: Input image. numpy.ndarry and torch.Tensor type supported.
        bands: If the channel dimension has more than 3 bands, use this tuple to select.
        multiply: Multiply a certain number on the whole image. Used when the image is dark.
        scale: A size scale factor. When the input image's size is much too small/big, use this.
        method: The plot method. Two choices: ('plt'|'pil') For some reason, VS Code Jupyter 
                may not work using matplotlib.pyplot, so you can use 'pil' instead. 
        cmap: When the image only has two dimension, or only select one band, the cmap used by
            matplotlib.pyplot. Default is gray.
        ret: Whether return the processed input image.
    """
    if not show:
        ret = True
    res = None

    if method not in ('plt', 'pil'):
        print("Please choose \'plt\' or \'pil\' for method.")
        return None
    
    def show_img(image, method, cmap):
        if method.lower() == 'plt':
            plt = importlib.import_module('matplotlib.pyplot')
            plt.figure()
            if len(image.shape) == 2:
                plt.imshow(image, cmap=cmap) # Default: "viridis"
            else:
                plt.imshow(image)
            return None
        elif method.lower() == 'pil':
            Image=importlib.import_module('PIL.Image')
            return Image.fromarray(image)


    def rescale(image, multiply, scale):
        imax = image.max()
        if imax <= 1:
            image = (image*255)
        elif imax > 255:
            image = ((image+1)/256)
        else:
            image = image
        image = (image*multiply).astype(np.uint8)
        image = np.clip(image, 0, 255)
        if scale <= 0:
            raise ValueError("scale should be bigger than 0!")
        elif scale != 1:
            cv2 = importlib.import_module('cv2')
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return image

    if not isinstance(img, np.ndarray):
        try:
            torch = importlib.import_module('torch')
            if isinstance(img, torch.Tensor):
                img = img.cpu().detach().numpy()
        except:
            raise ImportError("Pytorch not installed!")

    while len(img.shape) > 3:
        img = img[0]

    shp = img.shape

    if len(shp) < 2:
        print("Invalid Input Type!")
    elif len(shp) == 2:
        img = rescale(img, multiply, scale)
        if show:
            res = show_img(img, method, cmap)
    else:
        if shp[0] < shp[-1]:
            img = img.transpose(1,2,0)
            shp = img.shape
        img = rescale(img, multiply, scale)
        if shp[-1] > 3:
            img = img[:,:,bands]
        if show:
            res = show_img(img, method, cmap)
    if res is not None:
        return res
    if ret:
        return img


def batch_show(imgs, sub_titles=None, title=None, row_labels=None, col_labels=None):
    """ Show images. 
    Args:
        imgs: Supposed to be an 2-d list or tuple. each element is an image in numpy.ndarray format.
        sub_titles: Titles of each subplot.
        title: The image overall title.
    """
    if not (isinstance(imgs[0], list) or isinstance(imgs[0], tuple)):
        imgs = [imgs]
    rows = len(imgs)
    cols = max([len(i) for i in imgs])

    plt = importlib.import_module('matplotlib.pyplot')
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows), sharey=True)
    if rows == 1:
        axs = [axs]
    if cols == 1:
        axs = [[i] for i in axs]
    axs = np.array(axs)
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            img = imgs[i][j]
            if sub_titles is not None and len(sub_titles)>i and len(sub_titles[i])>j:
                sub_title = sub_titles[i][j]
            else:
                sub_title = ''
            axs[i, j].imshow(img)
            axs[i, j].set(xticks=[], yticks=[])
            if row_labels is not None and len(row_labels)>i:
                axs[i, j].set_ylabel(row_labels[i], fontsize=20)
            if col_labels is not None and len(col_labels)>j:
                axs[i, j].set_xlabel(col_labels[j], fontsize=20)
            if sub_title != '':
                axs[i, j].set_title(sub_title, fontsize=20, y=-0.15)
    
    for ax in axs.flat:
        ax.label_outer()
    
    if title is not None:
        fig.suptitle(title,fontsize=30)
    plt.tight_layout()
