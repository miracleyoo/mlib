import numpy as np
import importlib

def imshow(img, bands=(3,2,1), multiply=1, scale=1, method='plt', cmap='gray', ret=False):
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
            image = (image*255).astype(np.uint8)
        elif imax > 255:
            image = ((image+1)/256).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        image = image*multiply
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
        res = show_img(img, method, cmap)
    else:
        if shp[0] < shp[-1]:
            img = img.transpose(1,2,0)
            shp = img.shape
        img = rescale(img, multiply, scale)
        if shp[-1] > 3:
            img = img[:,:,bands]
        res = show_img(img, method, cmap)
    if res is not None:
        return res
    if ret:
        return img