def cons_box(box, w, h):
    """Check and make sure the box is not exceeding the boundary.
    Args:
        box: Box (Left, Top, Right, Bottom) location list.
        w: The width of the image which the box lives on.
        h: The height of the image which the box lives on.
    Returns:
        The constrained box (Left, Top, Right, Bottom) location list.
    """
    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(w, box[2])
    box[3] = min(h, box[3])
    return box
    
def bounding_box_rec2square(img, box, scale=1):
    # In case the box is out of image
    box = cons_box(box, img.shape[1], img.shape[0])
    big_box = []
    
    # 
    width = abs(box[0]-box[2])
    height = abs(box[1]-box[3])
    center = (int(box[0]+width/2), int(box[1]+height/2))
    square_len = int((width+height)/2)
    
    big_box.append(center[0] - scale*square_len//2)
    big_box.append(center[1] - scale*square_len//2)
    big_box.append(center[0] + scale*square_len//2)
    big_box.append(center[1] + scale*square_len//2)
    
    big_box = cons_box(big_box, img.shape[1], img.shape[0])
    return big_box

