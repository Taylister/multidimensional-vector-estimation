import random
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

def gen_input_mask(
        shape, hole_size, hole_area=None, max_holes=1):
    """
    * inputs:
        - shape (sequence, required):
                Shape of a mask tensor to be generated.
                A sequence of length 4 (N, C, H, W) is assumed.
        - hole_size (sequence or int, required):
                Size of holes created in a mask.
                If a sequence of length 4 is provided,
                holes of size (W, H) = (
                    hole_size[0][0] <= hole_size[0][1],
                    hole_size[1][0] <= hole_size[1][1],
                ) are generated.
                All the pixel values within holes are filled with 1.0.
        - hole_area (sequence, optional):
                This argument constraints the area where holes are generated.
                hole_area[0] is the left corner (X, Y) of the area,
                while hole_area[1] is its width and height (W, H).
                This area is used as the input region of Local discriminator.
                The default value is None.
        - max_holes (int, optional):
                This argument specifies how many holes are generated.
                The number of holes is randomly chosen from [1, max_holes].
                The default value is 1.
    * returns:
            A mask tensor of shape [N, C, H, W] with holes.
            All the pixel values within holes are filled with 1.0,
            while the other pixel values are zeros.
    """
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    for i in range(bsize):
        n_holes = random.choice(list(range(1, max_holes+1)))
        for _ in range(n_holes):
            # choose patch width
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_w = random.randint(hole_size[0][0], hole_size[0][1])
            else:
                hole_w = hole_size[0]

            # choose patch height
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_h = random.randint(hole_size[1][0], hole_size[1][1])
            else:
                hole_h = hole_size[1]

            # choose offset upper-left coordinate
            if hole_area:
                harea_xmin, harea_ymin = hole_area[0]
                harea_w, harea_h = hole_area[1]
                offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
                offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)
            else:
                offset_x = random.randint(0, mask_w - hole_w)
                offset_y = random.randint(0, mask_h - hole_h)
            mask[i, :, offset_y : offset_y + hole_h, offset_x : offset_x + hole_w] = 1.0
    return mask

def label2img(shape,labels):
    #make black image, shape(N, )
    imgs = torch.zeros(shape)
    #print("img_shape:{}".format(imgs.shape))

    for i, one_hot in enumerate(labels):
        #print(one_hot)
        index = torch.nonzero(one_hot)[0][1]
        imgs[i, index, :, :,] = 1.0
    
    return imgs


def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def extract_mask_region(x_mask):
    """
    * inputs:
        - x_mask (numpy ndarray(transformed from torch.Tensor), required)
                    A torch tensor of shape (N,C,H,W) is assumed.
    * return 
        - A sequence used for the input argument ---
    """
    rois = []
    for ele in x_mask:
        gray = np.clip(np.transpose(ele,[1,2,0]) * 255, 0, 255).astype(np.uint8)
        coords = cv2.findNonZero(gray) # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        #x = x - w
        #y = y - h
        roi = ((x,y),(w,h))
        #print("roi:{}".format(roi))
        rois.append(roi)
    
    return rois

def crop(x, area, size):
    """
    * inputs:
        - x ( numpy ndarray , required (transformed from torch.Tensor)) 
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                shape: (N,((X,Y),(W,H))
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
        - size (sequence, required)
                A sequence of length is assumed.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    crop_imgs = []

    resize_W,resize_H = size

    for i in range(len(x)):
        xmin, ymin = area[i][0]
        w, h = area[i][1]
        #print("xmin:{},ymin:{},w:{},h:{}".format(xmin,ymin,w,h))
        # shape (C, H, W)
        crop_img = x[i][:, ymin : ymin + h, xmin : xmin + w]

        dst = np.transpose(crop_img,[1,2,0])
        dst = cv2.resize(dst,(resize_W,resize_H)) # (W,H)
        dst = torch.from_numpy(dst.astype(np.float32)).clone()
        dst = dst.permute(2,0,1)

        crop_imgs.append(dst)

    # convert from py list to tensor
    return torch.cat(crop_imgs).reshape(len(crop_imgs),*crop_imgs[0].shape)

#def crop(x, area):
#    """
#    * inputs:
#        - x (torch.Tensor, required)
#                A torch tensor of shape (N, C, H, W) is assumed.
#        - area (sequence, required)
#                A sequence of length 2 ((X, Y), (W, H)) is assumed.
#                sequence[0] (X, Y) is the left corner of an area to be cropped.
#                sequence[1] (W, H) is its width and height.
#    * returns:
#            A torch tensor of shape (N, C, H, W) cropped in the specified area.
#    """
#    xmin, ymin = area[0]
#    w, h = area[1]
#    return x[:, :, ymin : ymin + h, xmin : xmin + w]


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch_image = []
    batch_label = []
    batch_vector = []

    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        image,label,vector = dataset[index]
        image = torch.unsqueeze(image, dim=0)
        label = torch.unsqueeze(label, dim=0)
        vector = torch.unsqueeze(vector, dim=0)
        
        batch_image.append(image)
        batch_label.append(label)
        batch_vector.append(vector)
        
    return torch.cat(batch_image, dim=0),torch.cat(batch_label, dim=0),torch.cat(batch_vector, dim=0)
class Translater:
    def __init__(self,num_class):

        self.num_class = num_class

        self.alpha_dict = {}
        for i,c in enumerate(range(ord('A'),ord('Z')+1)):
            self.alpha_dict[chr(c)] = i
        
        offset = len(self.alpha_dict)

        for i,c in enumerate(range(ord('a'),ord('z')+1)):
            self.alpha_dict[chr(c)] = i + offset


    def chr2num(self,label): 
        num_label = torch.tensor([self.alpha_dict[label]])
        return torch.nn.functional.one_hot(num_label,num_classes=self.num_class)

    def num2chr(self,num):
        char = list(self.alpha_dict.keys())[list(self.alpha_dict.values())[num]]
        return char
    

