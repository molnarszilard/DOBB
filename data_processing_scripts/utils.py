"""
This file contains many useful function. They are either used multiple times, or general enough to be written here.
The purpose is that if a function is used in multiple places, it does not have to be implemented in each script separately, but only once here.
Furthermore, if a bug is found in the function, it has to be corrected only once, and not at each occurence.
In a few cases, function repeat over various scripts, this usually means, that each one of them has a minor but unique differences,
 and it is a TODO to create a general version, that works for each script.
Some function, and their features might not be fully tested.
"""

import numpy as np
import math
import cv2
from skimage.exposure import match_histograms
import random
from scipy.spatial.transform import Rotation as Rot



def augment_ellipse(Cx,Cy,lambda1,lambda2,angle,obb=None):
    '''
    Add random but minor noise to the ellipse
    '''
    if obb is not None:
        angle+=math.radians(obb%(45))
    angle+=random.random()*30-15
    lambda1+=random.random()*lambda1*0.1-lambda1*0.05
    lambda2+=random.random()*lambda2*0.1-lambda2*0.05
    Cx+=random.random()*20-10
    Cy+=random.random()*20-10
    return int(Cx),int(Cy),int(lambda1),int(lambda2),angle

def get_ellipse_params(points,weights):
    '''
    Calculate the ellipse parameters using covariance matrix given a set of points
    From https://cookierobotics.com/007/
    '''
    points = np.asarray(points)
    points=points.T
    cov_matrix = np.cov(points,aweights=weights)
    a=cov_matrix[0,0]
    b=cov_matrix[0,1]
    c=cov_matrix[1,1]
    lambda1=(a+c)/2+np.sqrt(((a-c)/2)**2+b**2)
    lambda2=(a+c)/2-np.sqrt(((a-c)/2)**2+b**2)
    if lambda1<=0 or lambda2<=0:
        return None, None, None, None, None
    if b==0 and a>=c:
        theta = 0
    elif b==0 and a<c:
        theta = np.pi/2
    else:
        theta = np.arctan2(lambda1-a,b)
    center_coordinates = np.average(points,axis=1)
    if theta>math.pi/2:
        theta = theta-math.pi
    if theta<-math.pi/2:
        theta = theta+math.pi
    return int(center_coordinates[0]),int(center_coordinates[1]),int(np.sqrt(lambda1)), int(np.sqrt(lambda2)),theta

def get_ellipse_params_mask(mask, mask_orig, use_weights,obb=None, randomizer=False):
    '''
    Calculate the ellipse parameters using covariance matrix given a mask
    '''
    inner_points = list_points(mask,None)
    if use_weights:
        weights = list_points(mask,mask_orig)
        if len(weights)>1:
            weights/=weights.mean()
        else:
            weights=None
    else:
        weights=None
    if len(inner_points)>1:
        Cx,Cy,lambda1,lambda2,angle = get_ellipse_params(inner_points,weights)
        if randomizer:
            Cx,Cy,lambda1,lambda2,angle = augment_ellipse(Cx,Cy,lambda1,lambda2,angle,obb)
        return Cx,Cy,lambda1,lambda2,angle
    else:
        return None, None, None, None, None

def list_points(matrix,value):
    '''
    Given a matrix/mask return the coordinates of the points
    '''
    lines_points = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j]>0:
                if value is None:
                    lines_points.append([j,i])
                else:
                    lines_points.append(float(value[i,j])/255)
    return np.asarray(lines_points)

def get_green_mask(img):
    '''
    Get the mask of green pixels
    '''
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = Lab[:,:,1]
    green_mask = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    green_mask[get_mask(img) == 0] = 0
    return green_mask

def get_green(img):
    '''
    Mask non-green pixels from an image
    '''
    green_mask = get_green_mask(img)
    green_img = np.zeros_like(img, np.uint8)
    green_img[np.where(green_mask != 0)] = img[np.where(green_mask != 0)]
    return green_img

def get_mask(img):
    '''
    Get the mask of almost black or almost white pixels
    '''
    lower_range = np.array([235, 235, 235])
    upper_range = np.array([256, 256, 256])
    maskwhite = cv2.inRange(img, lower_range, upper_range)
    # set all other areas to zero except where mask area 
    ## masking black
    lower_range = np.array([0, 0, 0])
    upper_range = np.array([20, 20, 20]) 
    maskblack = cv2.inRange(img, lower_range, upper_range)
    # set all other areas to zero except where mask area 
    mask = np.ones((img.shape[0],img.shape[1])).astype(np.uint8)*255
    mask[maskblack != 0] = 0
    mask[maskwhite != 0] = 0

    return mask

def delete_edges(img,width):
    '''
    Clear almost black or almost white pixels
    '''
    img2 = img.copy()
    mask = get_mask(img)
    _,thresh_edge = cv2.threshold(mask, 127, 255, 0)
    contours_edge, _ = cv2.findContours(thresh_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cont_mask = np.zeros_like(mask)
    cv2.drawContours(cont_mask, contours_edge, -1, 255,width)
    mask[cont_mask>0]=0
    img2[mask == 0] = [0, 0, 0]
    return img2

def blur_edge(img,mask):
    '''
    Blur the image inside a given mask
    '''
    _,thresh_edge = cv2.threshold(mask, 127, 255, 0)
    contours_edge, _ = cv2.findContours(thresh_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cont_mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(cont_mask, contours_edge, -1, (255,255,255),1)
    blurred = cv2.GaussianBlur(img, (31, 31), 2)
    # blurred = cv2.bilateralFilter(sub_img, 11, 61, 39)
    img = np.where(cont_mask==np.array([255, 255, 255]), blurred, img)
    return img

def filter_bw(img):
    '''
    clean image from almost black/white pixels
    '''
    mask = get_mask(img)
    img[mask == 0] = [0, 0, 0]
    return img, mask

def hm_match(img,target,filter_green):
    '''
    Match two images given their color histograms.
    This function can take into account to check only the green pixels for histogram matching
    '''
    image = delete_edges(img,2)
    mask = get_mask(image)
    output = image.copy()
    if filter_green:        
        green_img = get_green(output)
        green_target = get_green(target)
        green_img_mask = get_green_mask(output)
        green_img = hm_match_cs(green_img,green_target,"lab",edge_delete=False,get_average=False)
        output[np.where(green_img_mask != 0)] = green_img[np.where(green_img_mask != 0)]
    else:
        green_img = hm_match_cs(output,output,"lab",edge_delete=False,get_average=False) 
    output[mask <127] = [0, 0, 0]
    return output

def hm_match_cs(img,target,cs,edge_delete=False,get_average=0):
    '''
    Match two images given their color histograms.
    Different color spaces can be used.
    '''
    if edge_delete:
        img = delete_edges(img,2)
    mask = get_mask(img)
    if cs=="rgb" or cs=="bgr":
        output = match_histograms(img, target,channel_axis=-1)
    if cs=="lab":
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        lab_output = match_histograms(lab_img, lab_target,channel_axis=-1) 
        output = cv2.cvtColor(lab_output, cv2.COLOR_LAB2BGR)
    if cs=="luv":
        luv_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        luv_target = cv2.cvtColor(target, cv2.COLOR_BGR2LUV)
        luv_output = match_histograms(luv_img, luv_target,channel_axis=-1) 
        output = cv2.cvtColor(luv_output, cv2.COLOR_LUV2BGR)
    if cs=="hls":
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hls_target = cv2.cvtColor(target, cv2.COLOR_BGR2HLS)
        hls_output = match_histograms(hls_img, hls_target,channel_axis=-1) 
        output = cv2.cvtColor(hls_output, cv2.COLOR_HLS2BGR)
    if cs=="hsv":
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        hsv_output = match_histograms(hsv_img, hsv_target,channel_axis=-1) 
        output = cv2.cvtColor(hsv_output, cv2.COLOR_HSV2BGR)
    if cs=="ycrcb":
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb_target = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
        ycrcb_output = match_histograms(ycrcb_img, ycrcb_target,channel_axis=-1) 
        output = cv2.cvtColor(ycrcb_output, cv2.COLOR_YCrCb2BGR)

    output[mask<127] = [0, 0, 0]
    if get_average>0:
        new_spot_diff = (img.astype(np.int16)-output.astype(np.int16))/2
        new_spot_diff[mask<127] = [0, 0, 0]
        new_spot = output+new_spot_diff*get_average
        new_spot[mask<127] = [0, 0, 0]
        output=new_spot    
    
    return output

def get_distance(Ax,Ay,Bx,By):
    '''
    Get distance between two 2Dimensional points
    '''
    return math.sqrt((Bx-Ax)**2+(By-Ay)**2)

def get_color(cls,max=10):
    '''
    Given the name of a color, or the number of a class, return its RGB value
    Possible colors: orange, fuchsia, blue, cyan, purple, pink, yellow, magenta, white, teal black, green, red, yolored
    Possible class numbers: [0,12]
    Invalid returns black
    '''
    if isinstance(cls,str) and cls.isnumeric():
        cls = int(float(cls))
    if isinstance(cls,int):
        if cls%max==0: #plane
            return (0,128,255) #orange
        elif cls%max==1: #ship
            return (255,0,255) #fuchsia
        elif cls%max==2: # large vehicle
            return (255,0,0) #blue
        elif cls%max==3: # small vehicle
            return (255,255,0) #cyan
        elif cls%max==4: # helicopter
            return (255,0,128) #purple
        elif cls%max==5: # 
            return (255,153,255) # pink
        elif cls%max==6: 
            return (0,255,255) #yellow
        elif cls%max==7: 
            return (128,0,255) #magenta
        elif cls%max==8: 
            return (255,255,255) #white
        elif cls%max==9: 
            return (128,255,0) #teal
        elif cls%max==10:
            return (0,0,0) #black
        elif cls%max==11:
            return (0,255,0) #green
        elif cls%max==12:
            return (0,0,255) #red
        else:
            # print("Try another class.")
            return (0,0,0) #black
    else:
        if cls=='orange': #plane
            return (0,128,255) #orange
        elif cls=='fuchsia': #ship
            return (255,0,255) #fuchsia
        elif cls=='blue': # large vehicle
            return (255,0,0) #blue
        elif cls=='cyan': # small vehicle
            return (255,255,0) #cyan
        elif cls=='purple': # helicopter
            return (255,0,128) #purple
        elif cls=='pink':
            return (255,153,255) #pink
        elif cls=='yellow': # special 2
            return (0,255,255) #yellow
        elif cls=='magenta':
            return (128,0,255) #magenta
        elif cls=='white': # special 4
            return (255,255,255) #white
        elif cls=='teal': # container crane
            return (128,255,0) #teal
        elif cls=='black':
            return (0,0,0) #black
        elif cls=='green': # special 3
            return (0,255,0) #green        
        elif cls=='red': # special 1
            return (0,0,255) #red        
        elif cls=='yolored':
            return (59,57,253) #yolo detection red        
        else:
            print(cls)
            print("Try another class.")
            return (0,0,0) #black
        
def xywhr2xyxyxyxy(rboxes):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians.

    Args:
        rboxes (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    ctr = rboxes[:2]
    w, h, angle = rboxes[2:]
    cos_value, sin_value = math.cos(angle), math.sin(angle)
    vec1 = np.array([w / 2 * cos_value, w / 2 * sin_value])
    vec2 = np.array([-h / 2 * sin_value, h / 2 * cos_value])
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    corners = np.stack([pt1, pt2, pt3, pt4], axis=-2)
    return corners

def xyxyxyxy2xywhr(corners):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation]. Rotation values are
    expected in degrees from 0 to 90.

    Args:
        corners (numpy.ndarray | torch.Tensor): Input corners of shape (n, 8).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
    """
    points = corners
    points = points.reshape( -1, 2)
    (x, y), (w, h), angle = cv2.minAreaRect(points)
    rboxes = (
        np.asarray([x, y, w, h, angle], dtype=points.dtype)
    )  # rboxes
    return rboxes


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs. From https://github.com/ultralytics/ultralytics

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    a, b = boxes[2:4]**2 / 12
    c = boxes[4:]
    # gbbs = np.array((boxes[2:4]**2 / 12, boxes[4:]))
    # a, b, c = gbbs.split(1, dim=-1)
    cos = math.cos(c)
    sin = math.sin(c)
    cos2 = cos**2
    sin2 = sin**2
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    From https://github.com/ultralytics/ultralytics
    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (N, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, ) representing obb similarities.
    """
    x1 = obb1[0]
    y1 = obb1[1]
    x2 = obb2[0]
    y2 = obb2[1]
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2)**2 + (b1 + b2) * (x1 - x2)**2) / ((a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps)) * 0.5
    t3 = math.log(
        ((a1 + a2) * (b1 + b2) - (c1 + c2)**2)
        / (4 * math.sqrt((a1 * b1 - c1**2).clip(0) * (a2 * b2 - c2**2).clip(0)) + eps)
        + eps
    ) * 0.5
    bd = (t1 + t2 + t3).clip(eps, 100.0)
    hd = math.sqrt(1.0 - math.exp(-bd) + eps)
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[2:4]
        w2, h2 = obb2[2:4]
        v = (4 / math.pi**2) * (math.atan(w2 / h2) - math.atan(w1 / h1))**2
        alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou

def get_center(elements,W=1,H=1,integer=False):
    '''
    Given 4 corner points of a 2D bounding box, return the coordinates of its center
    '''
    x1 = float(elements[1])*W
    y1 = float(elements[2])*H
    x2 = float(elements[3])*W
    y2 = float(elements[4])*H
    x3 = float(elements[5])*W
    y3 = float(elements[6])*H
    x4 = float(elements[7])*W
    y4 = float(elements[8])*H
    cx = (x1+x2+x3+x4)/4
    cy = (y1+y2+y3+y4)/4
    if integer:
        return int(cx),int(cy)
    else:
        return cx,cy

def dptFromAngle(cx,cy,w,h,phi,alpha):
        """Calculating the direction point from the bounding ellipse and the angle to the direction point
            Inspired from 
            https://math.stackexchange.com/a/22067
            and
            https://math.stackexchange.com/a/4517941
        Inputs:
            bbox (math.Tensor): (bs,N_dpts_per_object,5) - cx,cy,w,h,angle (angle - phi), the coordinates of the bounding ellipse
            alpha (math.Tensor): (bs,N_dpts_per_object,1) - alpha angle of the direction point
        Output:
            coordinates of the direction point
        """

        a=w/2
        b=h/2
        theta = alpha-phi
        if theta>math.pi:
             theta-=math.pi*2
        tan_th = math.tan(theta)
        x0 = a*b/(math.sqrt((b**2)+(a**2)*(tan_th**2)))
        if abs(theta)>=math.pi/2:
            x0*=-1
        y0 = x0*tan_th
        x = x0*math.cos(phi)-y0*math.sin(phi)
        y = x0*math.sin(phi)+y0*math.cos(phi)
        return cx+x, cy+y

def order_elements(elementsR,elementsP):
    '''
    This function tries to order the 4 corner points of elementsP (prediction)
      to match the order of points of elemntsR (reference)
    '''
    xR1 = elementsR[1]
    yR1 = elementsR[2]
    xR2 = elementsR[3]
    yR2 = elementsR[4]
    xR3 = elementsR[5]
    yR3 = elementsR[6]

    xP1 = elementsP[1]
    yP1 = elementsP[2]
    xP2 = elementsP[3]
    yP2 = elementsP[4]
    xP3 = elementsP[5]
    yP3 = elementsP[6]
    xP4 = elementsP[7]
    yP4 = elementsP[8]
    x1 = elementsP[1]
    y1 = elementsP[2]
    x2 = elementsP[3]
    y2 = elementsP[4]
    x3 = elementsP[5]
    y3 = elementsP[6]
    x4 = elementsP[7]
    y4 = elementsP[8]
    diff11 = get_distance(xR1,yR1,xP1,yP1)
    diff12 = get_distance(xR1,yR1,xP2,yP2)
    diff13 = get_distance(xR1,yR1,xP3,yP3)
    diff14 = get_distance(xR1,yR1,xP4,yP4)
    if diff12==min(diff11,diff12,diff13,diff14):
        tempX,tempY = x1,y1
        x1,y1 = x2,y2
        x2,y2 = tempX,tempY
        tempX,tempY = x3,y3
        x3,y3 = x4,y4
        x4,y4 = tempX,tempY
    if diff13==min(diff11,diff12,diff13,diff14):
        tempX,tempY = x1,y1
        x1,y1 = x3,y3
        x3,y3 = tempX,tempY
    if diff14==min(diff11,diff12,diff13,diff14):
        tempX,tempY = x1,y1
        x1,y1 = x4,y4
        x4,y4 = tempX,tempY
        tempX,tempY = x3,y3
        x3,y3 = x2,y2
        x2,y2 = tempX,tempY
    elementsP[1:9] = x1,y1,x2,y2,x3,y3,x4,y4
    xP1 = elementsP[1]
    yP1 = elementsP[2]
    xP2 = elementsP[3]
    yP2 = elementsP[4]
    xP3 = elementsP[5]
    yP3 = elementsP[6]
    xP4 = elementsP[7]
    yP4 = elementsP[8]
    x1 = elementsP[1]
    y1 = elementsP[2]
    x2 = elementsP[3]
    y2 = elementsP[4]
    x3 = elementsP[5]
    y3 = elementsP[6]
    x4 = elementsP[7]
    y4 = elementsP[8]
    diff22 = get_distance(xR2,yR2,xP2,yP2)
    diff23 = get_distance(xR2,yR2,xP3,yP3)
    diff24 = get_distance(xR2,yR2,xP4,yP4)
    if diff23==min(diff22,diff23,diff24):
        tempX,tempY = x2,y2
        x2,y2 = x3,y3
        x3,y3 = tempX,tempY
        tempX,tempY = x1,y1
        x1,y1 = x4,y4
        x4,y4 = tempX,tempY
    if diff24==min(diff22,diff23,diff24):
        tempX,tempY = x2,y2
        x2,y2 = x4,y4
        x4,y4 = tempX,tempY
    elementsP[1:9] = x1,y1,x2,y2,x3,y3,x4,y4
    xP1 = elementsP[1]
    yP1 = elementsP[2]
    xP2 = elementsP[3]
    yP2 = elementsP[4]
    xP3 = elementsP[5]
    yP3 = elementsP[6]
    xP4 = elementsP[7]
    yP4 = elementsP[8]
    x1 = elementsP[1]
    y1 = elementsP[2]
    x2 = elementsP[3]
    y2 = elementsP[4]
    x3 = elementsP[5]
    y3 = elementsP[6]
    x4 = elementsP[7]
    y4 = elementsP[8]
    diff33 = get_distance(xR3,yR3,xP3,yP3)
    diff34 = get_distance(xR3,yR3,xP4,yP4)
    if diff34==min(diff33,diff34):
        tempX,tempY = x3,y3
        x3,y3 = x4,y4
        x4,y4 = tempX,tempY
        tempX,tempY = x1,y1
        x1,y1 = x2,y2
        x2,y2 = tempX,tempY
    elementsP[1:9] = x1,y1,x2,y2,x3,y3,x4,y4
    return elementsP

def draw_rectangle(image,labels,colors=['white','white','white','white','red'],thickness=2,plot_kp=None,norm=False,dir_type=0):
    '''
    Plot rectangles onto an image, with a possible keypoint as well
    Inputs:
        image: 2D RGB image
        labels: minimum (N,9) or (N,11) required depending on if there is a keypoint -- other sizes are also possible
        colors: String array of the colors of the four edges, and the keypoint. Ideally its dimension is 4 or 5 colors, but 1 and 2 colors are also possible
        thickness: thickness of the lines
        plot_kp: None - does not print anything, "kp" - draw a circle at the keypoint, "arrow" - draws an arrow from the BB center to the keypoint
        norm: are the labels normalized? if they are, they will be multiplied by the image sizes
        dir_type: (0 - xy coordinates, 1 - cos+sin, 2 - [0:2pi]])
    '''
    image_new = image.copy()
    if norm:
        H,W,_ = image_new.shape
    else:
        H,W=1,1
    if len(np.asarray(labels).shape)<2:
        labels=[labels]
    for elements in labels:
        elements_float=elements.copy()
        cls = int(elements_float[0])
        if norm:
            x1 = int(float(elements_float[1])*W)
            y1 = int(float(elements_float[2])*H)
            x2 = int(float(elements_float[3])*W)
            y2 = int(float(elements_float[4])*H)
            x3 = int(float(elements_float[5])*W)
            y3 = int(float(elements_float[6])*H)
            x4 = int(float(elements_float[7])*W)
            y4 = int(float(elements_float[8])*H)
            elements_float[1:9]=[x1,y1,x2,y2,x3,y3,x4,y4]
        elements_float[1:9] = np.array(elements_float[1:9]).astype(np.int64)
        if len(elements_float)>9:
            if dir_type==1:
                rect = np.array(elements_float[1:9]).astype(np.int64)
                rect_xywhr = cv2.minAreaRect(rect.reshape(4, 2))
                rect_xywhr = np.array((rect_xywhr[0][0],rect_xywhr[0][1],rect_xywhr[1][0],rect_xywhr[1][1],rect_xywhr[2]))
                if rect_xywhr[2]<rect_xywhr[3]:
                    rect_xywhr[2],rect_xywhr[3] = rect_xywhr[3],rect_xywhr[2]
                    rect_xywhr[4] -= 90
                cosD=float(elements_float[9])
                sinD=float(elements_float[10])
                angleD=math.atan2(sinD,cosD)
                px,py=dptFromAngle(rect_xywhr[0],rect_xywhr[1],rect_xywhr[2],rect_xywhr[3],math.radians(rect_xywhr[4]),angleD)
                elements_float[9] = int(px)
                elements_float[10] = int(py)
            else:
                if norm:
                    elements_float[9] = int(float(elements_float[9])*W)
                    elements_float[10] = int(float(elements_float[10])*H)
        x1 = int(elements_float[1])
        y1 = int(elements_float[2])
        x2 = int(elements_float[3])
        y2 = int(elements_float[4])
        x3 = int(elements_float[5])
        y3 = int(elements_float[6])
        x4 = int(elements_float[7])
        y4 = int(elements_float[8])

        if len(colors)==0:
            colors=['white','white','white','white','red']
        if len(colors)==1:
            colors=[colors[0],colors[0],colors[0],colors[0]]
        if len(colors)==2:
            colors=[colors[0],colors[0],colors[0],colors[0],colors[1]]
        if colors[0]=='cls':
            image_new = cv2.line(image_new, (x1,y1), (x2,y2),get_color(cls), thickness)
        else:
            image_new = cv2.line(image_new, (x1,y1), (x2,y2),get_color(colors[0]), thickness)
        if colors[1]=='cls':
            image_new = cv2.line(image_new, (x2,y2), (x3,y3),get_color(cls), thickness)
        else:
            image_new = cv2.line(image_new, (x2,y2), (x3,y3),get_color(colors[1]), thickness)
        if colors[2]=='cls':
            image_new = cv2.line(image_new, (x3,y3), (x4,y4),get_color(cls), thickness)
        else:
            image_new = cv2.line(image_new, (x3,y3), (x4,y4),get_color(colors[2]), thickness)
        if colors[3]=='cls':
            image_new = cv2.line(image_new, (x4,y4), (x1,y1),get_color(cls), thickness)
        else:
            image_new = cv2.line(image_new, (x4,y4), (x1,y1),get_color(colors[3]), thickness)
        if plot_kp is not None or plot_kp!="None":
            if len(elements_float)<10:
                # print("Warning, asking to plot keypoint, but no keypoint found! Using the middle point between the first to points")
                dpx=((x1+x2))/2
                dpy=((y1+y2))/2
            else:
                dpx = elements_float[9]
                dpy = elements_float[10]
            if len(colors)==4:
                colors=[colors[0],colors[1],colors[2],colors[3],'red']
            
            if plot_kp=="kp":
                image_new = cv2.circle(image_new, (int(dpx),int(dpy)), thickness*2, get_color(colors[4]), thickness)
            elif plot_kp=="arrow":
                cx,cy = get_center(elements_float,integer=True)
                image_new = cv2.arrowedLine(image_new, (int(cx),int(cy)), (int(dpx),int(dpy)), get_color(colors[4]), thickness)
            else:
                print("Possible keypoints are [kp, arrow]!")
            # image_new = cv2.circle(image_new, (int(x1),int(y1)), thickness*2, get_color('red'), thickness)
    return image_new

def draw_ellipse(image,labels,colors=['white','red'],thickness=2,plot_kp=None,norm=False,dir_type=0):
    '''
    Plot rectangles onto an image, with a possible keypoint as well
    Inputs:
        image: 2D RGB image
        labels: minimum (N,9) or (N,11) required depending on if there is a keypoint -- other sizes are also possible. Labels are represented with 4 corner points.
        colors: String array of the colors of the four edges, and the keypoint. Ideally its dimension is 4 or 5 colors, but 1 and 2 colors are also possible
        thickness: thickness of the lines
        plot_kp: None - does not print anything, "kp" - draw a circle at the keypoint, "arrow" - draws an arrow from the BB center to the keypoint
        norm: are the labels normalized? if they are, they will be multiplied by the image sizes
        dir_type: (0 - xy coordinates, 1 - cos+sin, 2 - [0:2pi]])
    TODO: will not work with multiple keypoints per label
    TODO: option to use highlights
    TODO: interchangable ellipse/4corner representation
    '''
    image_new = image.copy()
    if norm:
        H,W,_ = image_new.shape
    else:
        H,W=1,1
    if len(np.asarray(labels).shape)<2:
        labels=[labels]
    for elements in labels:
        elements_float=elements.copy()
        cls = int(elements_float[0])
        if len(colors)==0:
            colors=['white','red']
        if norm:
            x1 = int(float(elements_float[1])*W)
            y1 = int(float(elements_float[2])*H)
            x2 = int(float(elements_float[3])*W)
            y2 = int(float(elements_float[4])*H)
            x3 = int(float(elements_float[5])*W)
            y3 = int(float(elements_float[6])*H)
            x4 = int(float(elements_float[7])*W)
            y4 = int(float(elements_float[8])*H)
            elements_float[1:9]=[x1,y1,x2,y2,x3,y3,x4,y4]
        rect = np.array((elements_float[1:9]))
        rect_xywhr = cv2.minAreaRect(rect.reshape(4, 2))
        rect_xywhr = np.array((rect_xywhr[0][0],rect_xywhr[0][1],rect_xywhr[1][0],rect_xywhr[1][1],rect_xywhr[2]))
        if rect_xywhr[2]<rect_xywhr[3]:
            rect_xywhr[2],rect_xywhr[3] = rect_xywhr[3],rect_xywhr[2]
            rect_xywhr[4]+=90
        if len(elements_float)>9:
            if dir_type==1:
                cosD=float(elements_float[9])
                sinD=float(elements_float[10])
                angleD=math.atan2(sinD,cosD)
                px,py=dptFromAngle(rect_xywhr[0],rect_xywhr[1],rect_xywhr[2],rect_xywhr[3],math.radians(rect_xywhr[4]),angleD)
                elements_float[9] = int(px)
                elements_float[10] = int(py)
            else:
                if norm:
                    elements_float[9] = int(float(elements_float[9])*W)
                    elements_float[10] = int(float(elements_float[10])*H)
                else:
                    elements_float[9] = int(float(elements_float[9]))
                    elements_float[10] = int(float(elements_float[10]))
        
        if colors[0]=='cls':
            draw_color = get_color(cls)
        else:
            draw_color = get_color(colors[0])
        image_new = cv2.ellipse(image_new, (int(rect_xywhr[0]),int(rect_xywhr[1])), (int(rect_xywhr[2]/2),int(rect_xywhr[3]/2)), rect_xywhr[4], 0, 360, draw_color, thickness)
        
        if plot_kp is not None or plot_kp!="None":
            if len(elements_float)<10:
                print("Error, asking to plot keypoint, but no keypoint found! Using the middle point between the first to points")
                dpx=(float(elements_float[1])+float(elements_float[3]))/2
                dpy=(float(elements_float[2])+float(elements_float[4]))/2
            else:
                dpx = int(float(elements_float[9]))
                dpy = int(float(elements_float[10]))
            if len(colors)==1:
                colors=[colors[0],colors[1],colors[2],colors[3],'red']
            if plot_kp=="kp":
                image_new = cv2.circle(image_new, (int(dpx),int(dpy)), thickness*2, get_color(colors[1]), thickness)
            elif plot_kp=="arrow":
                cx,cy = get_center(elements_float,integer=True)
                image_new = cv2.arrowedLine(image_new, (int(cx),int(cy)), (int(dpx),int(dpy)), get_color(colors[1]), thickness)
            elif plot_kp is None:
                continue
            else:
                print("Possible keypoints are [kp, arrow]!")
    return image_new

def get_median(array):
    '''
    Return the median of an array
    '''
    array.sort()
    l = len(array)
    if l==1:
        return array[0]
    if l%2:
        return array[int(l/2)]
    a = array[int(l/2)-1]
    b = array[int(l/2)]
    return (a+b)/2

### Comparing two 3 dimensional vectors
### From https://stackoverflow.com/a/13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def calc_tblr(image,H,norm=False):
    height = image.shape[0]
    width = image.shape[1] 
    tl1=np.array([1,1,1])
    tl2 = H@tl1
    tl2/=tl2[2] # normalize vector
    tr1=np.array([width,1,1])
    tr2 = H@tr1
    tr2/=tr2[2] # normalize vector
    bl1=np.array([1,height,1])
    bl2 = H@bl1
    bl2/=bl2[2] # normalize vector    
    br1=np.array([width,height,1])
    br2 = H@br1
    br2/=br2[2] # normalize vector
    left = min(tl2[0],tr2[0],bl2[0],br2[0])
    right = max(tl2[0],tr2[0],bl2[0],br2[0])
    top = min(tl2[1],tr2[1],bl2[1],br2[1])
    bottom = max(tl2[1],tr2[1],bl2[1],br2[1])
    # print(top,bottom, left, right)
    if norm:
        top=round(top)
        bottom=round(bottom)
        left=round(left)
        right=round(right)
    return top,bottom, left, right


def calc_transformed_corners(image,H):
    height = image.shape[0]
    width = image.shape[1] 
    tl1=np.array([1,1,1])
    tl2 = H@tl1
    tl2/=tl2[2] # normalize vector
    tr1=np.array([width,1,1])
    tr2 = H@tr1
    tr2/=tr2[2] # normalize vector
    bl1=np.array([1,height,1])
    bl2 = H@bl1
    bl2/=bl2[2] # normalize vector    
    br1=np.array([width,height,1])
    br2 = H@br1
    br2/=br2[2] # normalize vector
    
    return tl2,tr2,bl2,br2

def Apply_Matrix_To_Image(matrix_to_apply, image_map,new_width,new_height):
    new_image_map = cv2.warpPerspective(image_map, matrix_to_apply, (new_width, new_height),flags=cv2.INTER_LINEAR)
    return new_image_map

def rev_Apply_Matrix_To_Image(matrix_to_apply, image_map,new_width,new_height):
    new_image_map = cv2.warpPerspective(image_map, np.linalg.inv(matrix_to_apply), (new_width, new_height),flags=cv2.INTER_LINEAR)
    return new_image_map

def check_center_outlier(A,width,height):
    if A[0]<0 or A[0]>=width:
        return True
    if A[1]<0 or A[1]>=height:
        return True
    return False

def norm_labels(A,width,height):
    x1,y1,x2,y2,x3,y3,x4,y4,dx,dy = A
    if x1<0:
        x1=0
    if x1>((width-1)/width):
        x1=((width-1)/width)
    if x2<0:
        x2=0
    if x2>((width-1)/width):
        x2=((width-1)/width)
    if x3<0:
        x3=0
    if x4>((width-1)/width):
        x4=((width-1)/width)
    if dx<0:
        dx=0
    if dx>((width-1)/width):
        dx=((width-1)/width)
    if y1<0:
        y1=0
    if y1>((height-1)/height):
        y1=((height-1)/height)
    if y2<0:
        y2=0
    if y2>((height-1)/height):
        y2=((height-1)/height)
    if y3<0:
        y3=0
    if y4>((height-1)/height):
        y4=((height-1)/height)
    if dy<0:
        dy=0
    if dy>((height-1)/height):
        dy=((height-1)/height)
    return [x1,y1,x2,y2,x3,y3,x4,y4,dx,dy]


def get_matches(ref,pred,threshold=0.5):
    IOUs = np.zeros((len(ref),len(pred)))
    for Ri in range(len(ref)):
        for Pi in range(len(pred)):
            if ref[Ri][0]==pred[Pi][0]:
                if threshold==0.001:
                    IOUs[Ri,Pi]=1
                else:
                    center = (pred[Pi][1],pred[Pi][2])
                    size = (pred[Pi][3],pred[Pi][4])
                    rect_pred_xywhr = (center,size,pred[Pi][5])
                    center = (ref[Ri][1],ref[Ri][2])
                    size = (ref[Ri][3],ref[Ri][4])
                    rect_ref_xywhr = (center,size,ref[Ri][5])
                    r1 = cv2.rotatedRectangleIntersection(rect_ref_xywhr, rect_pred_xywhr)
                    if r1[0] != 0:
                        rect_ref_xywhr_ar = np.array((rect_ref_xywhr[0][0],rect_ref_xywhr[0][1],rect_ref_xywhr[1][0],rect_ref_xywhr[1][1],rect_ref_xywhr[2]))
                        rect_pred_xywhr = np.array((rect_pred_xywhr[0][0],rect_pred_xywhr[0][1],rect_pred_xywhr[1][0],rect_pred_xywhr[1][1],rect_pred_xywhr[2]))
                        rect_ref_xywhr_ar[4] = math.radians(rect_ref_xywhr_ar[4])
                        rect_pred_xywhr[4] = math.radians(rect_pred_xywhr[4])
                        iou = probiou(rect_ref_xywhr_ar,rect_pred_xywhr)
                        IOUs[Ri,Pi]=iou
    matches = np.nonzero(IOUs >= threshold)  # IoU > threshold and classes match
    matches = np.array(matches).T
    if matches.shape[0]:
        if matches.shape[0] > 1:
            matches = matches[IOUs[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return matches, IOUs

def compose(axes, angle, center):
    axes = np.asarray(axes).astype(float)
    center = np.asarray(center).astype(float)
    C_star = np.diag((axes**2).tolist() + [-1.0])
    T_center = np.eye(3, dtype=float)
    T_center[:2, 2] = center
    Rw_e = Rot.from_euler("z", angle).as_matrix()
    transf = T_center @ Rw_e
    C_star = transf @ C_star @ transf.T
    return C_star

def decompose(C):
    ### From https://gitlab.inria.fr/tangram/pyellcv/-/blob/master/ellcv/types/ellipse.py?ref_type=heads#L79
    C /= -C[2, 2]
    center = -C[:2, 2]
    T_c = np.eye(3, dtype=float)
    T_c[:2, 2] = -center
    C_center = T_c @ C @ T_c.T
    C_center = 0.5 * (C_center + C_center.T)
    D, R = np.linalg.eig(C_center[:2, :2])
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    if R[0, 0] < 0.0: # force the first axis to be positive in x
        R *= -1.0
    ax = np.sqrt(np.abs(D))
    angle = np.arctan2(R[1, 0], R[0, 0])
    if ax[0] < ax[1]:
        ax[0], ax[1] = ax[1], ax[0]
        angle += np.pi/2
    return np.copy(ax), np.copy(angle), np.copy(center)

def decompose2(Q):
    ### From https://math.stackexchange.com/questions/4327313/get-ellipse-parameters-from-conic-coefficient-matrix
    # Get the coefficients
    A = Q[0, 0]
    B = Q[0, 1] * 2.0
    C = Q[1, 1]
    D = Q[0, 2] * 2.0
    E = Q[1, 2] * 2.0
    F = Q[2, 2]

    # Check if it is indeed an ellipse
    if np.linalg.det(Q) == 0:
        raise ValueError("Degenerate conic found!")

    if np.linalg.det(Q[:2,:2]) <= 0: # According to Wikipedia
        raise ValueError("These parameters do not define an ellipse!")

    # Get centre
    denominator = B**2 - 4*A*C
    centre_x = (2*C*D - B*E) / denominator
    centre_y = (2*A*E - B*D) / denominator
    # print("Centre x:{} y:{}".format(centre_x, centre_y))

    # Get major and minor axes
    K = - np.linalg.det(Q[:3,:3]) / np.linalg.det(Q[:2,:2])
    root = math.sqrt(((A - C)**2 + B**2))
    a = math.sqrt(2*K / (A + C - root))
    b = math.sqrt(2*K / (A + C + root))
    # print("Major:{} minor:{}".format(a, b))

    # Get angle
    angle_rad = math.atan2(C - A + root, B)+math.pi/2
    if a<b:
        a,b=b,a
        angle_rad+=np.pi/2
    return (a,b), angle_rad, (centre_x, centre_y)

# The following 2 functions are part of 3D-Aware-Ellipses-for-Visual-Localization.
#
# Author: Matthieu Zins (matthieu.zins@inria.fr)
#
# 3D-Aware-Ellipses-for-Visual-Localization is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# 3D-Aware-Ellipses-for-Visual-Localization is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with 3D-Aware-Ellipses-for-Visual-Localization.
# If not, see <http://www.gnu.org/licenses/>.

import os


def force_square_bbox(bbox, margin=0):
    """
        Transform a box into a square box.
        - bbox : 2x2 matrix with first corner on first row and second corner 
                 on the second row or 1D list [x_min, y_min, x_max, y_max]
        - margin: (optional) margin applied on the left, right top and bottom
    """
    x1, y1, x2, y2 = bbox.flatten().tolist()
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    d = math.ceil(max(w, h) / 2)
    d += margin

    cx = round((x1 + x2) / 2)
    cy = round((y1 + y2) / 2)

    return list(map(int, [cx-d, cy-d, cx+d, cy+d]))


def create_if_needed(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
