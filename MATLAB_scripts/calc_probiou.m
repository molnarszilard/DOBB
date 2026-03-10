
function iou = calc_probiou(obb1,obb2)
    % Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    % From https://github.com/ultralytics/ultralytics
    % The function parameters are the two obbs: cx,cy,w,h,angle (angle in radian)
    % The output of the function is the Probabilistic IoU value between the two
    eps = 1e-7;
    x1 = obb1(1);
    y1 = obb1(2);
    x2 = obb2(1);
    y2 = obb2(2);
    [a1,b1,c1] = covariance_matrix(obb1);
    [a2,b2,c2] = covariance_matrix(obb2);
    t1 = (((a1+a2)*(y1-y2)^2+(b1+b2)*(x1-x2)^2)/((a1+a2)*(b1+b2)-(c1+c2)^2+eps))*0.25;
    t2 = (((c1+c2)*(x2-x1)*(y1-y2))/((a1+a2)*(b1+b2)-(c1+c2)^2+eps))*0.5;
    t3 = log(((a1+a2)*(b1+b2)-(c1+c2)^2)/(4*sqrt(clip(a1*b1-c1^2,0.0,-0.1)*clip(a2*b2-c2^2,0.0,-0.1))+eps)+eps)*0.5;
    bd = clip(t1+t2+t3,eps,100.0);
    hd = sqrt(1.0-exp(-bd)+eps);
    iou = 1-hd;
end

function [a,b,c] = covariance_matrix(box)
    w = box(3)^2/12;
    h = box(4)^2/12;
    alpha = box(5);
    a = w*cos(alpha)^2+h*sin(alpha)^2;
    b = w*sin(alpha)^2+h*cos(alpha)^2;
    c = (w-h)*cos(alpha)*sin(alpha);
end

function clip_res = clip(input, minVal, maxVal)
% From https://www.mathworks.com/matlabcentral/fileexchange/135406-clip
% readSonixLSCTable.m    get the LSC tabel of Sonix 
%   Input:
%       input       the input data       
%       minVal      the min value of the outpue value
%       maxVal      the max value of the outpue value
%   Output:
%       clip_res    the res
%   Instructions:
%       author:     wtzhu
%       e-mail:     wtzhu_13@163.com
% Last Modified by wtzhu v1.0 2021-06-29
% Note: 
    input(input < minVal) = minVal;
    if maxVal>=minVal
        input(input > maxVal) = maxVal;
    end
    clip_res = input;
end



