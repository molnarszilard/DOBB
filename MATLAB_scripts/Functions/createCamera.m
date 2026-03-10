function [Camera] = createCamera(Ew,Rw_c,intrinsic)
% Generate camera.
% Ew: position in World Coordinate Frame (WCF) ([3x1])
% Rw_c: orientation in WCF ([3x3])
% f: camera plane equation in Camera Coordinate Frame (CCF) 

    Camera.Ew = Ew;     
    Camera.Ec = [0;0;0];     

    Camera.Rw_c = Rw_c;  
    
    Camera.f = intrinsic(1,1);        
    Camera.Oc = Rw_c'*([0;0;0]-Ew); 
    
    K = intrinsic; 
    Camera.K = K;
    R = Camera.Rw_c';       
    T = -R*Camera.Ew;       
    P = K*[R T];
    Camera.R = R;
    Camera.T = T;
    Camera.P = P/P(end,end);        
end