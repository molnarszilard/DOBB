function [Ellipse] = projectEllipsoid(Ellipsoid,Camera)
% Faster than proj.

    P = Camera.P;

    Z = [Ellipsoid.Rw_ell Ellipsoid.Cw; 0 0 0 1];
    V = Z*diag([Ellipsoid.a^2,Ellipsoid.b^2,Ellipsoid.c^2,-1])*Z';
    V = V/V(end,end);
    C = P*V*P';
    C = C/C(end,end);
    
    center = squeeze(C(1:2,3));
    T = [1 0 -center(1); 0 1 -center(2); 0 0 1];
    C_cent = T*C*T';
    [ R, D ] = eig(C_cent(1:2,1:2));
    if(det(R)>0)
        a_ = sqrt(abs(D(1,1)));
        b_ = sqrt(abs(D(2,2)));
        theta = acos(R(1,1));
    else
        a_ = sqrt(abs(D(2,2)));
        b_ = sqrt(abs(D(1,1)));
        theta = acos(R(1,2));
        R = -R(:,[2 1]);
    end
    
    tmp1 = (inv(Camera.K)*[0 0 1]')*Camera.f;
    tmp2 = (inv(Camera.K)*[a_ b_ 1]')*Camera.f;
    a = tmp2(1)-tmp1(1);
    b = tmp2(2)-tmp1(2);

    Npts_ellipse = 50;
    t = linspace(0,2*pi,Npts_ellipse);
    xel = a*cos(t);
    yel = b*sin(t);
    
    Kc = (inv(Camera.K)*[center ; 1])*Camera.f;
    Kw = Camera.Rw_c*(Kc-Camera.Oc);
    
    tmp = R*[xel ; yel]+repmat(Kc(1:2),1,Npts_ellipse);
    xc = tmp(1,:);
    yc = tmp(2,:);
    zc = repmat(Kc(3),1,Npts_ellipse);
    
    tmp = Camera.Rw_c*([xc ; yc ; zc]-repmat(Camera.Oc,1,Npts_ellipse));
    xw = tmp(1,:); yw = tmp(2,:); zw = tmp(3,:);
 
    Ellipse.Ri_el = R;
    Ellipse.a = a;
    Ellipse.b = b;
    Ellipse.ai = a_;
    Ellipse.bi = b_;
    
    Ellipse.theta = theta;
    
    Ellipse.Kc = Kc;
    Ellipse.Kw = Kw;
    
    Ellipse.xc = xc;
    Ellipse.yc = yc;
    Ellipse.zc = zc;
    Ellipse.xw = xw;
    Ellipse.yw = yw;
    Ellipse.zw = zw;
    
    Ellipse.center = center;
    
end

