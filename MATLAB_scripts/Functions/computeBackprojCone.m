function [B_] = computeBackprojCone(Ellipse)
% Compute back-projection cone corresponding to Ellipse

    R = Ellipse.Ri_el;
    a = Ellipse.a;
    b = Ellipse.b;
    Kc = Ellipse.Kc;
    Ec = [0;0;0];
    
    Uc = [R(:,1) ; 0]; Vc = [R(:,2) ; 0]; Nc = [0 0 1]';     
    
    M=Uc*Uc'/a^2+Vc*Vc'/b^2;
    
    W=Nc/dot(Nc,(Kc-Ec));
    P=[1,0,0;0,1,0;0,0,1]-(Kc-Ec)*W';
    Q=W*W';
    B_=P'*M*P-Q;
    
end
