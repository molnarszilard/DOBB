function [Ellipsoid] = createEllipsoid(Cw,Rw_ell,a,b,c)
% Generate ellipsoid.
% Cw: center in World Coordinate Frame (WCF) [3x1]
% Rw_ell: orientation in WCF [3x3]
% a,b,c: radii

    Ellipsoid.a = a;     
    Ellipsoid.b = b;     
    Ellipsoid.c = c;     
    
    Ellipsoid.Aell = diag([1/a^2,1/b^2,1/c^2]);    
    Ellipsoid.Rw_ell = Rw_ell;     
    Ellipsoid.Aw = Ellipsoid.Rw_ell*Ellipsoid.Aell*Ellipsoid.Rw_ell';
    
    Ellipsoid.Cw = Cw;    
    Ellipsoid.Cell = [0;0;0];        
    Ellipsoid.Oell =  Ellipsoid.Rw_ell'*([0;0;0]-Ellipsoid.Cw);

    Qell = [a^2 0 0 ; 0 b^2 0 ; 0 0 c^2];     
    Ellipsoid.Qell = [Qell [0;0;0] ; [0 0 0 1]];     
    Qw = Ellipsoid.Rw_ell*Qell*Ellipsoid.Rw_ell';     
    Qw = [Qw [0;0;0] ; [0 0 0 1]];        
    T = [1 0 0 Ellipsoid.Cw(1); 0 1 0 Ellipsoid.Cw(2); 0 0 1 Ellipsoid.Cw(3); 0 0 0 1];      
    Ellipsoid.Qw = T*Qw*T';       

end

