function Q = qGetQ(R)
    % Ensure R is a 3x3 rotation matrix
    if size(R,1) ~= 3 || size(R,2) ~= 3
        error('Input R must be a 3x3 matrix.');
    end

    % Compute the squared scalar part
    b1_squared = 0.25 * (1.0 + R(1,1) + R(2,2) + R(3,3));

    if b1_squared >= 0.25
        % w is the dominant component
        b1 = sqrt(b1_squared);
        over_b1_4 = 0.25 / b1;
        b2 = (R(3,2) - R(2,3)) * over_b1_4;
        b3 = (R(1,3) - R(3,1)) * over_b1_4;
        b4 = (R(2,1) - R(1,2)) * over_b1_4;
    else
        % Find the dominant component among x, y, z
        if (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
            % x is the dominant component
            b2 = sqrt(0.25 * (1 + R(1,1) - R(2,2) - R(3,3)));
            over_b2_4 = 0.25 / b2;
            b1 = (R(3,2) - R(2,3)) * over_b2_4;
            b3 = (R(1,2) + R(2,1)) * over_b2_4;
            b4 = (R(1,3) + R(3,1)) * over_b2_4;
        elseif R(2,2) > R(3,3)
            % y is the dominant component
            b3 = sqrt(0.25 * (1 + R(2,2) - R(1,1) - R(3,3)));
            over_b3_4 = 0.25 / b3;
            b1 = (R(1,3) - R(3,1)) * over_b3_4;
            b2 = (R(1,2) + R(2,1)) * over_b3_4;
            b4 = (R(2,3) + R(3,2)) * over_b3_4;
        else
            % z is the dominant component
            b4 = sqrt(0.25 * (1 + R(3,3) - R(1,1) - R(2,2)));
            over_b4_4 = 0.25 / b4;
            b1 = (R(2,1) - R(1,2)) * over_b4_4;
            b2 = (R(1,3) + R(3,1)) * over_b4_4;
            b3 = (R(2,3) + R(3,2)) * over_b4_4;
        end
    end

    % Return quaternion as [w, x, y, z]
    Q = [b1; b2; b3; b4];

    % Normalize to ensure unit quaternion
    Q = Q / norm(Q);
end


%{
function Q = qGetQ( R )
% qGetQ: converts 3x3 rotation matrix into equivalent quaternion
% Q = qGetQ( R );

[r,c] = size( R );
if( r ~= 3 | c ~= 3 )
    fprintf( 'R must be a 3x3 matrix\n\r' );
    return;
end

% [ Rxx, Rxy, Rxz ] = R(1,1:3); 
% [ Ryx, Ryy, Ryz ] = R(2,1:3);
% [ Rzx, Rzy, Rzz ] = R(3,1:3);

Rxx = R(1,1); Rxy = R(1,2); Rxz = R(1,3);
Ryx = R(2,1); Ryy = R(2,2); Ryz = R(2,3);
Rzx = R(3,1); Rzy = R(3,2); Rzz = R(3,3);

w = sqrt( trace( R ) + 1 ) / 2;

% check if w is real. Otherwise, zero it.
if( imag( w ) > 0 )
     w = 0;
end

x = sqrt( 1 + Rxx - Ryy - Rzz ) / 2;
y = sqrt( 1 + Ryy - Rxx - Rzz ) / 2;
z = sqrt( 1 + Rzz - Ryy - Rxx ) / 2;

[element, i ] = max( [w,x,y,z] );

if( i == 1 )
    x = ( Rzy - Ryz ) / (4*w);
    y = ( Rxz - Rzx ) / (4*w);
    z = ( Ryx - Rxy ) / (4*w);
end

if( i == 2 )
    w = ( Rzy - Ryz ) / (4*x);
    y = ( Rxy + Ryx ) / (4*x);
    z = ( Rzx + Rxz ) / (4*x);
end

if( i == 3 )
    w = ( Rxz - Rzx ) / (4*y);
    x = ( Rxy + Ryx ) / (4*y);
    z = ( Ryz + Rzy ) / (4*y);
end

if( i == 4 )
    w = ( Ryx - Rxy ) / (4*z);
    x = ( Rzx + Rxz ) / (4*z);
    y = ( Ryz + Rzy ) / (4*z);
end

Q = [ w; x; y; z ];

%}