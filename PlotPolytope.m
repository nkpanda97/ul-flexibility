function [k1,x,y,z] = PlotPolytope(A,B, step_size, fc, ec, alpha, alpha_ec, tf_bool);
    %define the range of x,y,z coordinates
    x1=-100:step_size:100;
    y1=-100:step_size:100;
    z1=-100:step_size:100;
    %generate a grid with all triplets (x,y,z)
    [X,Y,Z] = meshgrid(x1,y1,z1);
%     I = (X+Y+Z<=e_max3) & (X+Y+Z>=e_min3) & (X+Y <=e_max2) & (X+Z<=e_max2) & (Y+Z<=e_max2) & (X+Z>=e_min2)  & (X+Y>=e_min2) & (Y+Z>=e_min2) ;
    I = (A(1,1)*X + A(1,2)*Y + A(1,3)*Z<=B(1)) ;
    for i=2:length(A)
        I = I & (A(i,1)*X + A(i,2)*Y + A(i,3)*Z<=B(i)) ;
    end
    x = X(I);
    y = Y(I);
    z = Z(I);
    [k1,~] = convhull(x,y,z,"Simplify",tf_bool);
  
    trisurf(k1,x,y,z, 'FaceColor',fc, 'EdgeColor', ec, 'FaceAlpha',alpha, 'EdgeAlpha', alpha_ec)
    axis equal
    hold on

end