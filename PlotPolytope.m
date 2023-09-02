function PlotPolytope(A,B, step_size, fc, alpha, edge_color,edge_width, projection_shade_color, projection_alpha, projection_edge_color,n_height,tf_bool,fig_title)
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
  
    trisurf(k1,x,y,z, 'FaceColor',fc, 'EdgeColor', fc, 'FaceAlpha',alpha, 'EdgeAlpha', alpha)
    axis equal
    hold on

    K = unique(k1);

    convex_hull_vertices =[x(K),y(K), z(K)];
    scatter3(x(K),y(K), z(K), 10,edge_color,'filled')
    % Get the number of rows in the matrix
    n = size(convex_hull_vertices, 1);
    
    % Initialize an empty cell array to store combinations
    combinations = cell(nchoosek(n, 2), 1);
    
    % Generate all combinations of two distinct rows
    count = 1;
    for i = 1:n
        for j = i+1:n
            combinations{count} = [convex_hull_vertices(i, :); convex_hull_vertices(j, :)];
            count = count + 1;
        end
    end
    
    e = 0;
    for i = 1 :length(combinations)
        mid_point = [mean(combinations{i}(:,1))+e, mean(combinations{i}(:,2))+e, mean(combinations{i}(:,3))+e];
        if_inside_points = sum(A*mid_point'==B);
        if if_inside_points>1
        
            plot3(combinations{i}(:,1), combinations{i}(:,2), combinations{i}(:,3), color=edge_color, LineWidth=edge_width)
            hold on
        end
    end
    kxz = convhull(x,z);
    kyx = convhull(x,y);
    kyz = convhull(y,z);
    fill3(x(kxz), n_height*ones(length(kxz)),z(kxz),projection_shade_color, FaceAlpha=projection_alpha,EdgeColor=projection_edge_color, LineStyle=":")
    fill3(x(kyx), y(kyx),n_height*ones(length(kyx)), projection_shade_color, FaceAlpha=projection_alpha,EdgeColor=projection_edge_color, LineStyle=":")
    fill3(n_height*ones(length(kyz)),y(kyz), z(kyz), projection_shade_color, FaceAlpha=projection_alpha,EdgeColor=projection_edge_color, LineStyle=":")
    title(fig_title,'Interpreter','latex')
    grid("on")
    view([-45.5608129184808 14.4])
% 
%     xlabel('$p_1$ (kW)','Interpreter','latex',Rotation=10,VerticalAlignment='top',HorizontalAlignment='center')
%     ylabel('$p_2$ (kW)','Interpreter','latex',Rotation=-10,VerticalAlignment='top',HorizontalAlignment='right')
%     zlabel('$p_3$ (kW)','Interpreter','latex')

    hold off
    
end

