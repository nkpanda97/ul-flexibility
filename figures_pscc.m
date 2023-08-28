%%
clear all
close all
step_size = .5;
shade_color = 'r';
c = 'blue';
n = [30,20,50];
a_p = 0.2;
matrix_input = load("polytope_matrix.mat");
A = double(matrix_input.A);
B = {double(matrix_input.B1), double(matrix_input.B2),double(matrix_input.B1)+double(matrix_input.B2)};

fc = ["#1b9e77","#d95f02","#7570b3"];
alpha = [0.1, 0.5, 0.5];
ec = 'k';
alpha_ec = [0.2, 0.4];

% PlotPolytope(A,B{1}, step_size, fc(1), ec, alpha(1), alpha_ec(1), true);
% PlotPolytope(A,B{2}, step_size, fc(2), ec, alpha(2), alpha_ec(2), true);

figure('units','inch','Position',[0,0, 6.8 4])

t = tiledlayout(1,3);

% First plot
ax1 = nexttile;
[~,x,y,z] = PlotPolytope(A,B{1}, step_size, fc(3), ec, alpha(2), alpha_ec(2), true);
kxz = convhull(x,z);
kyx = convhull(x,y);
kyz = convhull(y,z);
fill3(x(kxz), n(1)*ones(length(kxz)),z(kxz),shade_color, FaceAlpha=a_p,EdgeColor='k')
fill3(x(kyx), y(kyx),n(1)*ones(length(kyx)),shade_color, FaceAlpha=a_p,EdgeColor='k')
fill3(n(1)*ones(length(kyz)),y(kyz), z(kyz),shade_color, FaceAlpha=a_p,EdgeColor="k")
title('EV 1')
grid("on")

% Second plot
ax2 = nexttile;
[~,x,y,z] = PlotPolytope(A,B{2}, step_size, fc(3), ec, alpha(2), alpha_ec(2), true);
kxz = convhull(x,z);
kyx = convhull(x,y);
kyz = convhull(y,z);
fill3(x(kxz), n(2)*ones(length(kxz)),z(kxz),shade_color, FaceAlpha=a_p,EdgeColor="k")
fill3(x(kyx), y(kyx),n(2)*ones(length(kyx)),shade_color, FaceAlpha=a_p,EdgeColor="k")
fill3(n(2)*ones(length(kyz)),y(kyz), z(kyz),shade_color, FaceAlpha=a_p,EdgeColor="k")
title('EV 1')
grid("on")

% Third
ax3 = nexttile;
[~,x,y,z] = PlotPolytope(A,B{3}, step_size, fc(3), ec, alpha(2), alpha_ec(2), true);
kxz = convhull(x,z);
kyx = convhull(x,y);
kyz = convhull(y,z);
fill3(x(kxz), n(3)*ones(length(kxz)),z(kxz),shade_color, FaceAlpha=a_p,EdgeColor="k")
fill3(x(kyx), y(kyx),n(3)*ones(length(kyx)),shade_color, FaceAlpha=a_p,EdgeColor="k")
fill3(n(3)*ones(length(kyz)),y(kyz), z(kyz),shade_color, FaceAlpha=a_p,EdgeColor="k")
title('EV 1 \oplus EV 2')
grid("on")

% linkaxes([ax1 ax2],'xyz');









%% 
