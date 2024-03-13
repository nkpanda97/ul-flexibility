%%
clear all
close all
step_size = .5;
matrix_input = load("polytope_matrix.mat");
A = double(matrix_input.A);
B = {double(matrix_input.B1), double(matrix_input.B2),double(matrix_input.B1)+double(matrix_input.B2)};



f =figure(1)

subplot(1,3,1)
PlotPolytope(A,B{1}, step_size, '#66c2a5', 0.5,'k',1.5 ,[252,141,98]./255,0.5,'k',30,true,'$EV_1$');

subplot(1,3,2)
PlotPolytope(A,B{2}, step_size, '#66c2a5', 0.5,'k',1.5 ,[252,141,98]./255,0.5,'k',20,true,'$EV_2$');

subplot(1,3,3)
PlotPolytope(A,B{3}, step_size, '#66c2a5', 0.5,'k',1.5 ,[252,141,98]./255,0.5,'k',50,true,'$EV_1\oplus EV_2$');





