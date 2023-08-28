clear all
close all
step_size = 0.5;
matrix_input = load("polytope_matrix.mat");
A = double(matrix_input.A);
B = {double(matrix_input.B1), double(matrix_input.B2),double(matrix_input.B1)+double(matrix_input.B2)};

fc = ["#1b9e77","#d95f02","#7570b3"];
alpha = [0.1, 0.5, 0.5];
ec = 'k';
alpha_ec = [0.2, 0.4];
figure;
PlotPolytope(A,B{1}, step_size, fc(1), ec, alpha(1), alpha_ec(1), true)
PlotPolytope(A,B{2}, step_size, fc(2), ec, alpha(2), alpha_ec(2), true)
PlotPolytope(A,B{3}, step_size, fc(3), ec, alpha(2), alpha_ec(2), true)
legend("EV #1", "EV #2", " M-Sum", 'NumColumns',3)
xlabel('Power [kW] (t=1)', Rotation=20)
ylabel('Power [kW] (t=2)', Rotation=-35)
zlabel('Power [kW] (t=3)')

