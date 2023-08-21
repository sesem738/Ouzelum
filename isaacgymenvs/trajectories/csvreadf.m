% Read CSV file
data = csvread('flicker_0.01_ep_19.csv');  % Replace with your file name

% Extract data
robot_positions = data(:, 1:3);
target_positions = data(:, 4:6);

% Create a figure
figure;
hold on;

% Plot robot positions
plot3(robot_positions(:, 1), robot_positions(:, 2), robot_positions(:, 3), 50, 'b', 'filled', 'MarkerEdgeColor', 'k');
text(robot_positions(1, 1), robot_positions(1, 2), robot_positions(1, 3), 'Start', 'Color', 'r', 'FontSize', 12);
text(robot_positions(end, 1), robot_positions(end, 2), robot_positions(end, 3), 'Finish', 'Color', 'g', 'FontSize', 12);

% Plot target positions
plot3(target_positions(:, 1), target_positions(:, 2), target_positions(:, 3), 50, 'm', 'filled', 'MarkerEdgeColor', 'k');

startPosition = robotPositions(1, :);
finishPosition = robotPositions(end, :);

% Emphasize start and finish positions with markers
plot3(startPosition(1), startPosition(2), startPosition(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(finishPosition(1), finishPosition(2), finishPosition(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Set labels and title
xlabel('X Position');
ylabel('Y Position');
zlabel('Z Position');
title('Robot and Target Positions');

% Set grid and legend
grid on;
legend('Robot Positions', 'Target Positions', 'Start Position', 'Finish Position');

% Set view
view(3);

% Hold off
hold off;
