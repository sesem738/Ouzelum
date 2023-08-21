% Specify the path to the CSV file
csvFilePath = 'flicker_0.01_ep_19.csv';

% Read the CSV file
data = readmatrix(csvFilePath);

% Extract robot and target positions
robotPositions = data(:, 1:3);
targetPositions = data(:, 4:6);

% Extract start and finish positions
startPosition = robotPositions(1, :);
finishPosition = robotPositions(end, :);

hold on;
rX = robotPositions(:, 1);
rY = robotPositions(:, 2);
rZ = robotPositions(:, 3);
tX = targetPositions(:, 1);
tY = targetPositions(:, 2);
tZ = targetPositions(:, 3);
% Plot robot positions in blue

plot3(rX,rY,rZ,"-",'LineWidth', 3, 'Color','b')
plot3(tX,tY,tZ,'LineWidth', 3)
text(robotPositions(1, 1), robotPositions(1, 2), robotPositions(1, 3), '  Start', 'Color', 'r', 'FontSize', 14);
text(robotPositions(end, 1), robotPositions(end, 2), robotPositions(end, 3), '     Finish', 'Color', 'r', 'FontSize', 14);

% Emphasize start and finish positions with markers
plot3(startPosition(1), startPosition(2), startPosition(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot3(finishPosition(1), finishPosition(2), finishPosition(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'g');

% Add labels and title
xlabel('X Position');
ylabel('Y Position');
zlabel('Z Position');
title('UAV and Moving Platform Trajectories');

% Add legend
legend('UAV', 'Husky', 'Start', 'Finish');

% Set grid and view
grid on;
view(3);

% Hold off to finalize the plot
hold off;