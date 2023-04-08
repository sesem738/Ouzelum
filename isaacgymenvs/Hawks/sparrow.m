% Define the time step and simulation duration
dt = 0.01;
t_end = 100;

% Define the square path for the moving robot
path_size = 2;
path_speed = 0.5;
path_x = [-path_size path_size path_size -path_size -path_size];
path_y = [-path_size -path_size path_size path_size -path_size];

% Initialize the drone and robot positions and velocities
drone_pos = [7 3 10];
drone_vel = [0 0 0];
robot_pos = [path_x(1) path_y(1) 0];
robot_vel = [path_speed 0 0];

% Define the landing target position
target_pos = [robot_pos(1) robot_pos(2) 0];

% Define the drone controller gains
kp = 0.09;
kd = 0.1;

% Initialize the simulation data arrays
t_data = 0:dt:t_end;
drone_pos_data = zeros(length(t_data), 3);
robot_pos_data = zeros(length(t_data), 3);

% Run the simulation
for i = 1:length(t_data)
    % Get the current robot position and velocity
    robot_pos = robot_pos + robot_vel * dt;

    target_pos = [robot_pos(1) robot_pos(2) 0];
    
    % Update the robot velocity if it has reached a path waypoint
    if norm([robot_pos(1) robot_pos(2)] - [path_x(1) path_y(1)]) < 0.1
        path_x = circshift(path_x, -1);
        path_y = circshift(path_y, -1);
        robot_vel = [path_x(1) - robot_pos(1) path_y(1) - robot_pos(2) 0] / norm([path_x(1) - robot_pos(1) path_y(1) - robot_pos(2) 0]) * path_speed;
    end
    
    % Get the current drone position and velocity
    drone_pos = drone_pos + drone_vel * dt;
    
    % Compute the drone control input
    drone_error = target_pos - drone_pos;
    drone_vel_desired = kp * drone_error + kd * (zeros(1,3) - drone_vel);
    
    % Update the drone velocity
    drone_vel = drone_vel + drone_vel_desired * dt;
    
    % Save the simulation data
    drone_pos_data(i,:) = drone_pos;
    robot_pos_data(i,:) = robot_pos;
end

% Plot the simulation data
figure;
plot3(drone_pos_data(:,1), drone_pos_data(:,2), drone_pos_data(:,3), 'b-', 'LineWidth', 2);
hold on;
plot3(robot_pos_data(:,1), robot_pos_data(:,2), robot_pos_data(:,3), 'r-', 'LineWidth', 2);
plot3(target_pos(1), target_pos(2), target_pos(3), 'gx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('X');
ylabel('Y');
zlabel('Z');
legend('Drone', 'Robot', 'Landing Target');
axis equal;

% Save the simulation data to a MAT file
save('drone_landing_data.mat', 't_data', 'drone_pos_data', 'robot_pos_data');
