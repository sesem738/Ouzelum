import math
import torch


def lemniscate(a: float = math.sqrt(2), num_points: int = 200):
    # Parameters for the lemniscate curve
    # a = Max x-axis value 
    # num_points = Number of points to generate

    # Generate theta values
    theta = torch.linspace(-math.pi / 2, 3 * math.pi / 2, num_points)

    # Calculate x and y coordinates for the lemniscate curve
    x = a * torch.cos(theta) / (torch.sin(theta) ** 2 + 1)
    y = a * torch.cos(theta) * torch.sin(theta) / (torch.sin(theta) ** 2 + 1)

    return torch.vstack((x, y)).T

def circle(r: float = math.sqrt(2), num_points: int = 200):
    angle_step = 360 / num_points  # Angle increment between waypoints
    waypoints = []

    for i in range(num_points):
        angle = math.radians(i * angle_step)  # Convert angle to radians
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        waypoints.append((x, y))

    return torch.tensor(waypoints)

def square(side_length:float = 5,num_points:int = 8):
    if num_points < 4:
        raise ValueError("A square needs at least 4 waypoints.")
    
    # Calculate the number of waypoints per side
    waypoints_per_side = num_points // 4
    
    # Calculate the increment in x and y for each waypoint
    x_increment = side_length / (waypoints_per_side - 1)
    y_increment = side_length / (waypoints_per_side - 1)
    
    waypoints = []
    
    # Generate waypoints for the top side
    for i in range(waypoints_per_side):
        waypoints.append((i * x_increment, 0))
    
    # Generate waypoints for the right side
    for i in range(1, waypoints_per_side):
        waypoints.append((side_length, i * y_increment))
    
    # Generate waypoints for the bottom side
    for i in range(1, waypoints_per_side):
        waypoints.append((side_length - i * x_increment, side_length))

    # Generate waypoints for the left side
    for i in range(1, waypoints_per_side - 1):
        waypoints.append((0, side_length - i * y_increment))
    
    return -(torch.tensor(waypoints) - (side_length/2))


