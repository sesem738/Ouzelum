import torch
# from ahrs.common.orientation import q2R
import numpy as np
import csv

class PVFilter:
    def __init__(self,acc_var,device) -> None:
        self.device = device

        # States: [Positions-Nav Velocity-Nav AccelBias]
        self.state = torch.zeros(9,1,device=self.device)
        self.cov = torch.eye(9,9,device=self.device)*1000
        self.acc_var = torch.diag(acc_var).to(self.device)
        self.time = 0

    def get_states(self):
        return self.state
    
    def get_covariances(self):
        return self.cov
    
    def set_states(self,state):
        self.state = state

    def prediction_step(self,accels:torch.Tensor,orientation:torch.Tensor,dt:float=0.02,sim_time=0,flip_Qw=True):

        self.time = sim_time

        accels = accels.reshape(3,1)
        orientation = orientation.reshape(1,4)
        # Assuming body space accels
        if flip_Qw:
            R_body_to_nav = quaternion_to_matrix(orientation[:,[3,0,1,2]]).squeeze().T
        else:
            R_body_to_nav = quaternion_to_matrix(orientation[:,[0,1,2,3]]).squeeze().T

        # R_2 = q2R(orientation[:,[3,0,1,2]]).squeeze()
        # print(torch.allclose(R_2,R_body_to_nav.cpu(),rtol=1e-05, atol=1e-08, equal_nan=False))
        
        ang_vars = torch.zeros(3,3,device=self.device)
        
        # Time Update
        # x = Fx + Gu

        # self.state[0:3] = self.state[0:3] + self.state[3:6]*dt + (R_body_to_nav@accels)*(dt**2)
        # self.state[3:6] = self.state[3:6] + R_body_to_nav@accels*dt*0.5
        
        F = torch.eye(9,device=self.device)
        F[0:3,3:6] = R_body_to_nav*dt
        F[0:3,6:9] = R_body_to_nav*(dt**2)*0.5
        F[3:6,3:6] = R_body_to_nav
        F[3:6,6:9] = R_body_to_nav*dt
        
        G = torch.zeros(9,3,device=self.device)
        G[0:6,0:3] = F[0:6,6:9]

        self.state = F@self.state + G@(accels-self.state[6:9,0].reshape(3,1))

        #TODO: Make tunable
        # P_ = FPF' + Q
        # Q = GUG' + WSW' + B
        B = torch.zeros(9,9,device=self.device)
        W = torch.zeros(9,4,device=self.device)
        self.cov[:,:] = (F@self.cov@F.T)  + (G@self.acc_var@G.T) #+ (W@ang_vars@W.T) + B


    def correction_step(self,gps_data:torch.Tensor=None,gps_var:torch.Tensor=None,
                        vel_data:torch.Tensor=None,vel_var:torch.Tensor=None):
        
        # Velocity Measurement update
        if vel_data is not None:
            vel_data = vel_data.reshape(3,1)
            IKH = torch.eye(9,device=self.device)

            # Compute K
            if gps_var is None:
                R = torch.zeros(3,3).to(self.device)
            else:
                R= torch.diag(vel_var).to(self.device)

            # K = PH'/(HPH' + R)
            K = self.cov[:,3:6]@(torch.inverse(self.cov[3:6,3:6]+R).to(self.device))

            # Compute new state
            self.state = self.state + K@(vel_data - self.state[3:6])

            # Compute 
            IKH[:,3:6] = IKH[:,3:6]-K
            self.cov = IKH@self.cov
        
        # Position Measurement update
        if gps_data is not None:
            gps_data = gps_data.reshape(3,1)
            IKH = torch.eye(9,device=self.device)

            # Compute K
            if gps_var is None:
                R = torch.zeros(3,3).to(self.device)
            else:
                R= torch.diag(gps_var).to(self.device)

            # K = PH'/(HPH' + R)
            K = self.cov[:,0:3]@(torch.inverse(self.cov[0:3,0:3]+R).to(self.device))

            # Compute new state
            self.state = self.state + K@(gps_data - self.state[0:3])

            # Compute 
            IKH[:,0:3] = IKH[:,0:3]-K
            self.cov = IKH@self.cov


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    quaternions = quaternions/torch.norm(quaternions)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
