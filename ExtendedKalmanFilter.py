import math
import numpy as np

DT = 0.01

# # Covariance for EKF simulation
# Q = np.diag([
#     0.1,  # variance of location on x-axis
#     0.1,  # variance of location on y-axis
#     np.deg2rad(1.0),  # variance of yaw angle
#     1.0  # variance of velocity
# ]) ** 2  # predict state covariance
# R = np.diag([1.0, 1.0, 1.0]) ** 2  # Observation x,y position covariance

# Covariance for EKF simulation
Q = np.diag([
    0.1 * 0.05,  # variance of location on x-axis
    0.1 * 0.05,  # variance of location on y-axis
    (1.0/180 * np.pi),  # variance of yaw angle
    1.0 * 0.05  # variance of velocity
])  # predict state covariance
R = np.diag([2.050, 2.050, 1.0]) ** 2  # Observation x,y position covariance


class ExtendedKalmanFilter:
    def __init__(self, xEst, PEst):
        '''

        :param xEst: 4x1 matrix : [[x], [y], [theta], [v]]
        :param PEst:
        '''
        self.xEst = xEst
        self.PEst = PEst

    @staticmethod
    def motion_model(x, u):
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        B = np.array([[DT * math.cos(x[2, 0]), 0],
                      [DT * math.sin(x[2, 0]), 0],
                      [0.0, DT],
                      [1.0, 0.0]])

        x = F @ x + B @ u

        return x


    @staticmethod
    def observation_model(x):
        # H = np.array([
        #     [0, 1, 0, 0],
        #     [-1, 0, 0, 0],
        #     [0, 0, 1, 0]
        # ])
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        z = H @ x

        return z

    def getObservation(self):
        return np.squeeze(self.observation_model(self.xEst))

    @staticmethod
    def jacob_f(x, u):
        """
        Jacobian of Motion Model

        motion model
        x_{t+1} = x_t+v*dt*cos(yaw)
        y_{t+1} = y_t+v*dt*sin(yaw)
        yaw_{t+1} = yaw_t+omega*dt
        v_{t+1} = v{t}
        so
        dx/dyaw = -v*dt*sin(yaw)
        dx/dv = dt*cos(yaw)
        dy/dyaw = v*dt*cos(yaw)
        dy/dv = dt*sin(yaw)
        """
        # theta = theta + w * dt
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array([
            [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
            [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])

        return jF

    @staticmethod
    def jacob_h():
        # Jacobian of Observation Model
        jH = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        return jH

    def ekf_update(self, z, u):
        #  Predict
        xPred = self.motion_model(self.xEst, u)
        jF = self.jacob_f(self.xEst, u)
        PPred = jF @ self.PEst @ jF.T + Q

        #  Update
        jH = self.jacob_h()
        zPred = self.observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + R
        K = PPred @ jH.T @ np.linalg.inv(S)
        self.xEst = xPred + K @ y
        self.PEst = (np.eye(len(self.xEst)) - K @ jH) @ PPred

    def __getitem__(self, item):
        if item == "mean":
            return self.xEst
        elif item == "cov":
            return self.PEst
        assert True, "item not found!"
