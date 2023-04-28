import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ExtendedKalmanFilter import ExtendedKalmanFilter as EKF

sim_animation = True

def getData(cols):
    fields = map(lambda x:np.array(sensorData[x]), cols)
    for i, val in enumerate(zip(*fields)):
        data = {c: val[i] for i, c in enumerate(cols)}
        yield data

if __name__ == '__main__':
    filename = 'sensor_readings.csv'
    sensorData = pd.read_csv(filename)
    xEst = np.zeros((4, 1))
    PEst = np.eye(4)
    ekf = EKF(xEst, PEst)

    cols = ['cam_x', 'cam_y', 'cam_theta', 'cmd_vx', 'cmd_wz', 'odom_x', 'odom_y', 'odom_wz', 'odom_vx', 'odom_vy', 'odom_theta']
    getData(cols)

    traj = []

    cam_x0 = odom_x0 = 0
    cam_y0 = odom_y0 = 0
    thetaInit = np.pi / 12

    for i, val in enumerate(getData(cols)):
        cam_x = val['cam_x']
        cam_y = val['cam_y']
        cam_theta = val['cam_theta']

        x = val['odom_x']
        y = val['odom_y']
        theta = val['odom_theta']

        v = val['cmd_vx']
        w = val['cmd_wz']

        vx = val['odom_vx']
        vy = val['odom_vy']
        vOdom = np.sqrt(vx * vx + vy * vy)
        wOdom = val['odom_wz']

        if i == 0:
            cam_x0 = cam_x
            cam_y0 = cam_y
            # compute odom offset
            thetaInit = (cam_theta + theta) / 2

        x0 = x * np.cos(thetaInit) - y * np.sin(thetaInit)
        y0 = x * np.sin(thetaInit) + y * np.cos(thetaInit)
        x, y = x0 + cam_x0, y0 + cam_y0

        # initialize ekf filter
        if i == 0:
            odom_x0 = x0
            odom_y0 = y0

            xEst[0, 0] = ekf.xEst[0, 0] = x - odom_x0
            xEst[1, 0] = ekf.xEst[1, 0] = y - odom_y0
            xEst[2, 0] = ekf.xEst[2, 0] = theta
            print(theta, val['cam_theta'], thetaInit)

        x -= odom_x0
        y -= odom_y0
        # fuse odom + cmd_vel
        z = np.array(([x], [y], [theta]))
        u = np.array(([v], [w]))
        ekf.ekf_update(z, u)

        # fuse cam + odom_vel
        zCam = np.array(([cam_x], [cam_y], [theta]))
        uOdom = np.array(([vOdom], [wOdom]))
        ekf.ekf_update(zCam, uOdom)


        # for visualization
        state = ekf.getObservation()
        traj.append(state)

        if sim_animation:
            ptraj = np.array(traj)
            plt.cla()
            plt.plot(ptraj[:, 0], ptraj[:, 1])
            plt.axis([-3, 2, -2.5, 4])
            plt.pause(0.01)


    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1])
    plt.show()