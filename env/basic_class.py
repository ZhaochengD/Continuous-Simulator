import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand

from .libs.kinematic_model import KinematicBicycleModel
from matplotlib.animation import FuncAnimation
from .libs.stanley_controller import StanleyController
from .libs.car_description import Description
from .libs.cubic_spline_interpolator import generate_cubic_spline
from .libs.utils import *

class Simulation:
    def __init__(self):

        fps = 50.0

        self.dt = 1/fps
        self.map_size = 30
        self.frames = 2500
        self.loop = False

class Path:

    def __init__(self, dir_path):

        # Get path to waypoints.csv
        dir_path = dir_path
        df = pd.read_csv(dir_path)

        x = df['X-axis'].values
        y = df['Y-axis'].values
        ds = 0.05

        self.px, self.py, self.pyaw, _ = generate_cubic_spline(x, y, ds)
        
    def plot(self, ax):
        ax.plot(self.px, self.py, '-', color='grey', linewidth=10)
        ax.plot(self.px, self.py, '--', color='gold')

class Car:
    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, v, v_var, dt):

        # Model parameters
        self.name = 'normal'
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.v = v + np.random.uniform(- v_var, v_var)
        self.delta = 0.0
        self.omega = 0.0
        self.L = 2.5
        self.max_steer = np.deg2rad(33)
        self.dt = dt
        self.c_r = 0.0
        self.c_a = 0.0

        # Tracker parameters
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.k = 8.0
        self.ksoft = 1.0
        self.kyaw = 0.01
        self.ksteer = 0.0
        self.crosstrack_error = None
        self.target_id = None

        # Queue relationship
        self.leader = None
        self.follower = None

        # Description parameters
        self.length = 4.5
        self.width = 2.0
        self.rear2wheel = 1.0
        self.wheel_dia = 0.15 * 2
        self.wheel_width = 0.2
        self.tread = 0.7
        self.colour = 'black'

        self.tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, self.max_steer, self.L, self.px, self.py, self.pyaw)
        self.kbm = KinematicBicycleModel(self.L, self.max_steer, self.dt, self.c_r, self.c_a)
        
        self.desc = Description(self.length, self.width, self.rear2wheel, self.wheel_dia, self.wheel_width, self.tread, self.L)

        # simulation attributes
        self.t = 1 / 50
    
    def IDM(self, v, delta_v, s):
        T = 1.0
        s_0 = 2
        delta = 4
        a = 3
        b = 1.5
        v_desired = 30
        s_star = s_0 + np.maximum(0, (v * T + (v * delta_v) / (2 * np.power(a * b, 0.5))))
        a0 = a * (1 - np.power((v / v_desired), delta) - np.power((s_star / s), 2))
        v_update = v + a0 * self.t
        
        if v_update < 0:
            v_update = 0
            a0 = -v / self.t
        return a0, v_update

    def drive(self):
        throttle = rand.uniform(150, 200)
        if self.leader:
            _, self.v = self.IDM(self.v, self.v - self.leader.v, L2_dis(self,self.leader))
        else:
            _, self.v = self.IDM(self.v, rand.randint(-1,1), rand.randint(30,35))
        self.delta, self.target_id, self.crosstrack_error = self.tracker.stanley_control(self.x, self.y, self.yaw, self.v, self.delta)
        self.x, self.y, self.yaw, self.v, _, _ = self.kbm.kinematic_model(self.x, self.y, self.yaw, self.v, throttle, self.delta)

        os.system('cls' if os.name=='nt' else 'clear')
        print("Cross-track term: {}".format(self.crosstrack_error))
        
    def plot(self, ax, path):
        outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = self.desc.plot_car(self.x, self.y, self.yaw, self.delta)
        ax.plot(outline_plot[0], outline_plot[1], color=self.colour)
        ax.plot(fr_plot[0], fr_plot[1], color=self.colour)
        ax.plot(rr_plot[0], rr_plot[1], color=self.colour)
        ax.plot(fl_plot[0], fl_plot[1], color=self.colour)
        ax.plot(rl_plot[0], rl_plot[1], color=self.colour)
        ax.plot(path.px[self.target_id], path.py[self.target_id])
        
    def getrepr(self):
        return [self.x, self.y, self.yaw, self.v, self.delta, self.omega]

class Ego(Car):
    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, v, v_var, dt):
        super(Ego, self).__init__(init_x, init_y, init_yaw, px, py, pyaw, v, v_var, dt)
        self.colour = 'red'
        self.name = 'ego'
        self.perception_radis = 20

    def drive(self):
        throttle = rand.uniform(150, 200)
        self.v = 30
        self.delta, self.target_id, self.crosstrack_error = self.tracker.stanley_control(self.x, self.y, self.yaw, self.v, self.delta)
        self.x, self.y, self.yaw, self.v, _, _ = self.kbm.kinematic_model(self.x, self.y, self.yaw, self.v, throttle, self.delta)

        os.system('cls' if os.name=='nt' else 'clear')
