import logging
import numpy as np
import random
from gym import spaces
from numpy.core._multiarray_umath import ndarray
from scipy.stats import norm
import matplotlib.pyplot as plt
from .libs.utils import *
import gym
import time
import copy
import torch
from .basic_class import Car, Path, Ego, Simulation

class LaneChangeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }
    def __init__(self, src_lane_dir, dst_lane_dir, 
            src_veh_num, dst_veh_num, src_v, dst_v, src_v_var, dst_v_var, car_gen_ratio):
        # hyper parameter
        self.car_gen_ratio = car_gen_ratio
        self.sim = Simulation()
        # initial environment
        self.src_path = Path(src_lane_dir)
        self.dst_path = Path(dst_lane_dir)
        self.ori_src_cars = [None]
        self.ori_dst_cars = [None]

        for i in range(src_veh_num):
            self.src_car_enqueue(Car(self.src_path.px[0], self.src_path.py[0], self.src_path.pyaw[0], 
                self.src_path.px, self.src_path.py, self.src_path.pyaw, src_v, src_v_var, self.sim.dt))
        for i in range(dst_veh_num):
            self.dst_car_enqueue(Car(self.dst_path.px[0], self.dst_path.py[0], self.dst_path.pyaw[0], 
                self.dst_path.px, self.dst_path.py, self.dst_path.pyaw, dst_v, dst_v_var, self.sim.dt))
        self.ego = Ego(self.dst_path.px[0], self.dst_path.py[0], self.dst_path.pyaw[0], 
                self.dst_path.px, self.dst_path.py, self.dst_path.pyaw, dst_v, dst_v_var, self.sim.dt)
        self.insert_ego('dst_lane', -4)
        self.src_cars = self.ori_src_cars
        self.dst_cars = self.ori_dst_cars
        
        # variable registration
        self.cars_on_src_lane = []
        self.cars_on_src_lane_num = 0
        self.cars_on_dst_lane = []
        self.cars_on_dst_lane_num = 0
        self.ego_enter = 0
        
        # env vehicle record
        self.src_context = []
        self.dst_context = []
        self.src_mark = -1
        self.dst_mark = -1

    def src_car_enqueue(self, car):
        car.follower = self.ori_src_cars[-1]
        if self.ori_src_cars[-1]:
            self.ori_src_cars[-1].leader = car
        self.ori_src_cars.append(car)

    def dst_car_enqueue(self, car):
        car.follower = self.ori_dst_cars[-1]
        if self.ori_dst_cars[-1]:
            self.ori_dst_cars[-1].leader = car
        self.ori_dst_cars.append(car)

    def come_new_car(self, path):
        # here I need specific phisical meaning?
        if path == 'src_lane':
            last_car_x = self.cars_on_src_lane[-1].x if len(self.cars_on_src_lane) > 0 else -1
        elif path == 'dst_lane':
            last_car_x = self.cars_on_dst_lane[-1].x if len(self.cars_on_dst_lane) > 0 else -1
        else:
            raise('No type')
        if np.random.exponential(0.5) > 2.0 and (last_car_x == -1 or last_car_x >= 5):
            return 1
        else:
            return 0
        
    def new_car_regist(self, lane):
        if lane == 'src_lane':
            if self.src_mark == -1:
                self.src_mark = 0
            car = self.src_cars.pop()
            if car.name == 'ego':
                self.ego = car
                self.ego_enter = 1
            self.cars_on_src_lane.append(car)
            self.cars_on_src_lane_num += 1
            
        elif lane == 'dst_lane':
            if self.dst_mark == -1:
                self.dst_mark = 0
                
            car = self.dst_cars.pop()
            if car.name == 'ego':
                self.ego = car
                self.ego_enter = 1
            self.cars_on_dst_lane.append(car)
            self.cars_on_dst_lane_num += 1
        else:
            assert("type the right str")

    def is_arrive(self, car):
        # if car x or y exceed dest coordination, then is arrive 
        pass

    def old_car_delete(self, car, lane):
        if lane == 'src_lane':
            self.cars_on_src_lane.remove(car)
            self.cars_on_src_lane_num -= 1
        elif lane == 'dst_lane':
            self.cars_on_dst_lane.remove(car)
            self.cars_on_dst_lane_num -= 1
        else:
            assert("type the right str")

    def insert_ego(self, lane, index):
        if lane == 'dst_lane':
            self.ego.leader = self.ori_dst_cars[index]
            self.ego.follower = self.ori_dst_cars[index-1]
            self.ori_dst_cars[index].follower = self.ego
            self.ori_dst_cars[index-1].leader = self.ego
            self.ori_dst_cars.insert(index, self.ego)
        if lane == 'src_lane':
            self.ego.leader = self.ori_src_cars[index]
            self.ego.follower = self.ori_src_cars[index-1]
            self.ori_src_cars[index].follower = self.ego
            self.ori_src_cars[index-1].leader = self.ego
            self.ori_src_cars.insert(index, self.ego)
            
    def ego_near_src(self):
        self.src_context = []
        src_temp1 = self.src_mark
        src_temp2 = self.src_mark
        src_shortest = L2_dis(self.cars_on_src_lane[src_temp1], self.ego)
        if src_shortest < self.ego.perception_radis:
            self.src_context.append(src_temp1)
        
        for i in range(-3, 4):
            if src_temp1 - 1 >= 0:
                src_temp1 -= 1
                dis = L2_dis(self.cars_on_src_lane[src_temp1], self.ego)
                if dis < self.ego.perception_radis:
                    self.src_context.append(src_temp1)
                if dis < src_shortest:
                    src_shortest = dis
                    self.src_mark = src_temp1
                    
        for i in range(-3, 4):
            if src_temp2 + 1 < len(self.cars_on_src_lane):
                src_temp2 += 1
                dis = L2_dis(self.cars_on_src_lane[src_temp2], self.ego)
                if dis < self.ego.perception_radis:
                    self.src_context.append(src_temp2)
                if dis < src_shortest:
                    src_shortest = dis
                    self.src_mark = src_temp2
        print('self.src_context', self.src_context)
        print('self.src_mark', self.src_mark)
        
    def ego_near_dst(self):
        self.dst_context = []
        dst_temp1 = self.dst_mark
        dst_temp2 = self.dst_mark
        dst_shortest = L2_dis(self.cars_on_dst_lane[dst_temp1], self.ego)
        if dst_shortest < self.ego.perception_radis:
            self.dst_context.append(dst_temp1)
        for i in range(-3, 4):
            if dst_temp1 - 1 >= 0:
                dst_temp1 -= 1
                dis = L2_dis(self.cars_on_dst_lane[dst_temp1], self.ego)
                if dis < self.ego.perception_radis:
                    self.dst_context.append(dst_temp1)
                if dis < dst_shortest:
                    dst_shortest = dis
                    self.dst_mark = dst_temp1
                    
        for i in range(-3, 4):
            if dst_temp2 + 1 < len(self.cars_on_dst_lane):
                dst_temp2 += 1
                dis = L2_dis(self.cars_on_dst_lane[dst_temp2], self.ego)
                if dis < self.ego.perception_radis:
                    self.dst_context.append(dst_temp2)
                if dis < dst_shortest:
                    dst_shortest = dis
                    self.dst_mark = dst_temp2
        print('self.dst_context', self.dst_context)
        print('self.dst_mark', self.dst_mark)
        
    def compose_graph(self):
        # the ego is 0 and other vehicle are aiggned to different index
        self.all_context = self.dst_context + self.src_context
        self.edge_index = [i + 1 for i in range(len(self.all_context))]
        self.edge_index = torch.tensor([self.edge_index, [0 for i in self.edge_index]], dtype=torch.long)
        self.node_feature = [self.ego.getrepr()]
        for veh_index in self.dst_context:
            self.node_feature.append(self.cars_on_dst_lane[veh_index].getrepr())
        for veh_index in self.src_context:
            self.node_feature.append(self.cars_on_src_lane[veh_index].getrepr())
        self.node_feature = torch.tensor([self.node_feature], dtype=torch.float)
        print(self.edge_index)
        print(self.node_feature)
        
    def reset(self):
        self.src_cars = copy.deepcopy(self.ori_src_cars)
        self.dst_cars = copy.deepcopy(self.ori_dst_cars)
        self.cars_on_src_lane = []
        self.cars_on_src_lane_num = 0
        self.cars_on_dst_lane = []
        self.cars_on_dst_lane_num = 0
        return 

    def step(self, action):
        # each vehicle drive a step and determine new vehicle
        for car in self.cars_on_src_lane:
            car.drive()
            if self.is_arrive(car):
                self.old_car_delete(car, 'src_lane')

        for car in self.cars_on_dst_lane:
            car.drive()
            if self.is_arrive(car):
                self.old_car_delete(car, 'dst_lane')
        
        if self.come_new_car('src_lane'):
            self.new_car_regist('src_lane')
        if self.come_new_car('dst_lane'):
            self.new_car_regist('dst_lane')
            
        if self.src_mark >= 0:
            self.ego_near_src()
        if self.dst_mark >= 0:
            self.ego_near_dst()
        if self.ego_enter:
            self.compose_graph()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        ax = plt.axes()
        ax.set_aspect('equal')
        self.src_path.plot(ax)
        self.dst_path.plot(ax)
        for car in self.cars_on_src_lane:
            outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = car.desc.plot_car(car.x, car.y, car.yaw, car.delta)
            ax.plot(outline_plot[0], outline_plot[1], color=car.colour)
            ax.plot(fr_plot[0], fr_plot[1], color=car.colour)
            ax.plot(rr_plot[0], rr_plot[1], color=car.colour)
            ax.plot(fl_plot[0], fl_plot[1], color=car.colour)
            ax.plot(rl_plot[0], rl_plot[1], color=car.colour)
            ax.plot(car.x, car.y, '+', color=car.colour, markersize=2)
        for car in self.cars_on_dst_lane:
            outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = car.desc.plot_car(car.x, car.y, car.yaw, car.delta)
            ax.plot(outline_plot[0], outline_plot[1], color=car.colour)
            ax.plot(fr_plot[0], fr_plot[1], color=car.colour)
            ax.plot(rr_plot[0], rr_plot[1], color=car.colour)
            ax.plot(fl_plot[0], fl_plot[1], color=car.colour)
            ax.plot(rl_plot[0], rl_plot[1], color=car.colour)

        ax.set_xlim(self.ego.x - 50, self.ego.x + 50)
        ax.set_ylim(self.ego.y - 50, self.ego.y + 50)
        if self.ego_enter:
            circle = plt.Circle((self.ego.x, self.ego.y), self.ego.perception_radis, fill=False, alpha=0.3)
            circle.set_zorder(5)
            ax.add_patch(circle)
        plt.pause(0.0000001)
        plt.clf()
