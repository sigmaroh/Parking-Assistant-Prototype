import numpy as np
from scipy.optimize import minimize
import copy
import cv2

class Environment:
    def __init__(self,obstacles):
        self.margin = 5
        #coordinates are in [x,y] format
        self.car_length = 80
        self.car_width = 40
        self.wheel_length = 15
        self.wheel_width = 7
        self.wheel_positions = np.array([[25,15],[25,-15],[-25,15],[-25,-15]])
        
        self.color = np.array([0,0,255])/255
        self.wheel_color = np.array([20,20,20])/255

        self.car_struct = np.array([[+self.car_length/2, +self.car_width/2],
                                    [+self.car_length/2, -self.car_width/2],  
                                    [-self.car_length/2, -self.car_width/2],
                                    [-self.car_length/2, +self.car_width/2]], 
                                    np.int32)
        
        self.wheel_struct = np.array([[+self.wheel_length/2, +self.wheel_width/2],
                                      [+self.wheel_length/2, -self.wheel_width/2],  
                                      [-self.wheel_length/2, -self.wheel_width/2],
                                      [-self.wheel_length/2, +self.wheel_width/2]], 
                                      np.int32)

        #height and width
        self.background = np.ones((1000+20*self.margin,1000+20*self.margin,3))
        self.background[10:1000+20*self.margin:10,:] = np.array([200,200,200])/255
        self.background[:,10:1000+20*self.margin:10] = np.array([200,200,200])/255
        self.place_obstacles(obstacles)
                
    def place_obstacles(self, obs):
        obstacles = np.concatenate([np.array([[0,i] for i in range(100+2*self.margin)]),
                                    np.array([[100+2*self.margin-1,i] for i in range(100+2*self.margin)]),
                                    np.array([[i,0] for i in range(100+2*self.margin)]),
                                    np.array([[i,100+2*self.margin-1] for i in range(100+2*self.margin)]),
                                    obs + np.array([self.margin,self.margin])])*10
        for ob in obstacles:
            self.background[ob[1]:ob[1]+10,ob[0]:ob[0]+10]=0
    
    def draw_path(self, path):
        path = np.array(path)*10
        color = np.random.randint(0,150,3)/255
        path = path.astype(int)
        for p in path:
            self.background[p[1]+10*self.margin:p[1]+10*self.margin+3,p[0]+10*self.margin:p[0]+10*self.margin+3]=color

    def rotate_car(self, pts, angle=0):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        return ((R @ pts.T).T).astype(int)

    def render(self, x, y, psi, delta):
        # x,y in 100 coordinates
        x = int(10*x)
        y = int(10*y)
        # x,y in 1000 coordinates
        # adding car body
        rotated_struct = self.rotate_car(self.car_struct, angle=psi)
        rotated_struct += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        rendered = cv2.fillPoly(self.background.copy(), [rotated_struct], self.color)

        # adding wheel
        rotated_wheel_center = self.rotate_car(self.wheel_positions, angle=psi)
        for i,wheel in enumerate(rotated_wheel_center):
            
            if i <2:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=delta+psi)
            else:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=psi)
            rotated_wheel += np.array([x,y]) + wheel + np.array([10*self.margin,10*self.margin])
            rendered = cv2.fillPoly(rendered, [rotated_wheel], self.wheel_color)

        # gel
        gel = np.vstack([np.random.randint(-50,-30,16),np.hstack([np.random.randint(-20,-10,8),np.random.randint(10,20,8)])]).T
        gel = self.rotate_car(gel, angle=psi)
        gel += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        gel = np.vstack([gel,gel+[1,0],gel+[0,1],gel+[1,1]])
        rendered[gel[:,1],gel[:,0]] = np.array([60,60,135])/255

        new_center = np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        self.background = cv2.circle(self.background, (new_center[0],new_center[1]), 2, [255/255, 150/255, 100/255], -1)

        rendered = cv2.resize(np.flip(rendered, axis=0), (700,700))
        return rendered


class Car_Dynamics:
    def __init__(self, x_0, y_0, v_0, psi_0, length, dt):
        self.dt = dt             # sampling time
        self.L = length          # vehicle length
        self.x = x_0
        self.y = y_0
        self.v = v_0
        self.psi = psi_0
        self.state = np.array([[self.x, self.y, self.v, self.psi]]).T

    def move(self, accelerate, delta):
        x_dot = self.v*np.cos(self.psi)
        y_dot = self.v*np.sin(self.psi)
        v_dot = accelerate
        psi_dot = self.v*np.tan(delta)/self.L
        return np.array([[x_dot, y_dot, v_dot, psi_dot]]).T

    def update_state(self, state_dot):
        # self.u_k = command
        # self.z_k = state
        self.state = self.state + self.dt*state_dot
        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.v = self.state[2,0]
        self.psi = self.state[3,0]

    
class MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix

    def mpc_cost(self, u_k, my_car, points):
        mpc_car = copy.copy(my_car)
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((2, self.horiz+1))
    
        desired_state = points.T
        cost = 0.0

        for i in range(self.horiz):
            state_dot = mpc_car.move(u_k[0,i], u_k[1,i])
            mpc_car.update_state(state_dot)
        
            z_k[:,i] = [mpc_car.x, mpc_car.y]
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-5, 5),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]



######################################################################################################################################################################

class Linear_MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix
        self.dt=0.2   
        self.L=4                          

    def make_model(self, v, psi, delta):        
        # matrices
        # 4*4
        A = np.array([[1, 0, self.dt*np.cos(psi)         , -self.dt*v*np.sin(psi)],
                    [0, 1, self.dt*np.sin(psi)         , self.dt*v*np.cos(psi) ],
                    [0, 0, 1                           , 0                     ],
                    [0, 0, self.dt*np.tan(delta)/self.L, 1                     ]])
        # 4*2 
        B = np.array([[0      , 0                                  ],
                    [0      , 0                                  ],
                    [self.dt, 0                                  ],
                    [0      , self.dt*v/(self.L*np.cos(delta)**2)]])

        # 4*1
        C = np.array([[self.dt*v* np.sin(psi)*psi                ],
                    [-self.dt*v*np.cos(psi)*psi                ],
                    [0                                         ],
                    [-self.dt*v*delta/(self.L*np.cos(delta)**2)]])
        
        return A, B, C

    def mpc_cost(self, u_k, my_car, points):
        
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((2, self.horiz+1))
        desired_state = points.T
        cost = 0.0
        old_state = np.array([my_car.x, my_car.y, my_car.v, my_car.psi]).reshape(4,1)

        for i in range(self.horiz):
            delta = u_k[1,i]
            A,B,C = self.make_model(my_car.v, my_car.psi, delta)
            new_state = A@old_state + B@u_k + C
        
            z_k[:,i] = [new_state[0,0], new_state[1,0]]
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
            
            old_state = new_state
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-5, 5),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]