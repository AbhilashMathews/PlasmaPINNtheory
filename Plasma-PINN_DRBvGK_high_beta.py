# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:45:00 2021

@author: mathewsa
"""

import h5py
import numpy as np

B0 = 0.5 #T on-axis
R0 = 0.85 #m
a0 = 0.5 #m
R_ref = R0 + a0
B_ref = B0*R0/(R0 + a0)
n_ref = 7.*(10.**(19.)) #m^-3
Te_ref = 40.0 #eV
t_ref =  1.3273e-05 #s

#import h5py
#h5f = h5py.File('/home/mathewsa/Plasma-PINN_DRBvGK_manuscript_v0/NSTX_GK_highbetaEM.h5', 'w') 
#h5f.create_dataset('x_x', data=x_x)  
#h5f.create_dataset('x_y', data=x_y)  
#h5f.create_dataset('x_t', data=x_t)  
#h5f.create_dataset('y_den', data=y_den) 
#h5f.create_dataset('y_Te', data=y_Te)  
#h5f.create_dataset('y_phi', data=y_phi)   
#h5f.close()

data_file = '/home/mathewsa/Plasma-PINN_DRBvGK_manuscript_v0/NSTX_GK_highbetaEM.h5'
h5f = h5py.File(data_file, "r")
x_x = h5f['x_x'].value 
x_y = h5f['x_y'].value 
x_t = h5f['x_t'].value 
y_den = h5f['y_den'].value 
y_Te = h5f['y_Te'].value 


N_train = len(x_x)
idx = np.random.choice(len(x_x), N_train, replace=False)

x_train = x_x[idx,:]/a0
y_train = x_y[idx,:]/a0
t_train = x_t[idx,:]/t_ref
v1_train = y_den[idx,:]/n_ref
v5_train = y_Te[idx,:]/Te_ref

init_weight_den = (1./np.median(np.abs(v1_train)))
init_weight_Te = (1./np.median(np.abs(v5_train)))
sample_batch_size = int(1000)
N_outputs = 1 #number of predictions made by each PINN
timelen_end = 1.*20.0 #hours
layers = [3, 50, 50, 50, 50, 50, 50, 50, 50, N_outputs]

import time
import tensorflow as tf
np.random.seed(1234)
tf.set_random_seed(1234)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

class PhysicsInformedNN:
    def __init__(self, x, y, t, v1, v5, layers):
        X = np.concatenate([x, y, t], 1) 
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.X = X
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        self.v1 = v1
        self.v5 = v5
        self.layers = layers
        self.current_time = 0.0
        self.start_time = 0.0 
        self.weights_v1, self.biases_v1 = self.initialize_NN(layers)
        self.weights_v2, self.biases_v2 = self.initialize_NN(layers)
        self.weights_v3, self.biases_v3 = self.initialize_NN(layers)
        self.weights_v4, self.biases_v4 = self.initialize_NN(layers)
        self.weights_v5, self.biases_v5 = self.initialize_NN(layers) 
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                                     log_device_placement=False,
                                                     device_count={ "CPU": 32},
                                                     inter_op_parallelism_threads=1,
                                                     intra_op_parallelism_threads=32))
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.v1_tf = tf.placeholder(tf.float32, shape=[None, self.v1.shape[1]])
        self.v5_tf = tf.placeholder(tf.float32, shape=[None, self.v5.shape[1]])
        self.v1_pred, self.v5_pred,\
        self.PINN_v2_pred, self.PINN_v3_pred, self.PINN_v4_pred,\
        self.f_v1_pred, self.f_v5_pred = self.net_plasma(self.x_tf, self.y_tf, self.t_tf)
        self.loss1 = tf.reduce_mean(1.0*init_weight_den*tf.square(self.v1_tf - self.v1_pred))
        self.loss5 = tf.reduce_mean(1.0*init_weight_Te*tf.square(self.v5_tf - self.v5_pred))
        self.lossf1 = tf.reduce_mean(1.0*init_weight_den*tf.square(self.f_v1_pred))
        self.lossf5 = tf.reduce_mean(1.0*init_weight_Te*tf.square(self.f_v5_pred))
        self.optimizer_v1 = tf.contrib.opt.ScipyOptimizerInterface(self.loss1,
                                                                method = 'L-BFGS-B',
                                                                var_list=self.weights_v1+self.biases_v1,
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_v5 = tf.contrib.opt.ScipyOptimizerInterface(self.loss5,
                                                                method = 'L-BFGS-B',
                                                                var_list=self.weights_v5+self.biases_v5,
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_f = tf.contrib.opt.ScipyOptimizerInterface((self.lossf1 + self.lossf5),
                                                                method = 'L-BFGS-B',
                                                                var_list=self.weights_v2+self.biases_v2+self.weights_v3+self.biases_v3+self.weights_v4+self.biases_v4,
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        init = tf.global_variables_initializer() # Initialize Tensorflow variables
        self.sess.run(init)
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b) 
        return weights, biases
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim)) 
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        X_d0 = max(self.ub[0] - self.lb[0], 1e-6)
        X_d1 = max(self.ub[1] - self.lb[1], 1e-6)
        X_d2 = max(self.ub[2] - self.lb[2], 1e-6)
        X_d = np.array([X_d0, X_d1, X_d2])
        H = 2.0*(X - self.lb)/X_d - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) 
        return Y
    def net_plasma(self, x, y, t): 
        mi_me = 3672.3036
        eta = 67.1566
        B = (R_ref)/(a0*x) #has to use B_ref as norm since curvature needs to be evaluated at R0 + a0
        eps_R = 0.7407
        eps_v = 0.4303
        alpha_d = 0.0067
        tau_T = 1.0
        kappa_e = 11.0176
        kappa_i = 0.3134
        v1 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v1, self.biases_v1)
        v2 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v2, self.biases_v2)
        v3 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v3, self.biases_v3)
        v4 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v4, self.biases_v4)
        v5 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v5, self.biases_v5)
        PINN_v2 = v2
        PINN_v3 = v3
        PINN_v4 = v4 
        v1_t = tf.gradients(v1, t)[0]
        v1_x = tf.gradients(v1, x)[0]
        v1_y = tf.gradients(v1, y)[0]
        v2_x = tf.gradients(PINN_v2, x)[0]
        v2_y = tf.gradients(PINN_v2, y)[0]
        v5_t = tf.gradients(v5, t)[0]
        v5_x = tf.gradients(v5, x)[0]
        v5_y = tf.gradients(v5, y)[0] 
        pe = v1*v5
        pe_y = tf.gradients(pe, y)[0]
        jp = v1*((tau_T**0.5)*v4 - v3) 
        f_v1 = v1_t + (1./B)*(-v2_y*v1_x + v2_x*v1_y) - (-eps_R*(-v1*v2_y + alpha_d*pe_y))
        f_v5 = v5_t + (1./B)*(-v2_y*v5_x + v2_x*v5_y) - v5*(-5.*eps_R*alpha_d*v5_y/3. +\
                (2./3.)*(-eps_R*(-v2_y + alpha_d*pe_y/v1) +\
                (1./v1)*(0.71*eps_v*(0.0) + eta*jp*jp/(v5*mi_me))))
        return v1, v5, PINN_v2, PINN_v3, PINN_v4, f_v1, f_v5
    def callback(self, loss1, loss5, lossf1, lossf5):
        global Nfeval
        print(str(Nfeval)+' - PDE loss in loop: %.3e, %.3e, %.3e, %.3e' % (loss1, loss5, lossf1, lossf5))
        Nfeval += 1
    def fetch_minibatch(self, x_in, y_in, t_in, den_in, Te_in, N_train_sample):
        idx_batch = np.random.choice(len(x_in), N_train_sample, replace=False)
        x_batch = x_in[idx_batch,:]
        y_batch = y_in[idx_batch,:]
        t_batch = t_in[idx_batch,:]
        v1_batch = den_in[idx_batch,:]
        v5_batch = Te_in[idx_batch,:]
        return x_batch, y_batch, t_batch, v1_batch, v5_batch
    def train(self, timelen_end): 
        self.start_time = time.time()
        self.current_time = time.time() - self.start_time
        try:
            it = 0
            while self.current_time < timelen_end:
                it = it + 1
                print('Full iteration: '+str(it))
                x_res_batch, y_res_batch, t_res_batch, v1_res_batch, v5_res_batch = self.fetch_minibatch(self.x, self.y, self.t, self.v1, self.v5, sample_batch_size) # Fetch residual mini-batch
                tf_dict = {self.x_tf: x_res_batch, self.y_tf: y_res_batch, self.t_tf: t_res_batch,
                           self.v1_tf: v1_res_batch, self.v5_tf: v5_res_batch}
                self.optimizer_v1.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss1])
                self.optimizer_v5.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss5])
                self.optimizer_f.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss1, self.loss5, self.lossf1, self.lossf5],
                                        loss_callback = self.callback)
                self.current_time = time.time() - self.start_time
        except KeyboardInterrupt:
            raise 
        print('Finishing up training') 
    def predict(self, x_star, y_star, t_star): 
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        v1_star = self.sess.run(self.v1_pred, tf_dict)
        v5_star = self.sess.run(self.v5_pred, tf_dict)
        PINN_v2_star = self.sess.run(self.PINN_v2_pred, tf_dict)
        PINN_v3_star = self.sess.run(self.PINN_v3_pred, tf_dict)
        PINN_v4_star = self.sess.run(self.PINN_v4_pred, tf_dict) 
        return v1_star, v5_star, PINN_v2_star, PINN_v3_star, PINN_v4_star


model = PhysicsInformedNN(x_train, y_train, t_train, v1_train, v5_train, layers)
timelen_end = timelen_end*60.*60.
Nfeval = 1
model.train(timelen_end)

###### ---- code for plotting ---- ######

import math

len_loop_t = len(np.unique(x_t))
len_loop_x = len(np.unique(x_x))
len_loop_y = len(np.unique(x_y))

import matplotlib.pyplot as plt
N_time = 1
len_2d = len_loop_x*len_loop_y
len_skip = len_2d
X0 = x_x[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/a0
X1 = x_y[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/a0
#X2 = x_z[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/R_ref
X3 = x_t[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/t_ref
colormap = 'inferno'

output_model = model.predict(X0,X1,X3)

from scipy.interpolate import griddata
def grid(x, y, z, input_x, input_y): 
    xi = input_x
    yi = input_y 
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic')
    return X, Y, Z

resolution = 500


y_plot = []
y_plot.append(output_model[0]) #observed by PINN, must be learnt first for plotting
y_plot.append(output_model[1]) #observed by PINN, must be learnt first for plotting
norms = []
norms.append(n_ref)
norms.append(Te_ref)

i = 0
fig, axes = plt.subplots(nrows=2, ncols=1)
for ax in axes.flat:
    x_plot1 = np.linspace(np.min(X0), np.max(X0), resolution) #should be domain smaller than ne_test
    y_plot1 = np.linspace(np.min(X1), np.max(X1), resolution) #should be domain smaller than Te_test
    output = grid(X0[:,0], X1[:,0], y_plot[i][:,0], x_plot1, y_plot1)
    im = ax.scatter(output[0]*a0,output[1]*a0,c=output[2]*norms[i],cmap='YlOrRd_r')
    fig.colorbar(im, ax=axes[i])
    ax.set_xlim(1.2975,1.3825)
    ax.set_ylim(-0.084,0.084)
    ax.set_xlabel('R (m)')
    ax.set_ylabel('y (m)')
    if i%2 == 0:
        ax.set_title('Observed electron density: $n_e$ (m$^{-3}$)')
    if i%2 != 0:
        ax.set_title('Observed electron temperature: $T_e$ (eV)')
    i = i + 1

fig.subplots_adjust(right=0.8)
plt.subplots_adjust(hspace=0.5) 
plt.show()

y_phi = h5f['y_phi'].value 
factor_space = 1.0
phi_norm = B_ref*(a0**2)/t_ref 

xlim_min = factor_space*min(X0)[0]
xlim_max = factor_space*max(X0)[0]
ylim_min = factor_space*min(X1)[0]
ylim_max = factor_space*max(X1)[0]

inds = np.where((factor_space*X0[:,0] > xlim_min) & (factor_space*X0[:,0] < xlim_max))[0]

y_plot = []
var = y_phi[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/phi_norm
y_plot.append(var)
y_plot.append(output_model[2])
phi_norm = B_ref*(a0**2)/t_ref 
inds = np.where((X0[:,0] > xlim_min) & (X0[:,0] < xlim_max))[0]

i = 0
fig, axes = plt.subplots(nrows=2, ncols=1)
for ax in axes.flat:
    x_plot1 = np.linspace(np.min(X0), np.max(X0), resolution) #should be domain smaller than ne_test
    y_plot1 = np.linspace(np.min(X1), np.max(X1), resolution) #should be domain smaller than Te_test
    output = grid(X0[inds][:,0], X1[inds][:,0], y_plot[i][inds][:,0], x_plot1, y_plot1)
    im = ax.scatter(output[0]*a0,output[1]*a0,c=output[2]*phi_norm,cmap=colormap)
    ax.set_xlim(1.2975,1.3825)
    ax.set_ylim(-0.084,0.084)
    ax.set_xlabel('R (m)')
    ax.set_ylabel('y (m)')
    fig.colorbar(im, ax=axes[i])
    if i%2 == 0:
        ax.set_title(r'$\phi$ (V): gyrokinetic')
    if i%2 != 0:
        ax.set_title(r'$\phi$ (V): drift-reduced Braginskii')
    i = i + 1

fig.subplots_adjust(right=0.8) 
plt.subplots_adjust(hspace=0.5)
plt.show()


tot_e_field_true = []
X0_new = []
X1_new = []
i = 0
while i < len_loop_y:
    e_field_true = np.gradient(y_plot[0][i::len_loop_y][:,0],X0[i::len_loop_y][:,0])
    tot_e_field_true.append(e_field_true)
    X0_new.append(X0[i::len_loop_y][:,0])
    X1_new.append(X1[i::len_loop_y][:,0])
    i = i + 1

tot_e_field_true = np.hstack(tot_e_field_true)
X0_new = np.hstack(X0_new)
X1_new = np.hstack(X1_new)

tot_e_field_pred = []
i = 0
while i < len_loop_y:
    e_field_pred = np.gradient(y_plot[1][i::len_loop_y][:,0],X0[i::len_loop_y][:,0])
    tot_e_field_pred.append(e_field_pred)
    i = i + 1

tot_e_field_pred = np.hstack(tot_e_field_pred)

y_plot = [] 
y_plot.append(tot_e_field_true)
y_plot.append(tot_e_field_pred)

inds_new = np.where((X0_new > xlim_min) & (X0_new < xlim_max))[0]

i = 0
fig, axes = plt.subplots(nrows=2, ncols=1)
for ax in axes.flat:
    x_plot1 = np.linspace(np.min(X0), np.max(X0), resolution) #should be domain smaller than ne_test
    y_plot1 = np.linspace(np.min(X1), np.max(X1), resolution) #should be domain smaller than Te_test
    output = grid(X0_new[inds_new], X1_new[inds_new], y_plot[i][inds_new], x_plot1, y_plot1)
    im = ax.scatter(output[0]*a0,output[1]*a0,c=output[2]*phi_norm/a0,cmap=colormap)
    ax.set_xlim(1.2975,1.3825)
    ax.set_ylim(-0.084,0.084)
    ax.set_xlabel('R (m)')
    ax.set_ylabel('y (m)')
    fig.colorbar(im, ax=axes[i])
    if i%2 == 0:
        ax.set_title(r'$E_r$ (V/m): gyrokinetic')
    if i%2 != 0:
        ax.set_title(r'$E_r$ (V/m): drift-reduced Braginskii')
    i = i + 1

fig.subplots_adjust(right=0.8) 
plt.subplots_adjust(hspace=0.5)
plt.show()

#run this code 100 times and save data each time for statistical analysis