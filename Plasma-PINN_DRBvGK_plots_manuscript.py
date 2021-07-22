# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:40:22 2021

@author: mathewsa

This code is used to produce the plots in the paper; it can only be run after all 100 PINNs have been trained and their outputs each saved
The outputs correspond to that of `output_model = model.predict(X0,X1,X3)` where X0, X1, X3 need to cover the (x,y,t) domain of interest
1 PINN can be trained for the low and high beta cases by running Plasma-PINN_DRBvGK_low_beta.py and Plasma-PINN_DRBvGK_high_beta.py, respectively,

"""

#EM low beta: 34587378 
start_frac_t = 0.25 #fraction of time to consider
end_frac_t = 0.30
start_frac_x = 0.2
end_frac_x = 0.8
start_frac_y = 0.2
end_frac_y = 0.8
frac_train = 1.0 #randomly sampled fraction of data to create training set
 
R0 = 0.85 #m
a0 = 0.5 #m
R_ref = R0 + a0 
t_ref =  1.3273e-05 #s 

x = [1.257173698595873 , 1.259762020514752 , 1.2623503424336309,
       1.26493866435251  , 1.267526986271389 , 1.2701153081902679,
       1.2727036301091468, 1.275291952028026 , 1.2778802739469048,
       1.2804685958657838, 1.2830569177846627, 1.2856452397035416,
       1.2882335616224208, 1.2908218835412997, 1.2934102054601786,
       1.2959985273790575, 1.2985868492979367, 1.3011751712168156,
       1.3037634931356945, 1.3063518150545734, 1.3089401369734524,
       1.3115284588923315, 1.3141167808112104, 1.3167051027300893,
       1.3192934246489683, 1.3218817465678474, 1.3244700684867263,
       1.3270583904056052, 1.3296467123244842, 1.332235034243363 ,
       1.3348233561622422, 1.3374116780811212, 1.34              ,
       1.342588321918879 , 1.345176643837758 , 1.347764965756637 ,
       1.350353287675516 , 1.352941609594395 , 1.3555299315132738,
       1.3581182534321528, 1.360706575351032 , 1.3632948972699108,
       1.3658832191887897, 1.3684715411076687, 1.3710598630265478,
       1.3736481849454267, 1.3762365068643057, 1.3788248287831846,
       1.3814131507020635, 1.3840014726209426, 1.3865897945398216,
       1.3891781164587005, 1.3917664383775794, 1.3943547602964586,
       1.3969430822153375, 1.3995314041342164, 1.4021197260530953,
       1.4047080479719742, 1.4072963698908534, 1.4098846918097323,
       1.4124730137286112, 1.4150613356474901, 1.4176496575663693,
       1.4202379794852482, 1.4228263014041271]

y = [-0.1456526028082542, -0.1433767808893752, -0.1411009589704962,
       -0.1388251370516172, -0.1365493151327383, -0.1342734932138593,
       -0.1319976712949803, -0.1297218493761013, -0.1274460274572224,
       -0.1251702055383434, -0.1228943836194644, -0.1206185617005855,
       -0.1183427397817065, -0.1160669178628275, -0.1137910959439486,
       -0.1115152740250696, -0.1092394521061906, -0.1069636301873116,
       -0.1046878082684327, -0.1024119863495537, -0.1001361644306747,
       -0.0978603425117958, -0.0955845205929168, -0.0933086986740378,
       -0.0910328767551588, -0.0887570548362799, -0.0864812329174009,
       -0.0842054109985219, -0.081929589079643 , -0.079653767160764 ,
       -0.077377945241885 , -0.075102123323006 , -0.0728263014041271,
       -0.0705504794852481, -0.0682746575663691, -0.0659988356474902,
       -0.0637230137286112, -0.0614471918097322, -0.0591713698908533,
       -0.0568955479719743, -0.0546197260530953, -0.0523439041342163,
       -0.0500680822153374, -0.0477922602964584, -0.0455164383775794,
       -0.0432406164587004, -0.0409647945398215, -0.0386889726209425,
       -0.0364131507020635, -0.0341373287831846, -0.0318615068643056,
       -0.0295856849454266, -0.0273098630265477, -0.0250340411076687,
       -0.0227582191887897, -0.0204823972699107, -0.0182065753510318,
       -0.0159307534321528, -0.0136549315132738, -0.0113791095943949,
       -0.0091032876755159, -0.0068274657566369, -0.0045516438377579,
       -0.002275821918879 ,  0.                ,  0.002275821918879 ,
        0.0045516438377579,  0.0068274657566369,  0.0091032876755159,
        0.0113791095943949,  0.0136549315132738,  0.0159307534321528,
        0.0182065753510318,  0.0204823972699107,  0.0227582191887897,
        0.0250340411076687,  0.0273098630265476,  0.0295856849454266,
        0.0318615068643056,  0.0341373287831846,  0.0364131507020635,
        0.0386889726209425,  0.0409647945398215,  0.0432406164587004,
        0.0455164383775794,  0.0477922602964584,  0.0500680822153374,
        0.0523439041342163,  0.0546197260530953,  0.0568955479719743,
        0.0591713698908533,  0.0614471918097322,  0.0637230137286112,
        0.0659988356474902,  0.0682746575663691,  0.0705504794852481,
        0.0728263014041271,  0.0751021233230061,  0.077377945241885 ,
        0.079653767160764 ,  0.081929589079643 ,  0.0842054109985219,
        0.0864812329174009,  0.0887570548362799,  0.0910328767551588,
        0.0933086986740378,  0.0955845205929168,  0.0978603425117958,
        0.1001361644306747,  0.1024119863495537,  0.1046878082684327,
        0.1069636301873116,  0.1092394521061906,  0.1115152740250696,
        0.1137910959439485,  0.1160669178628275,  0.1183427397817065,
        0.1206185617005855,  0.1228943836194644,  0.1251702055383434,
        0.1274460274572224,  0.1297218493761014,  0.1319976712949803,
        0.1342734932138593,  0.1365493151327383,  0.1388251370516173,
        0.1411009589704962,  0.1433767808893752,  0.1456526028082542]

d_x = []
i = 0
while i < len(x) - 1:
    d_x_new = (x[i]+x[i+1])/2.
    d_x.append(d_x_new)
    i = i + 1

d_y = []
i = 0
while i < len(y) - 1:
    d_y_new = (y[i]+y[i+1])/2.
    d_y.append(d_y_new)
    i = i + 1

d_t = []
i = 0
while i < 1001:
    d_t_new = float(i)*0.0000002
    d_t.append(d_t_new)
    i = i + 1


import numpy as np

a0 = a = 0.5        #.minor radius (m)
Te0    = 40.0          #.Electron temperature (eV).
Ti0    = 40.0           #.Ion temperature (eV).
n0     = n_ref = 7.*(10.**(18.)) #m^-3
B0     = 0.3148 #T, is B0 in centre of simulation domain; 0.5 T on magnetic axis
R0     = 0.85 #m 
R_ref  = R0 + a0
B_ref  = B0*(R0 + a0)/(R0 + a0)        #.Magnetic field (T) on axis?
mime   = 3672.3036           #.Mass ratio = m_i/m_e.
Z      = 1             #.Ionization level.

u     = 1.660539040e-27     #.unifited atomic mass unit (kg).
m_H   = 1.007276*u          #.Hydrogen ion (proton) mass (kg).
mu    = 2.0 #.39.948              #.m_i/m_proton.
m_Ar  = mu*m_H              #.Argon ion (singly ionized?) mass (kg).
m_i   = m_Ar                #.ion mass (kg).
m_e   = 0.910938356e-30     #.electron mass (kg).
c     = 299792458.0         #.speed of light (m/s).
e     = 1.60217662e-19      #.electron charge (C).

cse0   = np.sqrt(e*Te0/m_i)     #.Electron sound speed (m/s).
csi0   = np.sqrt(e*Ti0/m_i)     #.Ion sound speed (m/s).

tRef   = np.sqrt((R_ref*a)/2.0)/cse0 
phi_norm = B_ref*(a**2)/tRef

sorted_vars = []
len_loop_x = int(end_frac_x*len(d_x))
len_loop_y = int(end_frac_y*len(d_y))
len_loop_t = int(end_frac_t*len(d_t))
print('Stage 3')
i_t = int(start_frac_t*len(d_t))
while i_t < len_loop_t:
    i_x = int(start_frac_x*len(d_x))
    while i_x < len_loop_x:
        i_y = int(start_frac_y*len(d_y))
        while i_y < len_loop_y:
            sorted_vars.append([[d_x[i_x]],[d_y[i_y]],[d_t[i_t]]])
            i_y = i_y + 1
        i_x = i_x + 1
    i_t = i_t + 1
    print(float(i_t)/len_loop_t)


full_vars_array = np.array(sorted_vars)
x_x = full_vars_array[:,0]
x_y = full_vars_array[:,1]
x_t = full_vars_array[:,2]

import math
number_of_models = 100
len_loop_t = int(end_frac_t*len(d_t)) - int(start_frac_t*len(d_t))
len_loop_x = int(end_frac_x*len(d_x)) - int(start_frac_x*len(d_x))
len_loop_y = int(end_frac_y*len(d_y)) - int(start_frac_y*len(d_y))

import matplotlib.pyplot as plt

N_time = int(len_loop_t - 2) #must be lower than len_loop_t and equal/greater than 0
len_2d = len_loop_x*len_loop_y
len_skip = len_2d
X0 = x_x[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/a0
X1 = x_y[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/a0
X3 = x_t[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/t_ref
xlim_min = min(X0)[0]#-0.02
xlim_max = max(X0)[0]#0.02
ylim_min = min(X1)[0]#-0.02
ylim_max = max(X1)[0]#0.02

import numpy as np
import h5py
data_file = '/nobackup1/mathewsa/DRBvGK/EM_low_beta.h5' #1.4 GB; too large to upload on Github
h5f = h5py.File(data_file, "r")
x_array = h5f['x_array'].value 
y_array = h5f['y_array'].value  
t_array = h5f['t_array'].value 
x_new_array = h5f['x_new_array'].value 
y_new_array = h5f['y_new_array'].value  
t_new_array = h5f['t_new_array'].value 
ne_array_true = h5f['ne_array_true'].value 
Te_array_true = h5f['Te_array_true'].value 
phi_array_true = h5f['phi_array_true'].value 
Er_array_true = h5f['Er_array_true'].value 
ne_array_pred = h5f['ne_array_pred'].value 
Te_array_pred = h5f['Te_array_pred'].value 
phi_array_pred = h5f['phi_array_pred'].value 
Er_array_pred = h5f['Er_array_pred'].value 
 
relative_path = '/results/NSTX_gyro_v_DRB_33377267_EM_v17/'
save_figs_path = '/home/mathewsa/PINNs/main/gdbh'+str(relative_path)

#y-averaged Er vs x comparison
time_plot_index = time_index = int(len_loop_t - 2) #selects time index; #must be less than 50 since only 50 time slices used

N_model = 0
avg_Er_pred_yvst = []
avg_Er_true_yvst = []
while N_model < number_of_models:
    i_avg = 0
    while i_avg < len_loop_x:
        avg_Er_pred_yvst.append(np.mean(Er_array_pred[N_model*len_loop_t:(N_model+1)*len_loop_t][time_plot_index][i_avg::len_loop_x]))
        avg_Er_true_yvst.append(np.mean(Er_array_true[N_model*len_loop_t:(N_model+1)*len_loop_t][time_plot_index][i_avg::len_loop_x]))
        i_avg = i_avg + 1
    N_model = N_model + 1

avg_Er_pred_yvst = np.array(avg_Er_pred_yvst)
avg_Er_true_yvst = np.array(avg_Er_true_yvst)

all_Er = [] 
N_model = 0
while N_model < number_of_models:
    all_Er.append(avg_Er_pred_yvst[N_model*len_loop_x:(N_model+1)*len_loop_x])
    N_model = N_model + 1  


all_Er = np.array(all_Er)
mean_Er = np.mean(all_Er, axis=0)
std_Er = np.std(all_Er, axis=0)


plt.figure()
#plt.style.use('seaborn-white')
plt.scatter((x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x])*a0,(avg_Er_true_yvst[0:len_loop_x])*(-phi_norm/a0),label='Gyrokinetic',color='black',marker='x')
plt.plot(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0,mean_Er*(-phi_norm/a0),label='Drift-reduced Braginskii',color='green')
plt.fill_between(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0,(mean_Er+2.*std_Er)*(-phi_norm/a0),(mean_Er-2.*std_Er)*(-phi_norm/a0),interpolate=True,color='green',alpha=0.1)
plt.xlabel('R (m)')
plt.ylabel('$<E_r>_y$ (V/m)')
#plt.title('$t$ = %.2f $\mu$s'%((10.**6.)*tRef*t_array[time_plot_index][0][0]))
plt.legend(frameon=False)
plt.xlim(1.2975,1.3825)#(np.min(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0),np.max(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0))
plt.savefig(str(save_figs_path)+'lowbetaEM_avgEry_vs_x.png')
plt.savefig(str(save_figs_path)+'lowbetaEM_avgEry_vs_x.eps')
plt.show()


h5f = h5py.File('/home/mathewsa/Plasma-PINN_DRBvGK_manuscript_v0/PlasmaPINN_DRBvGK_Figure1.h5', 'w') 
h5f.create_dataset('X', data=((x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x])*a0))  
h5f.create_dataset('Y', data=((avg_Er_true_yvst[0:len_loop_x])*(-phi_norm/a0)))  
h5f.create_dataset('Y_MEAN', data=(mean_Er*(-phi_norm/a0)))  
h5f.create_dataset('Y_STD', data=(std_Er*(-phi_norm/a0)))  
h5f.close()

















plt.figure()
#plt.style.use('seaborn-white')
plt.scatter((x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x])*a0,(avg_Er_true_yvst[0:len_loop_x])*(-phi_norm/a0),label='Gyrokinetic',color='black',marker='x')
plt.plot(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0,mean_Er*(-phi_norm/a0),label='Drift-reduced Braginskii',color='green')
plt.fill_between(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0,(mean_Er+2.*std_Er)*(-phi_norm/a0),(mean_Er-2.*std_Er)*(-phi_norm/a0),interpolate=True,color='green',alpha=0.1)
plt.xlabel('R (m)')
plt.ylabel('$<E_r>_y$ (V/m)')
#plt.title('$t$ = %.2f $\mu$s'%((10.**6.)*tRef*t_array[time_plot_index][0][0]))
plt.legend(frameon=False)
plt.xlim(np.min(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0),np.max(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0))
plt.savefig(str(save_figs_path)+'highbetaES_avgEry_vs_x1.png')
plt.savefig(str(save_figs_path)+'highbetaES_avgEry_vs_x1.eps')
plt.show()

h5f = h5py.File('/home/mathewsa/Plasma-PINN_DRBvGK_manuscript_v0/PlasmaPINN_DRBvGK_Figure4.h5', 'w') 
h5f.create_dataset('X', data=((x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x])*a0))  
h5f.create_dataset('Y', data=((avg_Er_true_yvst[0:len_loop_x])*(-phi_norm/a0)))  
h5f.create_dataset('Y_MEAN', data=(mean_Er*(-phi_norm/a0)))  
h5f.create_dataset('Y_STD', data=(std_Er*(-phi_norm/a0)))  
h5f.close()















#only pred ne, Te, phi, and Er necessary to average, all rest are constant across models
x_array_plot1 = np.mean(x_array[time_index::number_of_models],axis=0)
y_array_plot1 = np.mean(y_array[time_index::number_of_models],axis=0)
t_array_plot1 = np.mean(t_array[time_index::number_of_models],axis=0)
x_new_array_plot1 = np.mean(x_new_array[time_index::number_of_models],axis=0)
y_new_array_plot1 = np.mean(y_new_array[time_index::number_of_models],axis=0)
t_new_array_plot1 = np.mean(t_new_array[time_index::number_of_models],axis=0)
ne_array_true_plot1 = np.mean(ne_array_true[time_index::number_of_models],axis=0)
Te_array_true_plot1 = np.mean(Te_array_true[time_index::number_of_models],axis=0)
phi_array_true_plot1 = np.mean(phi_array_true[time_index::number_of_models],axis=0)
Er_array_true_plot1 = np.mean(Er_array_true[time_index::number_of_models],axis=0)
ne_array_pred_plot1 = np.mean(ne_array_pred[time_index::number_of_models],axis=0)
Te_array_pred_plot1 = np.mean(Te_array_pred[time_index::number_of_models],axis=0)
phi_array_pred_plot1 = np.mean(phi_array_pred[time_index::number_of_models],axis=0)
Er_array_pred_plot1 = np.mean(Er_array_pred[time_index::number_of_models],axis=0)

ne_array_pred_plot1_std = np.std(ne_array_pred[time_index::number_of_models],axis=0)
Te_array_pred_plot1_std = np.std(Te_array_pred[time_index::number_of_models],axis=0)
phi_array_pred_plot1_std = np.std(phi_array_pred[time_index::number_of_models],axis=0)
Er_array_pred_plot1_std = np.std(Er_array_pred[time_index::number_of_models],axis=0)

from scipy.interpolate import griddata
def grid(x, y, z, input_x, input_y): 
    xi = input_x
    yi = input_y 
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic')
    return X, Y, Z

resolution = 500
colormap = 'inferno'

i = 0
fig, ax = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row', figsize=(11,7))
for i in range(2):
    for j in range(3):
        x_plot1 = np.linspace(np.min(np.unique(x_array_plot1*a0)), np.max(np.unique(x_array_plot1*a0)), resolution) #should be domain smaller than ne_test
        y_plot1 = np.linspace(np.min(np.unique(y_array_plot1*a0)), np.max(np.unique(y_array_plot1*a0)), resolution) #should be domain smaller than Te_test
        if i == 0:
            if j == 0:
                output = grid(x_array_plot1[:,0]*a0, y_array_plot1[:,0]*a0, ne_array_true_plot1[:,0]*n0, x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap='YlOrRd_r')
                ax[i, j].set_ylabel('y (m)')
                ax[i, j].set_title('$n_e$ (m$^{-3}$)')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])#, pad=0.1)
            if j == 1:
                output = grid(x_array_plot1[:,0]*a0, y_array_plot1[:,0]*a0, phi_array_true_plot1[:,0]*phi_norm, x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap=colormap)
                ax[i, j].set_title(r'$\phi$ (V): gyrokinetic')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])
            if j == 2:
                output = grid(x_new_array_plot1*a0, y_new_array_plot1*a0, Er_array_true_plot1*(-phi_norm/a0), x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap=colormap)#,norm=norm)
                ax[i, j].set_title(r'$E_r$ (V/m): gyrokinetic')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])
        if i == 1:
            if j == 0:
                output = grid(x_array_plot1[:,0]*a0, y_array_plot1[:,0]*a0, Te_array_true_plot1[:,0]*Te0, x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap='YlOrRd_r')        
                ax[i, j].set_xlabel('R (m)')
                ax[i, j].set_ylabel('y (m)')
                ax[i, j].set_title('$T_e$ (eV)')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])
            if j == 1:
                output = grid(x_array_plot1[:,0]*a0, y_array_plot1[:,0]*a0, phi_array_pred_plot1[:,0]*phi_norm, x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap=colormap)
                ax[i, j].set_xlabel('R (m)')
                ax[i, j].set_title(r'$\phi$ (V): drift-reduced Braginskii')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])
            if j == 2:
                output = grid(x_new_array_plot1*a0, y_new_array_plot1*a0, Er_array_pred_plot1*(-phi_norm/a0), x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap=colormap)#,norm=norm)
                ax[i, j].set_xlabel('R (m)')
                ax[i, j].set_title(r'$E_r$ (V/m): drift-reduced Braginskii')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])
        
        
plt.savefig(str(save_figs_path)+'EM_low_beta_GKvDRB_avg.png')
plt.savefig(str(save_figs_path)+'EM_low_beta_GKvDRB_avg.eps')
plt.show()

h5f = h5py.File('/home/mathewsa/Plasma-PINN_DRBvGK_manuscript_v0/PlasmaPINN_DRBvGK_Figure2.h5', 'w') 
h5f.create_dataset('X', data=(x_array_plot1[:,0]*a0))  
h5f.create_dataset('Y', data=(y_array_plot1[:,0]*a0))  
h5f.create_dataset('Yplota', data=(ne_array_true_plot1[:,0]*n0))  
h5f.create_dataset('Yplotb', data=(Te_array_true_plot1[:,0]*Te0))  
h5f.create_dataset('Yplotc', data=(phi_array_true_plot1[:,0]*phi_norm))  
h5f.create_dataset('Yplotd', data=(phi_array_pred_plot1[:,0]*phi_norm))  
h5f.create_dataset('Xnew', data=(x_new_array_plot1*a0))  
h5f.create_dataset('Ynew', data=(y_new_array_plot1*a0))  
h5f.create_dataset('Ynewplote', data=(Er_array_true_plot1*(-phi_norm/a0)))
h5f.create_dataset('Ynewplotf', data=(Er_array_pred_plot1*(-phi_norm/a0)))
h5f.close()



time_pred = np.unique(x_t)[N_time] 
time_pred_inds = np.where(x_t == time_pred) 
merged_pred_inds = np.where(x_t == time_pred)
X0 = x_x[merged_pred_inds]/a0
X1 = x_y[merged_pred_inds]/a0 
X3 = x_t[merged_pred_inds]/t_ref
X0 = np.array([X0]).T
X1 = np.array([X1]).T 
X3 = np.array([X3]).T
output_Er_true = grid(x_new_array_plot1*a0, y_new_array_plot1*a0, Er_array_true_plot1, x_plot1, y_plot1)
output_Er_pred = grid(x_new_array_plot1*a0, y_new_array_plot1*a0, Er_array_pred_plot1, x_plot1, y_plot1)
output_Er_pred_std = grid(x_new_array_plot1*a0, y_new_array_plot1*a0, Er_array_pred_plot1_std, x_plot1, y_plot1)
delta_Er_rel_unnorm = (output_Er_pred[2]-output_Er_true[2])

colormap_noise = "RdYlGn"#"RdYlGn_r"
mini, maxi = -17.71,17.71#-10.6,10.6#-14.4,14.4
norm = plt.Normalize(mini, maxi)

i = 0
plt.figure(figsize=(6,7))
im = plt.scatter(output_Er_pred[0],output_Er_pred[1],c=delta_Er_rel_unnorm/output_Er_pred_std[2],cmap=colormap_noise,norm=norm)
plt.xlabel('R (m)')
plt.ylabel('y (m)')
plt.title(r'$\Delta E_r/ \sigma_{PINN} \ (\beta_e \sim 0.2\%)$')
plt.xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
plt.ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
plt.colorbar(im)
plt.savefig(str(save_figs_path)+'relative_Er_error_norm_GKvDRB1even_samescale_avg.png')
plt.savefig(str(save_figs_path)+'relative_Er_error_norm_GKvDRB1even_samescale_avg.eps')
plt.show()

h5f = h5py.File('/home/mathewsa/Plasma-PINN_DRBvGK_manuscript_v0/PlasmaPINN_DRBvGK_Figure3.h5', 'w') 
h5f.create_dataset('X', data=(output_Er_pred[0]))  
h5f.create_dataset('Y', data=(output_Er_pred[1]))  
h5f.create_dataset('Y_PLOT', data=(delta_Er_rel_unnorm/output_Er_pred_std[2]))   
h5f.close()



























#EM high beta: 37003300
start_frac_t = 0.90 #fraction of time to consider
end_frac_t = 0.95
start_frac_x = 0.2
end_frac_x = 0.8
start_frac_y = 0.2
end_frac_y = 0.8
start_frac_z = 5./6.
frac_train = 1.0 #randomly sampled fraction of data to create training set

R0 = 0.85 #m
a0 = 0.5 #m
R_ref = R0 + a0
t_ref =  1.3273e-05 #s

import numpy as np

x = [1.257173698595873,  1.2623503424336309, 1.267526986271389,
     1.2727036301091468, 1.2778802739469048, 1.2830569177846627,
     1.2882335616224208, 1.2934102054601786, 1.2985868492979367,
     1.3037634931356945, 1.3089401369734524, 1.3141167808112104,
     1.3192934246489683, 1.3244700684867263, 1.3296467123244842,
     1.3348233561622422, 1.34,               1.345176643837758,
     1.350353287675516,  1.3555299315132738, 1.360706575351032,
     1.3658832191887897, 1.3710598630265478, 1.3762365068643057,
     1.3814131507020635, 1.3865897945398216, 1.3917664383775794,
     1.3969430822153375, 1.4021197260530953, 1.4072963698908534,
     1.4124730137286112, 1.4176496575663693, 1.4228263014041271]

y = [-0.1456526028082542, -0.1411009589704962, -0.1365493151327383,
     -0.1319976712949803, -0.1274460274572224, -0.1228943836194644,
     -0.1183427397817065, -0.1137910959439486, -0.1092394521061906,
     -0.1046878082684327, -0.1001361644306747, -0.0955845205929168,
     -0.0910328767551588, -0.0864812329174009, -0.081929589079643,
     -0.077377945241885,  -0.0728263014041271, -0.0682746575663691,
     -0.0637230137286112, -0.0591713698908533, -0.0546197260530953,
     -0.0500680822153374, -0.0455164383775794, -0.0409647945398215,
     -0.0364131507020635, -0.0318615068643056, -0.0273098630265477,
     -0.0227582191887897, -0.0182065753510318, -0.0136549315132738,
     -0.0091032876755159, -0.0045516438377579,  0.,
      0.0045516438377579,  0.0091032876755159,  0.0136549315132738,
      0.0182065753510318,  0.0227582191887897,  0.0273098630265476,
      0.0318615068643056,  0.0364131507020635,  0.0409647945398215,
      0.0455164383775794,  0.0500680822153374,  0.0546197260530953,
      0.0591713698908533,  0.0637230137286112,  0.0682746575663691,
      0.0728263014041271,  0.077377945241885,   0.081929589079643,
      0.0864812329174009,  0.0910328767551588,  0.0955845205929168,
      0.1001361644306747,  0.1046878082684327,  0.1092394521061906,
      0.1137910959439485,  0.1183427397817065,  0.1228943836194644,
      0.1274460274572224,  0.1319976712949803,  0.1365493151327383,
      0.1411009589704962,  0.1456526028082542]

z = [-4.,                 -3.7142857142857144, -3.428571428571429,
     -3.142857142857143,  -2.857142857142857,  -2.5714285714285716,
     -2.2857142857142856, -2.,                 -1.7142857142857144,
     -1.4285714285714288, -1.1428571428571432, -0.8571428571428572,
     -0.5714285714285716, -0.285714285714286,   0.,
      0.2857142857142856,  0.5714285714285712,  0.8571428571428568,
      1.1428571428571423,  1.428571428571428,   1.7142857142857135,
      2.,                  2.2857142857142856,  2.571428571428571,
      2.8571428571428568,  3.1428571428571423,  3.428571428571428,
      3.7142857142857135,  4.]

d_x = []
i = 0
while i < len(x) - 1:
    d_x_new = (x[i]+x[i+1])/2.
    d_x.append(d_x_new)
    i = i + 1

d_y = []
i = 0
while i < len(y) - 1:
    d_y_new = (y[i]+y[i+1])/2.
    d_y.append(d_y_new)
    i = i + 1

d_z = []
i = 0
while i < len(z) - 1:
    d_z_new = (z[i]+z[i+1])/2.
    d_z.append(d_z_new)
    i = i + 1

d_t = []
i = 0
while i < 1001:
    d_t_new = float(i)*0.0000001
    d_t.append(d_t_new)
    i = i + 1

sorted_vars = []
len_loop_x = int(end_frac_x*len(d_x))
len_loop_y = int(end_frac_y*len(d_y))
len_loop_z = int(start_frac_z*len(d_z) + 1)
len_loop_t = int(end_frac_t*len(d_t))
print('Stage 3')
i_t = int(start_frac_t*len(d_t))
while i_t < len_loop_t:
    i_x = int(start_frac_x*len(d_x))
    while i_x < len_loop_x:
        i_y = int(start_frac_y*len(d_y))
        while i_y < len_loop_y:
            i_z = int(start_frac_z*len(d_z))
            while i_z < len_loop_z:
                sorted_vars.append([[d_x[i_x]],[d_y[i_y]],[d_z[i_z]],[d_t[i_t]]])
                i_z = i_z + 1
            i_y = i_y + 1
        i_x = i_x + 1
    i_t = i_t + 1
    print(float(i_t)/len_loop_t)

full_vars_array = np.array(sorted_vars)
x_x = full_vars_array[:,0]
x_y = full_vars_array[:,1]
x_z = full_vars_array[:,2]
x_t = full_vars_array[:,3]

a0 = a = 0.5        #.minor radius (m)
Te0    = 40.0          #.Electron temperature (eV).
Ti0    = 40.0           #.Ion temperature (eV).
n0     = n_ref = 7.*(10.**(19.)) #m^-3
B0     = 0.3148 #T, is B0 in centre of simulation domain; 0.5 T on magnetic axis
R0     = 0.85 #m 
R_ref  = R0 + a0
B_ref  = B0*(R0 + a0)/(R0 + a0)        #.Magnetic field (T) on axis?
mime   = 3672.3036           #.Mass ratio = m_i/m_e.
Z      = 1             #.Ionization level.

u     = 1.660539040e-27     #.unifited atomic mass unit (kg).
m_H   = 1.007276*u          #.Hydrogen ion (proton) mass (kg).
mu    = 2.0 #.39.948              #.m_i/m_proton.
m_Ar  = mu*m_H              #.Argon ion (singly ionized?) mass (kg).
m_i   = m_Ar                #.ion mass (kg).
m_e   = 0.910938356e-30     #.electron mass (kg).
c     = 299792458.0         #.speed of light (m/s).
e     = 1.60217662e-19      #.electron charge (C).

cse0   = np.sqrt(e*Te0/m_i)     #.Electron sound speed (m/s).
csi0   = np.sqrt(e*Ti0/m_i)     #.Ion sound speed (m/s).

tRef   = np.sqrt((R_ref*a)/2.0)/cse0 
phi_norm = B_ref*(a**2)/tRef

len_loop_t = int(end_frac_t*len(d_t)) - int(start_frac_t*len(d_t))
len_loop_x = int(end_frac_x*len(d_x)) - int(start_frac_x*len(d_x))
len_loop_y = int(end_frac_y*len(d_y)) - int(start_frac_y*len(d_y))
import matplotlib.pyplot as plt

N_time = 0
len_2d = len_loop_x*len_loop_y
len_skip = len_2d
X0 = x_x[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/a0
X1 = x_y[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/a0
X3 = x_t[int(N_time*len_skip):int(N_time*len_skip + len_2d)]/t_ref
xlim_min = min(X0)[0]#-0.02
xlim_max = max(X0)[0]#0.02
ylim_min = min(X1)[0]#-0.02
ylim_max = max(X1)[0]#0.02

import h5py
data_file = '/nobackup1/mathewsa/DRBvGK/NSTX_gyro_v_DRB-highbetahighres_truecollisionfrequency_EM_proper.h5'
h5f = h5py.File(data_file, "r")
x_array = h5f['x_array'].value 
y_array = h5f['y_array'].value  
t_array = h5f['t_array'].value 
x_new_array = h5f['x_new_array'].value 
y_new_array = h5f['y_new_array'].value  
t_new_array = h5f['t_new_array'].value 
ne_array_true = h5f['ne_array_true'].value 
Te_array_true = h5f['Te_array_true'].value 
phi_array_true = h5f['phi_array_true'].value 
Er_array_true = h5f['Er_array_true'].value 
ne_array_pred = h5f['ne_array_pred'].value 
Te_array_pred = h5f['Te_array_pred'].value 
phi_array_pred = h5f['phi_array_pred'].value 
Er_array_pred = h5f['Er_array_pred'].value 

relative_path = '/results/NSTX_gyro_v_DRB-highbetahighres_truecollisionfrequency_EM_properv99/' #347 MB; too large to upload on Github
save_figs_path = '/home/mathewsa/PINNs/main/gdbh'+str(relative_path)

#y-averaged Er vs x comparison
time_plot_index= time_index = 0
N_model = 0
avg_Er_pred_yvst = []
avg_Er_true_yvst = []
number_of_models = 100
while N_model < number_of_models:
    i_avg = 0
    while i_avg < len_loop_x:
        avg_Er_pred_yvst.append(np.mean(Er_array_pred[N_model*len_loop_t:(N_model+1)*len_loop_t][time_plot_index][i_avg::len_loop_x]))
        avg_Er_true_yvst.append(np.mean(Er_array_true[N_model*len_loop_t:(N_model+1)*len_loop_t][time_plot_index][i_avg::len_loop_x]))
        i_avg = i_avg + 1
    N_model = N_model + 1

avg_Er_pred_yvst = np.array(avg_Er_pred_yvst)
avg_Er_true_yvst = np.array(avg_Er_true_yvst)

all_Er = [] 
N_model = 0
while N_model < number_of_models:
    all_Er.append(avg_Er_pred_yvst[N_model*len_loop_x:(N_model+1)*len_loop_x])
    N_model = N_model + 1  
    
    
all_Er = np.array(all_Er)
mean_Er = np.mean(all_Er, axis=0)
std_Er = np.std(all_Er, axis=0)



plt.figure()
#plt.style.use('seaborn-white')
plt.scatter((x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x])*a0,(avg_Er_true_yvst[0:len_loop_x])*(-phi_norm/a0),label='Gyrokinetic',color='black',marker='x')
plt.plot(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0,mean_Er*(-phi_norm/a0),label='Drift-reduced Braginskii',color='green')
plt.fill_between(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0,(mean_Er+2.*std_Er)*(-phi_norm/a0),(mean_Er-2.*std_Er)*(-phi_norm/a0),interpolate=True,color='green',alpha=0.1)
plt.xlabel('R (m)')
plt.ylabel('$<E_r>_y$ (V/m)')
#plt.title('$t$ = %.2f $\mu$s'%((10.**6.)*tRef*t_array[time_plot_index][0][0]))
plt.xlim(1.2975,1.3825)
plt.legend(frameon=False)
plt.xlim(np.min(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0),np.max(x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x]*a0))
plt.savefig(str(save_figs_path)+'highbetaES_avgEry_vs_x1.png')
plt.savefig(str(save_figs_path)+'highbetaES_avgEry_vs_x1.eps')
plt.show()

h5f = h5py.File('/home/mathewsa/Plasma-PINN_DRBvGK_manuscript_v0/PlasmaPINN_DRBvGK_Figure4.h5', 'w') 
h5f.create_dataset('X', data=((x_new_array[0:len_loop_t][time_plot_index][0:len_loop_x])*a0))  
h5f.create_dataset('Y', data=((avg_Er_true_yvst[0:len_loop_x])*(-phi_norm/a0)))  
h5f.create_dataset('Y_MEAN', data=(mean_Er*(-phi_norm/a0)))  
h5f.create_dataset('Y_STD', data=(std_Er*(-phi_norm/a0)))  
h5f.close()


x_array_plot1 = np.mean(x_array[time_index::number_of_models],axis=0)
y_array_plot1 = np.mean(y_array[time_index::number_of_models],axis=0)
t_array_plot1 = np.mean(t_array[time_index::number_of_models],axis=0)
x_new_array_plot1 = np.mean(x_new_array[time_index::number_of_models],axis=0)
y_new_array_plot1 = np.mean(y_new_array[time_index::number_of_models],axis=0)
t_new_array_plot1 = np.mean(t_new_array[time_index::number_of_models],axis=0)
ne_array_true_plot1 = np.mean(ne_array_true[time_index::number_of_models],axis=0)
Te_array_true_plot1 = np.mean(Te_array_true[time_index::number_of_models],axis=0)
phi_array_true_plot1 = np.mean(phi_array_true[time_index::number_of_models],axis=0)
Er_array_true_plot1 = np.mean(Er_array_true[time_index::number_of_models],axis=0)
ne_array_pred_plot1 = np.mean(ne_array_pred[time_index::number_of_models],axis=0)
Te_array_pred_plot1 = np.mean(Te_array_pred[time_index::number_of_models],axis=0)
phi_array_pred_plot1 = np.mean(phi_array_pred[time_index::number_of_models],axis=0)
Er_array_pred_plot1 = np.mean(Er_array_pred[time_index::number_of_models],axis=0)

ne_array_pred_plot1_std = np.std(ne_array_pred[time_index::number_of_models],axis=0)
Te_array_pred_plot1_std = np.std(Te_array_pred[time_index::number_of_models],axis=0)
phi_array_pred_plot1_std = np.std(phi_array_pred[time_index::number_of_models],axis=0)
Er_array_pred_plot1_std = np.std(Er_array_pred[time_index::number_of_models],axis=0)

from scipy.interpolate import griddata
def grid(x, y, z, input_x, input_y): 
    xi = input_x
    yi = input_y 
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic')
    return X, Y, Z

resolution = 500
colormap = 'inferno'

x_plot1 = np.linspace(np.min(np.unique(x_array_plot1*a0)), np.max(np.unique(x_array_plot1*a0)), resolution) #should be domain smaller than ne_test
y_plot1 = np.linspace(np.min(np.unique(y_array_plot1*a0)), np.max(np.unique(y_array_plot1*a0)), resolution) #should be domain smaller than Te_test


i = 0
fig, ax = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row', figsize=(11,7))
for i in range(2):
    for j in range(3):
        x_plot1 = np.linspace(np.min(np.unique(x_array_plot1*a0)), np.max(np.unique(x_array_plot1*a0)), resolution) #should be domain smaller than ne_test
        y_plot1 = np.linspace(np.min(np.unique(y_array_plot1*a0)), np.max(np.unique(y_array_plot1*a0)), resolution) #should be domain smaller than Te_test
        if i == 0:
            if j == 0:
                output = grid(x_array_plot1[:,0]*a0, y_array_plot1[:,0]*a0, ne_array_true_plot1[:,0]*n0, x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap='YlOrRd_r')
                ax[i, j].set_ylabel('y (m)')
                ax[i, j].set_title('$n_e$ (m$^{-3}$)')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])#, pad=0.1)
            if j == 1:
                output = grid(x_array_plot1[:,0]*a0, y_array_plot1[:,0]*a0, phi_array_true_plot1[:,0]*phi_norm, x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap=colormap)
                ax[i, j].set_title(r'$\phi$ (V): gyrokinetic')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])
            if j == 2:
                output = grid(x_new_array_plot1*a0, y_new_array_plot1*a0, Er_array_true_plot1*(-phi_norm/a0), x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap=colormap)
                ax[i, j].set_title(r'$E_r$ (V/m): gyrokinetic')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])
        if i == 1:
            if j == 0:
                output = grid(x_array_plot1[:,0]*a0, y_array_plot1[:,0]*a0, Te_array_true_plot1[:,0]*Te0, x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap='YlOrRd_r')        
                ax[i, j].set_xlabel('R (m)')
                ax[i, j].set_ylabel('y (m)')
                ax[i, j].set_title('$T_e$ (eV)')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])
            if j == 1:
                output = grid(x_array_plot1[:,0]*a0, y_array_plot1[:,0]*a0, phi_array_pred_plot1[:,0]*phi_norm, x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap=colormap)
                ax[i, j].set_xlabel('R (m)')
                ax[i, j].set_title(r'$\phi$ (V): drift-reduced Braginskii')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])
            if j == 2:
                output = grid(x_new_array_plot1*a0, y_new_array_plot1*a0, Er_array_pred_plot1*(-phi_norm/a0), x_plot1, y_plot1)
                im = ax[i, j].scatter(output[0],output[1],c=output[2],cmap=colormap)
                ax[i, j].set_xlabel('R (m)')
                ax[i, j].set_title(r'$E_r$ (V/m): drift-reduced Braginskii')
                ax[i, j].set_xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
                ax[i, j].set_ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
                fig.colorbar(im, ax=ax[i, j])
        
        
plt.savefig(str(save_figs_path)+'EM_high_beta_GKvDRB_properavg.png')
plt.savefig(str(save_figs_path)+'EM_high_beta_GKvDRB_properavg.eps')
plt.show()


h5f = h5py.File('/home/mathewsa/Plasma-PINN_DRBvGK_manuscript_v0/PlasmaPINN_DRBvGK_Figure5.h5', 'w') 
h5f.create_dataset('X', data=(x_array_plot1[:,0]*a0))  
h5f.create_dataset('Y', data=(y_array_plot1[:,0]*a0))  
h5f.create_dataset('Yplota', data=(ne_array_true_plot1[:,0]*n0))  
h5f.create_dataset('Yplotb', data=(Te_array_true_plot1[:,0]*Te0))  
h5f.create_dataset('Yplotc', data=(phi_array_true_plot1[:,0]*phi_norm))  
h5f.create_dataset('Yplotd', data=(phi_array_pred_plot1[:,0]*phi_norm))  
h5f.create_dataset('Xnew', data=(x_new_array_plot1*a0))  
h5f.create_dataset('Ynew', data=(y_new_array_plot1*a0))  
h5f.create_dataset('Ynewplote', data=(Er_array_true_plot1*(-phi_norm/a0)))
h5f.create_dataset('Ynewplotf', data=(Er_array_pred_plot1*(-phi_norm/a0)))
h5f.close()





output_Er_true = grid(x_new_array_plot1*a0, y_new_array_plot1*a0, Er_array_true_plot1, x_plot1, y_plot1)
output_Er_pred = grid(x_new_array_plot1*a0, y_new_array_plot1*a0, Er_array_pred_plot1, x_plot1, y_plot1)
output_Er_pred_std = grid(x_new_array_plot1*a0, y_new_array_plot1*a0, Er_array_pred_plot1_std, x_plot1, y_plot1)
delta_Er_rel_unnorm = (output_Er_pred[2]-output_Er_true[2])
colormap_noise = "RdYlGn"
mini, maxi = -17.71,17.71
norm = plt.Normalize(mini, maxi)

i = 0
plt.figure(figsize=(6,7))
im = plt.scatter(output_Er_pred[0],output_Er_pred[1],c=delta_Er_rel_unnorm/output_Er_pred_std[2],cmap=colormap_noise,norm=norm)
plt.xlabel('R (m)')
plt.ylabel('y (m)')
plt.title(r'$\Delta E_r/ \sigma_{PINN} \ (\beta_e \sim 2.0\%)$')
plt.xlim(1.2975,1.3825)#(xlim_min*a0 + 0.005,xlim_max*a0 - 0.005)
plt.ylim(-0.084,0.084)#(ylim_min*a0,ylim_max*a0)
plt.colorbar(im)
plt.savefig(str(save_figs_path)+'relative_Er_error_norm_GKvDRB1even_samescale_avg.png')
plt.savefig(str(save_figs_path)+'relative_Er_error_norm_GKvDRB1even_samescale_avg.eps')
plt.show()


h5f = h5py.File('/home/mathewsa/Plasma-PINN_DRBvGK_manuscript_v0/PlasmaPINN_DRBvGK_Figure3_high_beta.h5', 'w') 
h5f.create_dataset('X', data=(output_Er_pred[0]))  
h5f.create_dataset('Y', data=(output_Er_pred[1]))  
h5f.create_dataset('Y_PLOT', data=(delta_Er_rel_unnorm/output_Er_pred_std[2]))   
h5f.close()
