
import os
import sys, getopt, optparse
import pickle
#import dill as pickle
sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
import time
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#cmap = plt.cm.jet

# import general simulation utilities
from data_utils import DataLoader
from pff_rnn import PFF_RNN

################################################################################

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

model_topdir = "../exp/"
out_dir = "../exp/"
data_dir = "../data/mnist/"
split = "test"
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["data_dir=","model_topdir=",
                                                      "gpu_id=","n_trials=",
                                                      "out_dir=","split="])
# Collect arguments from argv
n_trials = 1
gpu_id = -1
for opt, arg in options:
    if opt in ("--data_dir"):
        data_dir = arg.strip()
    elif opt in ("--model_topdir"):
        model_topdir = arg.strip()
    elif opt in ("--split"):
        split = arg.strip()
    elif opt in ("--out_dir"):
        out_dir = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())
    elif opt in ("--n_trials"):
        n_trials = int(arg.strip())
print(" Exp out dir: ",out_dir)

mid = gpu_id # 0
if mid >= 0:
    print(" > Using GPU ID {0}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
    #gpu_tag = '/GPU:0'
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

print(" >>> Run sim on {} w/ GPU {}".format(data_dir, mid))

dev_batch_size = 157 # dev batch size
print(" > Loading test set")
xfname = "{}/{}X.npy".format(data_dir, split)
yfname = "{}/{}Y.npy".format(data_dir, split)
Xdev = ( tf.cast(np.load(xfname, allow_pickle=True),dtype=tf.float32) )

if len(Xdev.shape) > 2:
    Xdev = tf.reshape(Xdev, [Xdev.shape[0], Xdev.shape[1] * Xdev.shape[2] * Xdev.shape[3]])

print("Xdev.shape = ",Xdev.shape)
max_val = float(tf.reduce_max(Xdev))
if max_val > 1.0:
    Xdev = Xdev/max_val

nC = 10
Ydev = ( tf.cast(np.load(yfname, allow_pickle=True),dtype=tf.float32) )
if len(Ydev.shape) == 1:
    #nC = Y.shape[1]
    Ydev = tf.one_hot(Ydev.numpy().astype(np.int32), depth=nC)
elif Ydev.shape[1] == 1:
    #nC = 10
    Ydev = tf.one_hot(tf.squeeze(Ydev).numpy().astype(np.int32), depth=nC)
print("Ydev.shape = ",Ydev.shape)

devset = DataLoader(design_matrices=[("x",Xdev.numpy()), ("y",Ydev.numpy())],
                                     batch_size=dev_batch_size, disable_shuffle=True)

# 3 channel
def plot_img_grid(samples, fname, nx, ny, px, py, plt, rotNeg90=False): # rows, cols,...
    
    #Visualizes a matrix of vector patterns in the form of an image grid plot.
    samples_ = samples.reshape([samples.shape[0],32,32,3])
    #samples = samples_[:,:,:,0]*0.33 + samples_[:,:,:,1]*0.33 + samples_[:,:,:,2]*0.33
    
    px_dim = px
    py_dim = py
    #canvas = np.empty((px_dim*nx, py_dim*ny))
    canvas = np.empty((px_dim*nx, py_dim*ny, 3))
    ptr = 0
    for i in range(0,nx,1):
        for j in range(0,ny,1):
            #xs = tf.expand_dims(tf.cast(samples[ptr,:],dtype=tf.float32),axis=0)
         
            #######
            ###xs = np.expand_dims(samples[ptr,:],axis=0)
            #xs = xs.numpy() #tf.make_ndarray(x_mean)
            ###xs = xs[0].reshape(px_dim, py_dim)
            #####
            
            xs = samples_[i*nx+j]
            if rotNeg90 is True:
                xs = np.rot90(xs, -1)
            for c in range(3):
                canvas[(nx-i-1)*px_dim:(nx-i)*px_dim, j*py_dim:(j+1)*py_dim,c] = xs[:,:,c]
            ptr += 1
    plt.figure(figsize=(12, 14))
    plt.imshow(canvas, origin="upper")#, cmap="gray")
    plt.tight_layout()
    plt.axis('off')
    #print(" SAVE: {0}{1}".format(out_dir,"gmm_decoded_samples.jpg"))
    plt.savefig("{0}".format(fname), bbox_inches='tight', pad_inches=0)
    plt.clf()

'''
def plot_img_grid(samples, fname, nx, ny, px, py, plt, rotNeg90=False): # rows, cols,...
    
    px_dim = px
    py_dim = py
    canvas = np.empty((px_dim*nx, py_dim*ny))
    ptr = 0
    for i in range(0,nx,1):
        for j in range(0,ny,1):
            #xs = tf.expand_dims(tf.cast(samples[ptr,:],dtype=tf.float32),axis=0)
            xs = np.expand_dims(samples[ptr,:],axis=0)
            #xs = xs.numpy() #tf.make_ndarray(x_mean)
            xs = xs[0].reshape(px_dim, py_dim)
            if rotNeg90 is True:
                xs = np.rot90(xs, -1)
            canvas[(nx-i-1)*px_dim:(nx-i)*px_dim, j*py_dim:(j+1)*py_dim] = xs
            ptr += 1
    plt.figure(figsize=(12, 14))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.axis('off')
    #print(" SAVE: {0}{1}".format(out_dir,"gmm_decoded_samples.jpg"))
    plt.savefig("{0}".format(fname), bbox_inches='tight', pad_inches=0)
    plt.clf()
'''
def classify(agent, x):
    K_low = int(agent.K/2) - 1 # 3
    K_high = int(agent.K/2) + 1 # 5
    x_ = x
    Ey = None
    z_lat = agent.forward(x_) # do forward init pass
    for i in range(agent.y_dim):
        z_lat_ = []
        for ii in range(len(z_lat)):
            z_lat_.append(z_lat[ii] + 0)

        yi = tf.ones([x.shape[0],agent.y_dim]) * tf.expand_dims(tf.one_hot(i,depth=agent.y_dim),axis=0)

        gi = 0.0
        for k in range(K_high):
            z_lat_, p_g = agent._step(x_, yi, z_lat_, thr=0.0)
            if k >= K_low and k <= K_high: # only keep goodness in middle iterations
                gi = ((p_g[0] + p_g[1])*0.5) + gi

        if i > 0:
            Ey = tf.concat([Ey,gi],axis=1)
        else:
            Ey = gi

    Ey = Ey / (3.0)
    y_hat = tf.nn.softmax(Ey)
    return y_hat, Ey

def eval(agent, dataset, debug=False):
    '''
    Evaluates the current state of the agent given a dataset (data-loader).
    '''
    N = 0.0
    Ny = 0.0
    Acc = 0.0
    Ly = 0.0
    Lx = 0.0
    tt = 0
    #debug = True
    ind = 0
    for batch in dataset:
        _, x = batch[0]
        #_, y_tag = batch[1]
        _, y = batch[1]
        N += x.shape[0]
        Ny += float(tf.reduce_sum(y))


        y_hat, Ey = classify(agent, x)
        Ly += tf.reduce_sum(-tf.reduce_sum(y * tf.math.log(y_hat), axis=1, keepdims=True))

        z_lat = agent.forward(x)
        x_hat = agent.sample(z=z_lat[len(z_lat)-2])

        ex = x_hat - x
        Lx += tf.reduce_sum(tf.reduce_sum(tf.math.square(ex),axis=1,keepdims=True))

        if debug == True:
            print("------------------------")
            print(Ey[0:4,:])
            print(y_hat[0:4,:])
            print("------------------------")

        #y_m = tf.squeeze(y_m)
        y_ind = tf.cast(tf.argmax(y,1),dtype=tf.int32)
        y_pred = tf.cast(tf.argmax(Ey,1),dtype=tf.int32)
        comp = tf.cast(tf.equal(y_pred,y_ind),dtype=tf.float32) #* y_m
        '''
        print("------------------------")
        for i in range(10):
            y_i = y_ind[y_ind == i]
            y_i_pred = y_pred[y_ind == i]
            comp_i = tf.reduce_sum(tf.cast(tf.equal(y_i,y_i_pred),dtype=tf.float32))
            n = y_i.shape[0]
            print("number {} Acc = {}".format(i, comp_i/n))
        print("------------------------")
        '''
        Acc += tf.reduce_sum( comp )
        print("\r Lx = {}  Ly = {}  Acc = {} ({} samples)".format(Lx/Ny,Ly/Ny,Acc/Ny,Ny))#,end=" ")
    print()

    
    fname = "{}/x_g.png".format(out_dir)
    plot_img_grid(x_hat.numpy(), fname, nx=12, ny=12, px=32, py=32, plt=plt)
    plt.close()

    
    fname = "{}/x_data.png".format(out_dir)
    plot_img_grid(x, fname, nx=12, ny=12, px=32, py=32, plt=plt)
    plt.close()
    

    Ly = Ly/Ny
    Acc = Acc/Ny
    Lx = Lx/Ny

    return Ly, Acc, Lx

print("----")
with tf.device(gpu_tag):

    best_acc_list = []
    acc_list = []
    for tr in range(n_trials):
        ########################################################################
        ## load model
        ########################################################################
        agent = PFF_RNN(model_dir="{}trial{}/".format(model_topdir, tr))
        K = agent.K
        ########################################################################

        ########################################################################
        Ly, Acc, Lx = eval(agent, devset, False)
        print("{}: L {} Acc = {}  Lx {}".format(-1, Ly, Acc, Lx))

        acc_list.append(1.0 - Acc)

    ############################################################################
    ## calc post-trial statistics
    n_dec = 4
    mu = round(np.mean(np.asarray(acc_list)), n_dec)
    sd = round(np.std(np.asarray(acc_list)), n_dec)
    print("  Test.Error = {:.4f} \pm {:.4f}".format(mu, sd))

    ## store result to disk just in case...
    results_fname = "{}/test_results.txt".format(out_dir)
    log_t = open(results_fname,"a")
    log_t.write("Generalization Results:\n")
    log_t.write("  Test.Error = {:.4f} \pm {:.4f}\n".format(mu, sd))
    log_t.close()
