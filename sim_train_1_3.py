
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
import matplotlib.image as mpimg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
#cmap = plt.cm.jet

# import general simulation utilities
from data_utils import DataLoader
IMAGE_SIZE = 32 #color:32, fMNIST,MNIST:28, RPS:32
def plot_img_grid_gray(samples, fname, nx, ny, px, py, plt, rotNeg90=False): # rows, cols,...
    
    #Visualizes a matrix of vector patterns in the form of an image grid plot.
    samples_ = samples.reshape([samples.shape[0],IMAGE_SIZE,IMAGE_SIZE])
    px_dim = px
    py_dim = py

    canvas = np.empty((px_dim*nx, py_dim*ny))
    for i in range(0,nx,1):
        for j in range(0,ny,1):
            xs = samples_[i*nx+j]
            if rotNeg90 is True:
                xs = np.rot90(xs, -1)
            canvas[(nx-i-1)*px_dim:(nx-i)*px_dim, j*py_dim:(j+1)*py_dim] = xs[:,:]
           
    plt.figure(figsize=(12, 14))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("{0}".format(fname), bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_img_grid(samples, fname, nx, ny, px, py, plt, rotNeg90=False): # rows, cols,...
    
    #Visualizes a matrix of vector patterns in the form of an image grid plot.
    samples_ = samples.reshape([samples.shape[0],IMAGE_SIZE,IMAGE_SIZE,3])
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
            #for c in range(4):
                canvas[(nx-i-1)*px_dim:(nx-i)*px_dim, j*py_dim:(j+1)*py_dim,c] = xs[:,:,c]
            ptr += 1
    plt.figure(figsize=(12, 14))
    plt.imshow(canvas, origin="upper")#, cmap="gray")
    plt.tight_layout()
    plt.axis('off')
    #print(" SAVE: {0}{1}".format(out_dir,"gmm_decoded_samples.jpg"))
    plt.savefig("{0}".format(fname), bbox_inches='tight', pad_inches=0)
    plt.clf()

from pff_rnn_3_y_entropy_newg import PFF_RNN
layers = 3
################################################################################

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

out_dir = "../exp/"
data_dir = "../data/mnist/"
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["data_dir=","gpu_id=","n_trials=","out_dir="])
# Collect arguments from argv
n_trials = 1
gpu_id = -1
for opt, arg in options:
    if opt in ("--data_dir"):
        data_dir = arg.strip()
    elif opt in ("--out_dir"):
        out_dir = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())
    elif opt in ("--n_trials"):
        n_trials = int(arg.strip())
print(" Exp out dir: ",out_dir)
# gpu check
mid = gpu_id # 0
#print("CUDA Available",tf.test.is_built_with_cuda())
if mid >= 0:
    print(" > Using GPU ID {}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(mid)
    #gpu_tag = '/GPU:0'
    gpu_tag = '/GPU:{}'.format(gpu_id)
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

print(" >>> Run sim on {} w/ GPU {}".format(data_dir, mid))

nC = 10
path = 'dataset/cifar10/'
X = []
Xdev = []
Y = []
Ydev = []
print(" > Loading train and dev set")


#数据集区分了train和test
lim = 0.6#0.2
with tf.compat.v1.Session() as sess:
    for i in range(nC):
        train_path = path + 'train/' + str(i) + '/'
        test_path = path + 'test/' + str(i) + '/'
        # train
        filepaths = os.listdir(train_path)
        l = int(len(filepaths) * lim)
        #for j in range(l):
        for j in range(l):
            img = cv2.imread(train_path+filepaths[j])#, cv2.IMREAD_GRAYSCALE) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #color
            resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC) #color
            #resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR) #gray
            X.append(resized_img)
            Y.append(i)
        # test
        filepaths = os.listdir(test_path)
        l = int(len(filepaths) * lim)
        for j in range(l):
            img = cv2.imread(test_path+filepaths[j])#, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #color
            resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC) #color
            #resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR) #gray
            Xdev.append(resized_img)
            Ydev.append(i)

'''
# 数据集没有区分train/test
ratio = 0.2
lim = 1.0
with tf.compat.v1.Session() as sess:
    for i in range(nC):
        class_path = path + str(i) + '/'
        filepaths = os.listdir(class_path)
        #l = len(filepaths)
        l = int(len(filepaths) * lim)
        test_num = int(ratio*l + 1)
        train_num = l - test_num
        for j in range(l):
            img = cv2.imread(class_path+filepaths[j])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            if j < train_num:
                X.append(resized_img)
                Y.append(i)
            else:
                Xdev.append(resized_img)
                Ydev.append(i)
'''
# shuffle 
permutation = np.random.permutation(len(Y))
permutation_dev = np.random.permutation(len(Ydev))
X = tf.cast(np.array(X)[permutation], dtype=tf.float32)
Y = tf.cast(np.array(Y)[permutation],dtype=tf.float32)
Xdev = tf.cast(np.array(Xdev)[permutation_dev], dtype=tf.float32)
Ydev = tf.cast(np.array(Ydev)[permutation_dev],dtype=tf.float32)

print('X shape:',X.shape)
print('Xdev shape:',Xdev.shape)


# train dataset
width = X.shape[1]
if len(X.shape) == 4:
    X = tf.reshape(X, [X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]])
    print(X.shape)
elif len(X.shape) == 3:
    X = tf.reshape(X, [X.shape[0], X.shape[1] * X.shape[2]])
    print(X.shape)
x_dim = X.shape[1]
max_val = float(tf.reduce_max(X))
if max_val > 1.0:
    X = X/max_val

if len(Y.shape) == 1:
    #nC = Y.shape[1]
    Y = tf.one_hot(Y.numpy().astype(np.int32), depth=nC)
elif Y.shape[1] == 1:
    #nC = 10
    Y = tf.one_hot(tf.squeeze(Y).numpy().astype(np.int32), depth=nC)
y_dim = Y.shape[1]
print("Y.shape = ",Y.shape)

# dev dataset
if len(Xdev.shape) == 4:
    Xdev = tf.reshape(Xdev, [Xdev.shape[0], Xdev.shape[1] * Xdev.shape[2] * Xdev.shape[3]])
elif len(Xdev.shape) == 3:
    Xdev = tf.reshape(Xdev, [Xdev.shape[0], Xdev.shape[1] * Xdev.shape[2]])
print("Xdev.shape = ",Xdev.shape)
max_val = float(tf.reduce_max(Xdev))
if max_val > 1.0:
    Xdev = Xdev/max_val

if len(Ydev.shape) == 1:
    #nC = Y.shape[1]
    Ydev = tf.one_hot(Ydev.numpy().astype(np.int32), depth=nC)
elif Ydev.shape[1] == 1:
    Ydev = tf.one_hot(tf.squeeze(Ydev).numpy().astype(np.int32), depth=nC)
print("Ydev.shape = ",Ydev.shape)


n_iter = 1000 #100 #60 # number of training iterations
batch_size = 512 #512 # batch size
dev_batch_size = 256 #256 # dev batch size #
dataset = DataLoader(design_matrices=[("x",X.numpy()),("y",Y.numpy())], batch_size=batch_size)
devset = DataLoader(design_matrices=[("x",Xdev.numpy()), ("y",Ydev.numpy())],
                                     batch_size=dev_batch_size, disable_shuffle=True)


# 用thr(=0) - delta（goodness） 衡量（正样本delta越小，负样本delta越大
# 实质上使用-goodness衡量，正样本-goodness越大，负样本越小
def classify(agent, x, layers):
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
                p_g_delta = 0
                for layer in range(layers):
                    p_g_delta += p_g[layer]
                #gi = ((p_g[0] + p_g[1])*0.5) + gi
                gi = (p_g_delta/(layers)) + gi

        if i > 0:
            Ey = tf.concat([Ey,gi],axis=1)
        else:
            Ey = gi

    #Ey = Ey / (3.0)
    Ey = Ey / (K_high - K_low + 1)
    y_hat = tf.nn.softmax(Ey)
    return y_hat, Ey


def eval(agent, dataset, t, width, layers, debug=False, save_img=True, out_dir=""):
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
    for batch in dataset:
        _, x = batch[0]
        _, y = batch[1]
        N += x.shape[0]
        Ny += float(tf.reduce_sum(y))

        y_hat, Ey = classify(agent, x, layers)
        Ly += tf.reduce_sum(-tf.reduce_sum(y * tf.math.log(y_hat), axis=1, keepdims=True))
        
        z_lat = agent.forward(x)
        # last 2 layer
        x_hat = agent.sample(z=z_lat)
        ex = x_hat - x
        Lx += tf.reduce_sum(tf.reduce_sum(tf.math.square(ex),axis=1,keepdims=True))
        
        if debug == True:
            print("------------------------")
            print(Ey[0:4,:])
            print(y_hat[0:4,:])
            print("------------------------")

        y_ind = tf.cast(tf.argmax(y,1),dtype=tf.int32)
        y_pred = tf.cast(tf.argmax(Ey,1),dtype=tf.int32)
        comp = tf.cast(tf.equal(y_pred,y_ind),dtype=tf.float32) #* y_m
        Acc += tf.reduce_sum( comp )
    
    Ly = Ly/N
    Acc = Acc/N
    Lx = Lx/N

    if save_img == True:
        
        if t < 0 :
            fname = "{}/x_best_g_samples_{}.png".format(out_dir,t*(-1))
            plot_img_grid(x_hat.numpy(), fname, nx=10, ny=10, px=width, py=width, plt=plt)
            plt.close()

            fname = "{}/x_data.png".format(out_dir)
            plot_img_grid(x, fname, nx=10, ny=10, px=width, py=width, plt=plt)
            plt.close()
        

    return Ly, Acc, Lx

print("----")
with tf.device(gpu_tag):

    best_acc_list = []
    acc_list = []
    Lx_list = []
    print('##############################')
    print('########## layer {} ##########'.format(layers))
    print('##############################')
    for tr in range(n_trials):
        acc_scores = [] # tracks acc during training w/in a trial
        ########################################################################
        ## create model
        ########################################################################
        model_dir = "{}/trial{}/".format(out_dir, tr)
        generater_dir = "{}/generater{}/".format(out_dir, tr)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        args = {"x_dim": x_dim,
                "y_dim": y_dim,
                "n_units": 2000, #2000
                "K":10,#12
                "thr": 10.0, #10.0
                "eps_r": 0.015, #0.01
                "eps_g": 0.015} #0.025
        load_flag = False
        if load_flag == False:
            agent = PFF_RNN(args=args)
        else:
            #agent = PFF_RNN(model_dir=generater_dir)
            agent = PFF_RNN(model_dir=model_dir)
        ## set up optimization
        eta = 0.00025#0.00025 # 0.0005 # for grnn
        reg_lambda = 0
        #reg_lambda = 0.0001 # works nice for kmnist
        opt = tf.keras.optimizers.Adam(eta)

        g_eta = 0.00025#0.0001#0.00025 # 0.0005 #0.001 #0.0005 #0.001
        g_reg_lambda = 0
        g_opt = tf.keras.optimizers.Adam(g_eta)
        ########################################################################

        ########################################################################
        ## begin simulation
        ########################################################################
        Ly, Acc, Lx = eval(agent, devset, -1, width, layers, out_dir=model_dir)
        acc_scores.append(Acc)
        print("{}: L {} Acc = {}  Lx {}".format(-1, Ly, Acc, Lx))

        best_Acc = Acc  #best test acc
        best_Ly = Ly
        best_Lx = Lx
        best_Lx_train = 10000

        flag = False
        g_flag = True #False
        
        moment = 1000
        Lx_thresh_1 = 90  #stage 1 to stage 2
        Lx_thresh_2 = 60  #stage 2 to stage 3
        
        for t in range(n_iter):
            N = 0.0
            Ng = 0.0
            Lg = 0.0
            Ly = 0.0
            L_gen = 0.0
            Acc_train = 0.0
            cnt = 0
            h_flag = False
            for batch in dataset:
                _, x = batch[0]
                _, y = batch[1]
                N += x.shape[0]
                
                # create negative data (x, y_neg)
                if t > moment:
                    # only 回传生成作为输入
                    if Lx < Lx_thresh_2:
                    #if Lx_train < Lx_thresh_2:
                        # 对半分
                        x_pos = x_hat#[:int(x_hat.shape[0]/2)]
                        x_neg = x_hat#[int(x_hat.shape[0]/2):]

                        y_neg = tf.random.uniform(y.shape, 0.0, 1.0) * (1.0 - y)
                        y_neg = tf.one_hot(tf.argmax(y_neg,axis=1), depth=agent.y_dim)
                        
                        x_ = tf.concat([x_pos,x_neg],axis=0)
                        y_ = tf.concat([y,y_neg],axis=0)

                        lab = tf.concat([tf.ones([x_pos.shape[0],1]),tf.zeros([x_neg.shape[0],1])],axis=0)
                        lab_pos = x_pos.shape[0]
                        lab_neg = x_neg.shape[0]

                        flag = True

                    elif Lx < Lx_thresh_1:
                    #elif Lx_train < Lx_thresh_1:
                        # 回传 mix 原数据集 作为输入
                        x_pos_0 = x[:int(x.shape[0]/2)]
                        x_pos_1 = x_hat[int(x.shape[0]/2):]
                        x_pos = tf.concat([x_pos_0,x_pos_1],axis=0)
                        
                        x_neg = x

                        y_neg = tf.random.uniform(y.shape, 0.0, 1.0) * (1.0 - y)
                        y_neg = tf.one_hot(tf.argmax(y_neg,axis=1), depth=agent.y_dim)
                        
                        x_ = tf.concat([x_pos,x_neg],axis=0)
                        y_ = tf.concat([y,y_neg],axis=0)

                        lab = tf.concat([tf.ones([x_pos.shape[0],1]),tf.zeros([x_neg.shape[0],1])],axis=0)
                        lab_pos = x_pos.shape[0]
                        lab_neg = x_neg.shape[0]

                        flag = True

                    else:
                    # 原数据集
                        x_neg = x
                        y_neg = tf.random.uniform(y.shape, 0.0, 1.0) * (1.0 - y)
                        y_neg = tf.one_hot(tf.argmax(y_neg,axis=1), depth=agent.y_dim)
                        
                        x_ = tf.concat([x,x_neg],axis=0)
                        y_ = tf.concat([y,y_neg],axis=0)
                        
                        lab = tf.concat([tf.ones([x.shape[0],1]),tf.zeros([x_neg.shape[0],1])],axis=0)
                        lab_pos = x.shape[0]
                        lab_neg = x_neg.shape[0]

                        flag = False

                else:
                # 原数据集
                    x_neg = x
                    
                    y_neg = tf.random.uniform(y.shape, 0.0, 1.0) * (1.0 - y)
                    y_neg = tf.one_hot(tf.argmax(y_neg,axis=1), depth=agent.y_dim)
                    
                    x_ = tf.concat([x,x_neg],axis=0)
                    y_ = tf.concat([y,y_neg],axis=0)
                    
                    lab = tf.concat([tf.ones([x.shape[0],1]),tf.zeros([x_neg.shape[0],1])],axis=0)
                    lab_pos = x.shape[0]
                    lab_neg = x_neg.shape[0]

                    flag = False

                ## update model given full batch
                z_lat = agent.forward(x_)
                #z_lat_ = agent.ff_with_y(x_, y_)

                # both train
                Lg_t, _, Lgen_t, x_hat,z_lat_ = agent.infer(x_, y_, lab, z_lat, agent.K, g_flag, flag, x, opt, g_opt, reg_lambda=reg_lambda, g_reg_lambda=g_reg_lambda) # total goodness
                
                ## functional entropy
                ## Ent
                if h_flag == False:
                    h_i = np.sum(np.square(np.array(z_lat_)),axis = 2)
                    h_i_pos = h_i[:, :lab_pos]
                    h_i_neg = h_i[:, lab_pos:]
                    h_flag = True
                    
                # 不是第一次
                h_i_pos = np.concatenate((h_i_pos, np.sum(np.square(np.array(z_lat_)),axis = 2)[:, :lab_pos]), axis = 1)
                h_i_neg = np.concatenate((h_i_neg, np.sum(np.square(np.array(z_lat_)),axis = 2)[:, lab_pos:]), axis = 1)
                
                Lg_t = Lg_t * (x.shape[0] + x_neg.shape[0])

                #y_hat = y_hat[0:x.shape[0],:]
                #y_hat = agent.classify(x)
                y_hat, Ey = classify(agent, x, layers)

                Ly_t = tf.reduce_sum(-tf.reduce_sum(y * tf.math.log(y_hat), axis=1, keepdims=True))

                ## track losses
                Ly = Ly_t + Ly
                Lg = Lg_t + Lg
                #Lgen_t = Lgen_t/x_hat.shape[1]
                L_gen = Lgen_t + L_gen
                y_ind = tf.cast(tf.argmax(y,1),dtype=tf.int32)
                y_pred = tf.cast(tf.argmax(y_hat,1),dtype=tf.int32)
                comp = tf.cast(tf.equal(y_pred,y_ind),dtype=tf.float32) #* y_m
                acc_train = tf.reduce_sum( comp )/x.shape[0]
                Acc_train += acc_train
                cnt+=1
                Ng += (x.shape[0] + x_neg.shape[0])
                
                ex_train = x_hat - x_[:x.shape[0]]
                Lx_train = tf.reduce_sum(tf.reduce_sum(tf.math.square(ex_train),axis=1,keepdims=True))/x.shape[0]
                if Lx_train < best_Lx_train:
                    best_Lx_train = Lx_train
                    # train generation
                    fname = "{}/train/x_best_g_samples_{}.png".format(out_dir,t)
                    plot_img_grid(x_hat.numpy(), fname, nx=10, ny=10, px=width, py=width, plt=plt)
                    plt.close()

                    fname = "{}/train/x_data_{}.png".format(out_dir,t)
                    plot_img_grid(x, fname, nx=10, ny=10, px=width, py=width, plt=plt)
                    plt.close()

                print("\r   {}: Ly = {}  Acc_train = {} L_gen = {} w/ {} samples".format(t, Ly/N, acc_train, Lx_train, N), end="")
                
                # Acc file
                f = open('acc/avg_train_acc.txt','a+')
                f.write(str(acc_train.numpy()))
                f.write(',')
                f.write('\n')
                f.close() 
            print()

            # positive
            E_xy_pos = np.mean(h_i_pos, axis = 1) #### E_xy 对所有样本求期望
            E_z_pos = np.mean(E_xy_pos) #### 再对层求期望
            Ent_1_pos = np.mean(E_xy_pos * np.log(E_xy_pos/E_z_pos))
            ## Ent_2
            Ent_2_i_pos = np.zeros(layers)
            for layer_i in range(layers):
                Ent_2_i_pos[layer_i] = np.mean(h_i_pos[layer_i] * np.log(h_i_pos[layer_i]/E_xy_pos[layer_i]))
            Ent_2_pos = np.mean(Ent_2_i_pos)
            # Ent
            print('Positive Ent_1 = {}, Ent_2 = {}'.format(Ent_1_pos, Ent_2_pos))
            Ent_pos = Ent_1_pos + Ent_2_pos
            # file
            f = open('entropy/pos_{}.txt'.format(layers),'a+')
            f.write(str(Ent_pos))
            f.write(',')
            f.write('\n')
            f.close()

            # negative
            E_xy_neg = np.mean(h_i_neg, axis = 1) #### E_xy 对所有样本求期望
            E_z_neg = np.mean(E_xy_neg) #### 再对层求期望
            Ent_1_neg = np.mean(E_xy_neg * np.log(E_xy_neg/E_z_neg))
            ## Ent_2
            Ent_2_i_neg = np.zeros(layers)
            for layer_i in range(layers):
                Ent_2_i_neg[layer_i] = np.mean(h_i_neg[layer_i] * np.log(h_i_neg[layer_i]/E_xy_neg[layer_i]))
            Ent_2_neg = np.mean(Ent_2_i_neg)
            # Ent
            print('Negative Ent_1 = {}, Ent_2 = {}'.format(Ent_1_neg, Ent_2_neg))
            Ent_neg = Ent_1_neg + Ent_2_neg
            # file
            f = open('entropy/neg_{}.txt'.format(layers),'a+')
            f.write(str(Ent_neg))
            f.write(',')
            f.write('\n')
            f.close()


            Ly, Acc, Lx = eval(agent, devset, t, width, layers, out_dir=model_dir)
            acc_scores.append(Acc)
            Lx_list.append(Lx)
            np.save("{}/dev_acc.npy".format(model_dir), np.asarray(acc_scores))
            print("{} test : L {} Acc = {}  Lx {}".format(t, Ly, Acc, Lx))
            # Acc file
            f = open('acc/test_acc.txt','a+')
            f.write(str(Acc.numpy()))
            f.write(',')
            f.write('\n')
            f.close()

            Acc_check_pt = Acc#Acc_train/cnt
            if Acc_check_pt > best_Acc:
                best_Acc = Acc_check_pt
                best_Ly = Ly

                print(" >> Saving model to:  ",model_dir)
                agent.save_model(model_dir)
            
            if Lx < best_Lx:
                best_Lx = Lx 
                print(" >> Saving generator to:  ",generater_dir)
                agent.save_model(generater_dir)
                z = t*(-1)
                Ly, Acc, Lx = eval(agent, devset, z, width, layers, out_dir=model_dir)

            print("--------------------------------------")
        print("************")
        '''
        Ly, Acc, _ = eval(agent, dataset, -2, width, layers, out_dir=model_dir, save_img=False)
        print("   Train: Ly {} Acc = {}".format(Ly, Acc))
        print("Best.Dev: Ly {} Acc = {}".format(best_Ly, best_Acc))
        '''
        acc_list.append(1.0 - Acc)
        best_acc_list.append(1.0 - best_Acc)

    ############################################################################
    ## calc post-trial statistics
    n_dec = 4
    mu = round(np.mean(np.asarray(best_acc_list)), n_dec)
    sd = round(np.std(np.asarray(best_acc_list)), n_dec)
    print("  Dev.Error = {:.4f} \pm {:.4f}".format(mu, sd))

    ## store result to disk just in case...
    results_fname = "{}/post_train_results.txt".format(out_dir)
    log_t = open(results_fname,"a")
    log_t.write("Generalization Results:\n")
    log_t.write("  Dev.Error = {:.4f} \pm {:.4f}\n".format(mu, sd))

    n_dec = 4
    mu = round(np.mean(np.asarray(acc_list)), n_dec)
    sd = round(np.std(np.asarray(acc_list)), n_dec)
    print("  Train.Error = {:.4f} \pm {:.4f}".format(mu, sd))
    log_t.write("Training-Set/Optimization Results:\n")
    log_t.write("  Train.Error = {:.4f} \pm {:.4f}\n".format(mu, sd))

    log_t.close() # close the log file

    f = open('lx_list.txt','a+')
    f.write(str(Lx_list))
    f.write('\n')
    f.close()
    
