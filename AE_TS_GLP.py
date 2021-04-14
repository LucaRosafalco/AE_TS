#AE_TS_GLP.py

#AE_TS_GLP: AutoEncoder Time Series Giorgia Luca Filippo

# Editor: Luca Rosafalco
# Suggestions: https://en.wikipedia.org/wiki/Pimp_My_Ride (pimpatelo un po' G&F)


# import packages %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import tensorflow as tf
print(tf.__version__)
import numpy as np
import pandas as pd
import math
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error
from matplotlib.ticker       import FormatStrFormatter
from tensorflow import keras
from keras      import regularizers
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# analysis control %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# general controls
predict_or_train      = 0        # train the net or use it to make predictions                        (0-predict ; 1-train)
fnn_loss_if           = 0        # use the false nearest neighbor as regularization term for the loss (0-just mse; 1-mse+fnn)
double_training       = 0        # initialize the network weights using saved weights                 (0-from sketch ; 1-transfer learning)
fine_tuning           = 0        # fine tuning of specific layers                                     (0-no fine tuning; 1-fine tuning)
just_test             = 0        # use for the testing the train and val splitting, or not            (0-train val; 1-test)
                                 # the reason behinf the use of 'just_test' is the following:
                                 # you may be interested in plotting the reconstructed signals, evaluate the error norms etc. for
                                 # the same sets that have been used during the training for sake of comparison with the test set

# data folder strings
folder_string_model = 'D:\\Luca\\Dati\\'   # model data folder
ID_pb               = 'model_enrichment'   # string identifying the problem
ID_pb_class         = 'telaio'             # string of the classification problem
ID_string           = 'U'                  # identificative string of each set of recordingss
ID_pb_string        = 'BC'                 # orign of the data
train_model         = 'test_model'         # string used in the train case
param_model         = 'model'              # suffix for the parameters
test_model          = 'test_model'         # string used in the test case

# postprocessing
multi_eval          = 1                    # in test case, it plots many reconstructed signals
save_latent         = 1                    # save latent variables for sensitivity analysis
multi_channel       = 1                    # run this evaluation for all the input channels
plot_box_whiskers   = 0                    # plot latent variables (box-whiskers only)
plot_lat_variable   = 0                    # plot some of the latent variables (box-whiskers and dispersion)
plot_lat_var_all    = 0                    # plot all the latent variables (box-whiskers and dispersion)
plot_corr_heat      = 0                    # compute correlation matrix for the latent variables
eval_error_statistics = 1                  # evaluate the error statistics of the signal reconstruction (like in paper 3_RMMC arXiv)

# data specifications
n_channels         = 2                     # number of channels
which_channels     = [1,2]                 # which dofs are monitored
#2dof       which_channels = [1,2]
# Pirellone which_channels = [20,39]
n_param            = 2                     # number of loaded parameters
which_params       = [0,1]                 # you may select just some parameters for the analysis
latent_dim         = 4                     # number of latent variables
extra_latent       = latent_dim - n_param
signal_sampling    = 0.02                  # signal sampling
#2dof
seq_len_input      = 1000                  # sequence length
#2dof      seq_len_input  = 1000 
#Pirellone seq_len_input  = 999#1000 (dipende)
seq_len_start = 0                          # time series starting point for analysis
seq_len       = 250                        # length of the considered sequence
seq_len_copy  = seq_len                    # 'copy' is needed for post-processing
seq_sampling  = 1                          # sequence sampling (=1 to not have subsampling)

# Hyperparameters
n_epochs_1     = 1000                      # number of epochs
#2dof      n_epochs_1  = 1000  
#Pirellone n_epochs_1  = 500
l_rate_1       = 0.01                      # learning rate
batch_size_1   = 128                       # batch size
lambda_latent  = 0.1                       # lambda parameter for FNN training

n_epochs_2     = 750                       # number of epochs (double training)
l_rate_2       = 0.1                       # learning rate (double training)
batch_size_2   = 512                       # batch size (double training)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# data memory address %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# model data
damage_ID_model    = '4_3'
data_root_model    = folder_string_model + ID_pb + '\\damaged_'
data_root_ID_model = data_root_model + damage_ID_model

# directory in which the tuned model is saved
save_ID_network  = '4_3Yprova'
save_string_weights_1_enc   = '.\\tmp\\SEWAVENET_AE_TS_enc_' + save_ID_network + '.h5'
save_string_weights_1_dec   = '.\\tmp\\SEWAVENET_AE_TS_dec_' + save_ID_network + '.h5'
save_string_weights_1_ae    = '.\\tmp\\SEWAVENET_AE_TS_' + save_ID_network + '.h5'
save_root_ID                = data_root_model + save_ID_network

# directory in which the tuned model is recalled 
restore_ID      = '4_3Yprova'
restore_string_1_enc   = '.\\tmp\\SEWAVENET_AE_TS_enc_' + restore_ID + '.h5'
restore_string_1_dec   = '.\\tmp\\SEWAVENET_AE_TS_dec_' + restore_ID + '.h5'
restore_string_1_ae    = '.\\tmp\\SEWAVENET_AE_TS_' + restore_ID + '.h5'
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Network Hyperparameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pad_ding    = 'SAME'  # padding - 'same' aka zero padding
act_vation  = 'selu'  # activation function - used when specified in the NN
fine_t      = True    # allows for making the weights of some selected layers not trainable
l_two_value = 1e-4    # l2 regularization of the loss

# Encoder hyperparameters   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kernel_size_init_enc_k = 3            # first convolutional layer kernel size

# inception modules specifications
n_block_1   = 3                  # number of inception blocks
n_split_1   = 3                  # number of splits in each inception module
kernel_size_1_enc_k  = [13,8,5]  # inception modules kernel size 

out_0_res_layer = n_channels*n_split_1                #*6
out_1_res_layer = n_channels*n_split_1*2              #*12

# number of filters in the convolutional encoding branch
filter_1_enc_k_1      = n_channels*n_split_1*2        #12
filter_2_enc_k_1      = n_channels*n_split_1*2        #12
filter_3_enc_k_1      = n_channels*n_split_1*2        #12
filter_4_enc_k_1      = n_channels*n_split_1*2        #12
filter_5_enc_k_1      = n_channels*n_split_1*2        #12         
filter_6_enc_k_1      = n_channels*n_split_1*2        #12
filter_7_enc_k_1      = n_channels*n_split_1*2        #12
filter_8_enc_k_1      = n_channels*n_split_1*2        #12
filter_9_enc_k_1      = n_channels*n_split_1*2        #12

# kernel sizes in the encoding branch
kernel_size_1_enc_k_1 = 8;   kernel_size_2_enc_k_1 = 8;   kernel_size_3_enc_k_1 = 8; 
kernel_size_4_enc_k_1 = 5;   kernel_size_5_enc_k_1 = 5;   kernel_size_6_enc_k_1 = 5; 
kernel_size_7_enc_k_1 = 3;   kernel_size_8_enc_k_1 = 3;   kernel_size_9_enc_k_1 = 3

# stride in the encoding branch
stride_1_enc_k = 1;     stride_2_enc_k = 1;      stride_3_enc_k = 1
stride_4_enc_k = 1;     stride_5_enc_k = 1;      stride_6_enc_k = 1
stride_7_enc_k = 1;     stride_8_enc_k = 1;      stride_9_enc_k = 1

# Decoder hyperparameters   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hmmore = 4    # used to relatively measure the number of filter respect to the number of latent variables

# decoder filter dimension
filter_1_dec  = int(latent_dim+(hmmore-extra_latent))*16;   filter_2_dec = int(latent_dim+(hmmore-extra_latent))*16
filter_3_dec  = int(latent_dim+(hmmore-extra_latent))*16;   filter_4_dec = int(latent_dim+(hmmore-extra_latent))*16
filter_5_dec  = int(latent_dim+(hmmore-extra_latent))*16;   filter_6_dec = int(latent_dim+(hmmore-extra_latent))*16
filter_7_dec  = int(latent_dim+(hmmore-extra_latent))*16;   filter_8_dec = int(latent_dim+(hmmore-extra_latent))*16

# decoder kernel dimension
kernel_size_1_dec = 2;   kernel_size_2_dec = 2
kernel_size_3_dec = 2;   kernel_size_4_dec = 2
kernel_size_5_dec = 2;   kernel_size_6_dec = 2; kernel_size_7_dec = 2

# decoder dilation rate - 1st branch (see Wavenet paper Van der Oord et al.)
dilation_rate_1_1 = 2;   dilation_rate_2_1  = 4
dilation_rate_3_1 = 8;   dilation_rate_4_1  = 16
dilation_rate_5_1 = 32;  dilation_rate_6_1  = 64; dilation_rate_7_1 =  128

# decoder dilation rate - 2nd branch (see Wavenet paper Van der Oord et al.)
dilation_rate_1_2 = 2;   dilation_rate_2_2   = 4
dilation_rate_3_2 = 8;   dilation_rate_4_2   = 16
dilation_rate_5_2 = 32;  dilation_rate_6_2   = 64; dilation_rate_7_2 =  128

# decoder dilation rate - 3rd branch (see Wavenet paper Van der Oord et al.)
dilation_rate_1_3 = 2;   dilation_rate_2_3   = 4
dilation_rate_3_3 = 8;   dilation_rate_4_3   = 16
dilation_rate_5_3 = 32;  dilation_rate_6_3   = 64; dilation_rate_7_3 =  128

# decoder stride
stride_1_branch   =  1
stride_2_branch   =  1
stride_3_branch   =  1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# compute time axis and set double training specifications %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
seq_len_end      = seq_len_start + seq_len * seq_sampling
seq_len_end_copy = seq_len_start + seq_len_copy * seq_sampling

t_axis_start   = seq_len_start * signal_sampling    #t_axis
t_axis_end     = seq_len_end * signal_sampling
t_axis_end_copy= seq_len_end_copy * signal_sampling
if double_training:
    n_epochs   = n_epochs_2
    l_rate     = l_rate_2
    batch_size = batch_size_2
else:
    n_epochs   = n_epochs_1
    l_rate     = l_rate_1
    batch_size = batch_size_1
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# read time series function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_data(data_root_ID, case, seq_len, seq_sampling):

    for i1 in range(n_channels):                                                      # load time series

        if len(ID_string) == 0:
            data_path  = data_root_ID + '\\' + ID_pb_string + '_damaged_concat_' + case + '_gdl_' + str(which_channels[i1]) + '.csv'
        else:
            data_path  = data_root_ID + '\\' + ID_pb_string + '_' + ID_string + '_damaged_concat_' + case + '_gdl_' + str(which_channels[i1]) + '.csv'

        print("Sensor to be loaded: {:d}".format(i1))
        X_single_dof = np.genfromtxt(data_path)
        print("Loaded sensor: {:d}".format(i1))

        X_single_dof.astype(np.float32)

        if i1 == 0:
            n_instances = len(X_single_dof) / seq_len_input
            n_instances = int(n_instances)
            print("n_instances: {:d}".format(n_instances))
            X = np.zeros((n_instances, seq_len, n_channels))

        i4 = 0
        for i3 in range(n_instances):                                                 # subsample the time series if required
            X_single_label = X_single_dof[i4 : (i4 + seq_len_input)]
            X_to_pooler = X_single_label[seq_len_start:(seq_len_start+seq_len*seq_sampling):seq_sampling]
            X[i3, 0 : (seq_len), i1] = X_to_pooler
            i4 = i4 + seq_len_input
    # Return 
    return X, n_instances
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# load parameters function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_params(data_root_ID,case,which_params,n_param):
    param_path = data_root_ID + '\\' + 'parameters_' + case + '.csv'                                             
    load_param = np.genfromtxt(param_path)
    for i0 in range(n_param):
        for i1 in range(load_param.shape[0]):
            if which_params[i0] == i1:
                to_be_load = load_param[i1,:]
                # Standardize the param to be loaded
                mean_p     = [np.mean(to_be_load)]
                std_p      = [np.std(to_be_load)]
                to_be_load = (to_be_load - mean_p) / std_p
                mean_p     = np.expand_dims(mean_p, axis=1)
                std_p      = np.expand_dims(std_p, axis=1)
                to_be_load = np.expand_dims(to_be_load, axis=1)
                if i0 == 0:
                    params     = to_be_load
                    param_mean = mean_p
                    param_std  = std_p
                else:
                    params     = np.concatenate((params, to_be_load), axis=1)
                    param_mean = np.concatenate((param_mean, mean_p), axis=1)
                    param_std  = np.concatenate((param_std,  std_p),  axis=1)
    # Return
    return params, param_mean, param_std
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# compute error measureas as in 3_RMMC arXiv %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_errors(X_t,decoded_output_t):
    X_std   = np.std(X_t[:,:],dtype=np.float64)
    # L2 error
    rmse    = mean_squared_error(X_t[:,:],decoded_output_t[:,:],multioutput='raw_values',squared=False)
    rmse_a  = rmse / X_std
    # Linf error
    max1    = np.max(abs(X_t[:,:]-decoded_output_t[:,:]), axis=0)
    max_a   = max1 / X_std
    for i0 in range(X_t.shape[1]):
        if i0 == 0:
            rmse_ap = rmse[0] / np.std(X_t[:,0],dtype=np.float64)
            rmse_ap = np.expand_dims(rmse_ap, axis=0)
            # max_ap  = max1[0] / abs(X_t[maxp[0],0,0])
            max_ap  = max1[0] / np.std(X_t[:,0],dtype=np.float64)
            max_ap  = np.expand_dims(max_ap, axis=0)
        else:
            to_be_rmse_ap = rmse[i0] / np.std(X_t[:,i0],dtype=np.float64)
            to_be_rmse_ap = np.expand_dims(to_be_rmse_ap, axis=0)
            rmse_ap   = np.concatenate((rmse_ap,to_be_rmse_ap), axis=0)
            # to_be_max_test1_ap  = max_test1[i0] / abs(X_test_t[maxp_test1[i0],i0,0])
            to_be_max_ap  = max1[i0] / np.std(X_t[:,i0],dtype=np.float64)
            to_be_max_ap  = np.expand_dims(to_be_max_ap, axis=0)
            max_ap     = np.concatenate((max_ap, to_be_max_ap), axis=0)
    rmse    = np.expand_dims(rmse, axis=1)
    rmse_a  = np.expand_dims(rmse_a, axis=1)
    max1    = np.expand_dims(max1, axis=1)
    max_a   = np.expand_dims(max_a, axis=1)
    rmse_ap = np.expand_dims(rmse_ap, axis=1)
    max_ap  = np.expand_dims(max_ap, axis=1)
    # Return
    return np.concatenate([rmse,rmse_a,max1,max_a,rmse_ap,max_ap],axis=1)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# plot reconstructed signal function (postprocessing) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_reconstruction_multi(t_axis,n,n_sub,dof,X,sim_type,param):
    matplotlib.style.use('classic')
    matplotlib.rc('font',  size=8, family='serif')
    matplotlib.rc('axes',  titlesize=8)
    matplotlib.rc('text',  usetex=True)
    matplotlib.rc('lines', linewidth=0.5)
    #fig, axs = plt.subplots(n, n, figsize=(4.75,4.75))
    fig, axs = plt.subplots(n, n)
    fig.tight_layout()
    for i in range(n):
        for j in range(n):
            encoder_input  = np.expand_dims(X[i+j*n], axis = 0)
            z              = encoder.predict(encoder_input)
            decoded_output = decoder.predict(z)
            X_plot              = (X[i+j*n,:,dof]*std_data) + mean_data
            decoded_output_plot = (decoded_output[0,:,dof]*std_data) + mean_data

            axs[i,j].plot(time_axis[:],X_plot,'k', time_axis[:],decoded_output_plot,'orange')
            for axis in ['top','bottom','left','right']:
                axs[i,j].spines[axis].set_linewidth(0.5)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines
            axs[i,j].yaxis.set_ticks_position('left')
            axs[i,j].xaxis.set_ticks_position('bottom')
            #Pirellone            
            axs[i,j].xaxis.set_major_locator(ticker.LinearLocator(5))
            axs[i,j].yaxis.set_major_locator(ticker.LinearLocator(5))
            axs[i,j].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

            param_to_save_ij = param[i+j*n]
            param_to_save_ij = np.expand_dims(param_to_save_ij, axis=1)
            if i == 0 and j == 0:
                param_to_save = param_to_save_ij
            else:
                param_to_save = np.concatenate([param_to_save,param_to_save_ij],axis=1)

            if n == 1:
                plt.xlabel(r'time [s]', fontsize=24)
                plt.ylabel(r'normalized displ. [-]', fontsize=24)
                plt.legend([r'instance for inference',r'generated instance'],loc='upper right',fontsize=24)
                fig_save_string1  = save_root_ID + '\\inference_vs_generated_instance_' + sim_type + '_' + save_ID_network + 'dof_' + str(dof) + '.png'
                fig_save_string2  = save_root_ID + '\\inference_vs_generated_instance_' + sim_type + '_' + save_ID_network + 'dof_' + str(dof) + '.pdf'
                plt.savefig(fig_save_string1, bbox_inches='tight')
                plt.savefig(fig_save_string2, bbox_inches='tight')

            else:
                plt.xlabel(r'$t$ [s]', labelpad=-1, fontsize=8)
                plt.ylabel(r'displ. [m]', labelpad=-2, fontsize=8)
    if dof == 0:                    
        axs[i,j].legend([r'$\mathbf{v}_1 $',r'$\mathbf{u}_1 $'],loc='upper right')
    elif dof == 1:
        axs[i,j].legend([r'$\mathbf{v}_2 $',r'$\mathbf{u}_2 $'],loc='upper right')
    fig_save_string1  = save_root_ID + '\\inference_vs_generated_instance_' + sim_type + '_' + save_ID_network + 'multi_dof_' + str(dof) + '.png'
    fig_save_string2  = save_root_ID + '\\inference_vs_generated_instance_' + sim_type + '_' + save_ID_network + 'multi_dof_' + str(dof) + '.pdf'
    plt.savefig(fig_save_string1, bbox_inches='tight')
    plt.savefig(fig_save_string2, bbox_inches='tight')

    param_save_string = save_root_ID + '\\param_inference_vs_generated_instance_' + sim_type + '_' + save_ID_network + 'multi_dof_' + str(dof) + '.csv'
    np.savetxt(param_save_string, param_to_save, delimiter=",")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
# compute fnn loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# l'implementazione si trova su github; a Milano devo avere degli appunti dove ho ripercorso passo passo
# il codice per verificarlo. Purtroppo, non sono a Milano. Non ricordo se i commenti di sotto li ho fatti
# io o erano già presenti
def comp_fnn_loss(z, n_batch):
    """
    An activity regularizer based on the False-Nearest-Neighbor
    Algorithm of Kennel, Brown,and Arbanel. Phys Rev A. 1992
    
    Parameters
    ----------
    - k : int 
        DEPRECATED. The number of nearest neighbors to use to compute 
        neighborhoods. Right now this is set to a constant, since it 
        doesn't affect the embedding
    Development
    -----------
    + Try activity regularizing based on the variance of a neuron, rather than
    its absolute average activity
    
    """
    # changing these parameters is equivalent to
    # changing the strength of the regularizer, so we keep these fixed (these values
    # correspond to the original values used in Kennel et al 1992).
    rtol = 20.0
    atol = 2.0
    k_frac = .01

    k = max(1, int(k_frac*n_batch))

    ## Vectorized version of distance matrix calculation
    tri_mask = tf.linalg.band_part(tf.ones((latent_dim, latent_dim), tf.float32), -1, 0)
    batch_masked = tf.multiply(tri_mask[:, tf.newaxis, :], z[tf.newaxis, ...])
    X_sq = tf.reduce_sum(batch_masked*batch_masked, axis=2, keepdims=True)
    pdist_vector = X_sq + tf.transpose(X_sq, [0,2,1]) - 2*tf.matmul(batch_masked, tf.transpose(batch_masked, [0,2,1])) 
    all_dists = pdist_vector
    all_ra = tf.sqrt((1/(n_batch*tf.range(1, 1+latent_dim, dtype=tf.float32)))*tf.reduce_sum(tf.square(batch_masked - tf.reduce_mean(batch_masked, axis=1, keepdims=True)), axis=(1,2)))
    
    # Avoid singularity in the case of zeros
    all_dists = tf.clip_by_value(all_dists, 1e-14, tf.reduce_max(all_dists))

    #inds = tf.argsort(all_dists, axis=-1)
    _, inds = tf.math.top_k(-all_dists, k+1)
    # top_k currently faster than argsort because it truncates matrix
    
    neighbor_dists_d = tf.gather(all_dists, inds, batch_dims=-1)
    neighbor_new_dists = tf.gather(all_dists[1:], inds[:-1], batch_dims=-1)
    
    # Eq. 4 of Kennel et al.
    scaled_dist = tf.sqrt((tf.square(neighbor_new_dists) - tf.square(neighbor_dists_d[:-1]))/tf.square(neighbor_dists_d[:-1]))
    
    # Kennel condition #1
    is_false_change = (scaled_dist > rtol) 
    # Kennel condition 2
    is_large_jump = (neighbor_new_dists > atol*all_ra[:-1, tf.newaxis, tf.newaxis])

    is_false_neighbor = tf.math.logical_or(is_false_change, is_large_jump)
    total_false_neighbors = tf.cast(is_false_neighbor, tf.int32)[..., 1:(k+1)]
    
    # Pad zero to match dimensionality of latent space
    reg_weights = 1 - tf.reduce_mean(tf.cast(total_false_neighbors, tf.float64), axis=(1,2))
    reg_weights = tf.pad(reg_weights, [[1, 0]])

    # Find batch average activity
    activations_batch_averaged  = tf.sqrt(tf.reduce_mean(tf.square(z), axis=0))

    # L2 Activity regularization
    activations_batch_averaged = tf.cast(activations_batch_averaged, tf.float64)
    loss = tf.reduce_sum(tf.multiply(reg_weights, activations_batch_averaged))
    
    return tf.cast(loss, tf.float32)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if predict_or_train:
    X_train, n_instances = read_data(data_root_ID=data_root_ID_model, case=train_model, seq_len=seq_len, seq_sampling=seq_sampling)
    if eval_error_statistics:
        params_tr_mod, param_mean_model, param_std_model = load_params(data_root_ID=data_root_ID_model,case=param_model, which_params=which_params, n_param=n_param)

    if eval_error_statistics:
        params_train   = params_tr_mod
        param_mean     = param_mean_model
        param_std      = param_std_model

    # Standardize the data
    mean_data = [np.mean(X_train)]
    std_data  = [np.std(X_train)]
    X_train   = (X_train - mean_data) / std_data
    # Save mean and standard deviation
    data_mean_name   = save_root_ID + '\\data_mean.csv'
    data_st_dev_name = save_root_ID + '\\data_st_dev.csv'
    np.savetxt(data_mean_name,   mean_data, delimiter=",")        
    np.savetxt(data_st_dev_name, std_data,  delimiter=",")
    
    # Split between train and validation set
    X_tr, X_vld = train_test_split(X_train, random_state=5)

    # save parameter mean and standard deviation
    if eval_error_statistics:
        data_mean_p_name = save_root_ID + '\\data_param_mean.csv'
        data_std_p_name  = save_root_ID + '\\data_param_st_dev.csv'
        np.savetxt(data_mean_p_name, param_mean, delimiter=",")        
        np.savetxt(data_std_p_name,  param_std,  delimiter=",")

        # Split between train and validation set (time series and parameters are splitted in the same way)
        X_tr, X_vld, params_tr, params_vld = train_test_split(X_train, params_train, random_state=5)

else:
    X_test_model, n_instances_model = read_data(data_root_ID=data_root_ID_model, case=test_model, seq_len=seq_len, seq_sampling=seq_sampling)
    if eval_error_statistics:
        params_tr_mod, param_mean_model, param_std_model = load_params(data_root_ID=data_root_ID_model,case=param_model, which_params=which_params, n_param=n_param)

    X_test         = X_test_model
    if eval_error_statistics:
        param_mean     = param_mean_model
        param_std      = param_std_model

    # Standardize the data
    mean_data = [np.mean(X_test)]
    std_data  = [np.std(X_test)]
    X_test   = (X_test - mean_data) / std_data

    data_mean_name   = save_root_ID + '\\data_test_mean.csv'
    data_st_dev_name = save_root_ID + '\\data_test_st_dev.csv'
    np.savetxt(data_mean_name,   mean_data, delimiter=",")        
    np.savetxt(data_st_dev_name, std_data,  delimiter=",")

    if eval_error_statistics:
        data_mean_p_name = save_root_ID + '\\data_test_param_mean.csv'
        data_std_p_name  = save_root_ID + '\\data_test_param_st_dev.csv'
        np.savetxt(data_mean_p_name, param_mean, delimiter=",")        
        np.savetxt(data_std_p_name,  param_std,  delimiter=",")

        # the splitting is operated if the error norms are required even for the train and validation sets
        # it depends on just_test
        X_tr, X_vld, params_tr, params_vld = train_test_split(X_test, params_tr_mod, random_state=5)
        params_test = np.concatenate((params_tr,params_vld),axis=0)

    else:
        # the splitting is operated if the error norms are required even for the train and validation sets
        # it depends on just_test
        X_tr, X_vld = train_test_split(X_test, random_state=5)

    X_test         = np.concatenate((X_tr,X_vld),axis=0)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Encoder %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
encoder_inputs      = keras.layers.Input(shape=(None, n_channels), name='encoder_inputs_k')
#initial_layer
z_0_k               = keras.layers.Conv1D(filters=out_0_res_layer,kernel_size=kernel_size_init_enc_k,strides=1,padding=pad_ding,activation=None,name='conv_initial_layer')(encoder_inputs)
previous_filters    = out_0_res_layer

# inception module
for i1 in range(n_block_1):
    n_input_channels = int(np.shape(z_0_k)[-1])
    stride=1
    #split_layer (parallel operation of transform layers for ResNeXt structure)
    splitted_branches_1 = list()
    for i2 in range(n_split_1):
        #transform_layer
        branch_1 = keras.layers.Conv1D(filters=previous_filters // n_split_1,kernel_size=1,strides=1,padding=pad_ding,activation=None)(z_0_k)
        branch_1 = keras.layers.Activation('relu')(branch_1)
        branch_1 = keras.layers.Conv1D(filters=previous_filters // n_split_1,kernel_size=kernel_size_1_enc_k[i2],strides=stride,kernel_regularizer=regularizers.l2(l_two_value),padding=pad_ding,activation=None)(branch_1)
        branch_1 = keras.layers.Activation('relu')(branch_1)
        splitted_branches_1.append(branch_1)
    subway_x_1 = keras.layers.concatenate(splitted_branches_1,axis=-1)
    if i1 == (n_block_1-1):
        previous_filters = out_1_res_layer
    z_1_k = subway_x_1

# convolutional branch
conv1_en_k  = keras.layers.Conv1D(filters=filter_1_enc_k_1,kernel_size=kernel_size_1_enc_k_1,strides=stride_1_enc_k,padding=pad_ding,activation=act_vation,kernel_regularizer=regularizers.l2(l_two_value),name='conv_1_en_k')(encoder_inputs)
conv2_en_k  = keras.layers.Conv1D(filters=filter_2_enc_k_1,kernel_size=kernel_size_2_enc_k_1,strides=stride_2_enc_k,padding=pad_ding,activation=act_vation,kernel_regularizer=regularizers.l2(l_two_value),name='conv_2_en_k')(conv1_en_k)
conv3_en_k  = keras.layers.Conv1D(filters=filter_3_enc_k_1,kernel_size=kernel_size_3_enc_k_1,strides=stride_3_enc_k,padding=pad_ding,activation=act_vation,kernel_regularizer=regularizers.l2(l_two_value),name='conv_3_en_k')(conv2_en_k)
conv4_en_k  = keras.layers.Conv1D(filters=filter_4_enc_k_1,kernel_size=kernel_size_4_enc_k_1,strides=stride_4_enc_k,padding=pad_ding,activation=act_vation,kernel_regularizer=regularizers.l2(l_two_value),name='conv_4_en_k')(conv3_en_k)
conv5_en_k  = keras.layers.Conv1D(filters=filter_5_enc_k_1,kernel_size=kernel_size_5_enc_k_1,strides=stride_5_enc_k,padding=pad_ding,activation=act_vation,kernel_regularizer=regularizers.l2(l_two_value),name='conv_5_en_k')(conv4_en_k)
conv6_en_k  = keras.layers.Conv1D(filters=filter_6_enc_k_1,kernel_size=kernel_size_6_enc_k_1,strides=stride_6_enc_k,padding=pad_ding,activation=act_vation,kernel_regularizer=regularizers.l2(l_two_value),name='conv_6_en_k')(conv5_en_k)
conv7_en_k  = keras.layers.Conv1D(filters=filter_7_enc_k_1,kernel_size=kernel_size_7_enc_k_1,strides=stride_7_enc_k,padding=pad_ding,activation=act_vation,kernel_regularizer=regularizers.l2(l_two_value),name='conv_7_en_k')(conv6_en_k)
conv8_en_k  = keras.layers.Conv1D(filters=filter_8_enc_k_1,kernel_size=kernel_size_8_enc_k_1,strides=stride_8_enc_k,padding=pad_ding,activation=act_vation,kernel_regularizer=regularizers.l2(l_two_value),name='conv_8_en_k')(conv7_en_k)
conv9_en_k  = keras.layers.Conv1D(filters=filter_9_enc_k_1,kernel_size=kernel_size_9_enc_k_1,strides=stride_9_enc_k,padding=pad_ding,activation=act_vation,kernel_regularizer=regularizers.l2(l_two_value),name='conv_9_en_k')(conv8_en_k)

# merge the output of the inception module and of the convolutional branch
z_out_k = keras.layers.concatenate([conv9_en_k,z_1_k],axis=-1)
z_out_k = keras.layers.Conv1D(filters=(out_0_res_layer+filter_9_enc_k_1),kernel_size=1,strides=1,padding=pad_ding,activation=act_vation,kernel_regularizer=regularizers.l2(l_two_value),name='conv_conc_en_k')(z_out_k)

# apply global average pooling for reducing dimensionality
z_gl_pool = keras.layers.GlobalAveragePooling1D(name='GlobalPooling_enc_fin')(z_out_k)

# final fully connected layers
z_dense_1       = keras.layers.Dense(latent_dim**2, activation=act_vation, trainable=fine_t, name='regression_encodings_1_k')(z_gl_pool)
z_k             = keras.layers.Dense(latent_dim, trainable=fine_t, name='regression_encodings_2_k')(z_dense_1)

# define model object in Keras
encoder = keras.models.Model(encoder_inputs,z_k,name='encoder_k')
encoder.summary()    #print a summary of the model

# Decoder %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# see Wavenet architecture Van der Oord et al. (2015)
# decoder input
decoder_inputs      = keras.layers.Input(shape=(latent_dim), name='z')

# 1st branch
conv0_flat_dec_1    = keras.layers.Dense(seq_len*filter_1_dec, kernel_initializer = tf.keras.initializers.VarianceScaling, bias_initializer = tf.keras.initializers.VarianceScaling, trainable=fine_t, name = 'dense_back_1')(decoder_inputs), 
conv0_dec_1         = keras.layers.Reshape((seq_len, filter_1_dec))(conv0_flat_dec_1[0])
conv1_dec_1         = keras.layers.Conv1D(filters = filter_1_dec, kernel_size=kernel_size_1_dec, dilation_rate = (dilation_rate_1_1), strides = (stride_1_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_1_1')(conv0_dec_1)
conv2_dec_1         = keras.layers.Conv1D(filters = filter_2_dec, kernel_size=kernel_size_2_dec, dilation_rate = (dilation_rate_2_1), strides = (stride_1_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_2_1')(conv1_dec_1)
conv3_dec_1         = keras.layers.Conv1D(filters = filter_3_dec, kernel_size=kernel_size_3_dec, dilation_rate = (dilation_rate_3_1), strides = (stride_1_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_3_1')(conv2_dec_1)
conv4_dec_1         = keras.layers.Conv1D(filters = filter_4_dec, kernel_size=kernel_size_4_dec, dilation_rate = (dilation_rate_4_1), strides = (stride_1_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_4_1')(conv3_dec_1)
conv5_dec_1         = keras.layers.Conv1D(filters = filter_5_dec, kernel_size=kernel_size_5_dec, dilation_rate = (dilation_rate_5_1), strides = (stride_1_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_5_1')(conv4_dec_1)
conv6_dec_1         = keras.layers.Conv1D(filters = filter_6_dec, kernel_size=kernel_size_6_dec, dilation_rate = (dilation_rate_6_1), strides = (stride_1_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_6_1')(conv5_dec_1)
conv7_dec_1         = keras.layers.Conv1D(filters = filter_7_dec, kernel_size=kernel_size_7_dec, dilation_rate = (dilation_rate_7_1), strides = (stride_1_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_7_1')(conv6_dec_1)

# 2nd branch
conv1_dec_2         = keras.layers.Conv1D(filters = filter_1_dec, kernel_size=kernel_size_1_dec, dilation_rate = (dilation_rate_1_2), strides = (stride_2_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_1_2')(conv7_dec_1)
conv2_dec_2         = keras.layers.Conv1D(filters = filter_2_dec, kernel_size=kernel_size_2_dec, dilation_rate = (dilation_rate_2_2), strides = (stride_2_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_2_2')(conv1_dec_2)
conv3_dec_2         = keras.layers.Conv1D(filters = filter_3_dec, kernel_size=kernel_size_3_dec, dilation_rate = (dilation_rate_3_2), strides = (stride_2_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_3_2')(conv2_dec_2)
conv4_dec_2         = keras.layers.Conv1D(filters = filter_4_dec, kernel_size=kernel_size_4_dec, dilation_rate = (dilation_rate_4_2), strides = (stride_2_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_4_2')(conv3_dec_2)
conv5_dec_2         = keras.layers.Conv1D(filters = filter_5_dec, kernel_size=kernel_size_5_dec, dilation_rate = (dilation_rate_5_2), strides = (stride_2_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_5_2')(conv4_dec_2)
conv6_dec_2         = keras.layers.Conv1D(filters = filter_6_dec, kernel_size=kernel_size_6_dec, dilation_rate = (dilation_rate_6_2), strides = (stride_2_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_6_2')(conv5_dec_2)
conv7_dec_2         = keras.layers.Conv1D(filters = filter_7_dec, kernel_size=kernel_size_7_dec, dilation_rate = (dilation_rate_7_2), strides = (stride_2_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_7_2')(conv6_dec_2)

# 3rd branch
conv1_dec_3         = keras.layers.Conv1D(filters = filter_3_dec, kernel_size=kernel_size_3_dec, dilation_rate = (dilation_rate_1_3), strides = (stride_3_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_1_3')(conv7_dec_2)
conv2_dec_3         = keras.layers.Conv1D(filters = filter_2_dec, kernel_size=kernel_size_2_dec, dilation_rate = (dilation_rate_2_3), strides = (stride_3_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_2_3')(conv1_dec_3)
conv3_dec_3         = keras.layers.Conv1D(filters = filter_3_dec, kernel_size=kernel_size_3_dec, dilation_rate = (dilation_rate_3_3), strides = (stride_3_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_3_3')(conv2_dec_3)
conv4_dec_3         = keras.layers.Conv1D(filters = filter_4_dec, kernel_size=kernel_size_4_dec, dilation_rate = (dilation_rate_4_3), strides = (stride_3_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_4_3')(conv3_dec_3)
conv5_dec_3         = keras.layers.Conv1D(filters = filter_5_dec, kernel_size=kernel_size_5_dec, dilation_rate = (dilation_rate_5_3), strides = (stride_3_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_5_3')(conv4_dec_3)
conv6_dec_3         = keras.layers.Conv1D(filters = filter_6_dec, kernel_size=kernel_size_6_dec, dilation_rate = (dilation_rate_6_3), strides = (stride_3_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_6_3')(conv5_dec_3)
conv7_dec_3         = keras.layers.Conv1D(filters = filter_7_dec, kernel_size=kernel_size_7_dec, dilation_rate = (dilation_rate_7_3), strides = (stride_3_branch), activation = act_vation, padding = pad_ding, kernel_regularizer=regularizers.l2(l_two_value), trainable=fine_t, name = 'deconv_7_3')(conv6_dec_3)

# final dense layers
decoder_outputs_1     = keras.layers.Dense(n_channels*16, activation=act_vation, kernel_regularizer=regularizers.l2(l_two_value), bias_regularizer=regularizers.l2(l_two_value), name='dense_final_1')(conv7_dec_3)
decoder_outputs_2     = keras.layers.Dense(n_channels*4, activation=act_vation, kernel_regularizer=regularizers.l2(l_two_value), bias_regularizer=regularizers.l2(l_two_value), name='dense_final_2')(decoder_outputs_1)
decoder_outputs       = keras.layers.Dense(n_channels, kernel_regularizer=regularizers.l2(l_two_value), bias_regularizer=regularizers.l2(l_two_value), name='dense_final')(decoder_outputs_2)

# define a Keras model for the decoder
decoder = keras.models.Model(decoder_inputs,decoder_outputs,name='decoder')
decoder.summary()     # print a summary of the model

# AE #############################################################################
# define a Keras model merging the encoder and the decoder
z = encoder(encoder_inputs)
decoder_outputs = decoder(z)
# model
ae = keras.models.Model(encoder_inputs, decoder_outputs, name='SEWAVENET_AE_TS_DB')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Define the loss function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# define reconstruction loss
def rec_loss(encoder_inputs,decoder_outputs):
    reconstruction_loss = keras.losses.mse(encoder_inputs,decoder_outputs)
    reconstruction_loss *= seq_len
    reconstruction_loss = keras.backend.mean(keras.backend.mean(reconstruction_loss, axis=-1))
    return reconstruction_loss

# specify if you want to use the fnn heuristics to update the loss
if fnn_loss_if:
    # Reconstruction loss
    reconstruction_loss = keras.losses.mse(encoder_inputs,decoder_outputs)      # reconstruction loss
    reconstruction_loss *= seq_len
    reconstruction_loss = keras.backend.mean(keras.backend.mean(reconstruction_loss, axis=-1))
    fnn_loss            = comp_fnn_loss(z,batch_size)                           # fnn loss
    ae_loss_fnn         = reconstruction_loss + seq_len*lambda_latent*fnn_loss  # total loss (lambda hyperparameter)
else:
    # Reconstruction loss
    reconstruction_loss = keras.losses.mse(encoder_inputs,decoder_outputs)     # reconstruction loss
    reconstruction_loss *= seq_len
    # AE loss
    ae_loss             = keras.backend.mean(keras.backend.mean(reconstruction_loss, axis=-1)) # total loss

if fnn_loss_if:
    ae.add_loss(ae_loss_fnn)
    ae.compile(optimizer='adam',metrics=[rec_loss])  #ask for the reconstruction loss as metric
else:
    ae.add_loss(ae_loss)
    ae.compile(optimizer='adam')
ae.summary()

# forse inutile: probabilmente serviva per plottare la rete su tensorboard %%%
ae._layers= [                                                             #%%%
    layer for layer in ae._layers if isinstance(layer, keras.layers.Layer)#%%%
]                                                                         #%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# model execution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if predict_or_train:
        
    # training %%%%%%%%##################################################
    if double_training:    #needed for transfer learning (not start the training from scratch)
        # Load the network weights
        encoder.load_weights(restore_string_1_enc)  # restore encoder model weights
        decoder.load_weights(restore_string_1_dec)  # restore decoder model weights
        ae.load_weights(restore_string_1_ae)        # restore AE model weights
        # ask for fine tuning
        if fine_tuning:
            fine_t = False
        else:
            fine_t = True
        ae.compile(optimizer='adam')    # compile the AE model using Adam optimization algorithm

    # callbacks defined for the training
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True) # set the callback for early stopping
    checkpoint_cb     = keras.callbacks.ModelCheckpoint(save_string_weights_1_ae, save_weights_only=True, save_best_only=True) #needed to save the best model

    # 'fit' allows for training the NN
    history = ae.fit(X_tr,epochs=n_epochs,batch_size=batch_size,validation_data=(X_vld,None),callbacks=[early_stopping_cb,checkpoint_cb])
    # save <<just>> the weights of the NN (you can not load it if the models are not defined elsewhere)
    encoder.save_weights(save_string_weights_1_enc)
    decoder.save_weights(save_string_weights_1_dec)
    ae.save_weights(save_string_weights_1_ae)

    # training postprocessing #############################
    # plot training history #####################
    # pd.DataFrame(history.history).plot(figsize=(8,5))     # plot train hystory (optional)
    plt.plot(history.history['loss'])                       # plot loss (training set)
    plt.plot(history.history['val_loss'])                   # plot loss (validation set)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()                                             # show the graphs previously defined
    # save the history plot - ERROR a me non funziona
    fig_save_string1  = save_root_ID + '\\loss_' + save_ID_network + '.png'
    fig_save_string2  = save_root_ID + '\\loss_' + save_ID_network + '.pdf'
    plt.savefig(fig_save_string1, bbox_inches='tight')
    plt.savefig(fig_save_string2, bbox_inches='tight')
    
    # save latent variables activation ###########
    # if you are using the fnn regularization of the loss function, save the latent variables activation
    if fnn_loss_if:
        # save for the training set  #########
        encoder_input  = X_tr[:]
        z            = encoder.predict(encoder_input)
        for i in range(latent_dim): #cycle on the latent variables
            z_tr_std_i   = np.std(z[:,i])
            z_tr_std_i   = np.expand_dims(z_tr_std_i, axis=0)
            if i == 0:
                z_tr_std = z_tr_std_i
            else:
                z_tr_std = np.concatenate([z_tr_std,z_tr_std_i],axis=0)
        lat_val_tr = save_root_ID + '\\latent_variances_tr_' + save_ID_network + '.csv'
        np.savetxt(lat_val_tr, z_tr_std, delimiter=",")
        # save for the validation set ########
        encoder_input  = X_vld[:]
        z            = encoder.predict(encoder_input)
        for i in range(latent_dim):
            z_vld_std_i   = np.std(z[:,i])
            z_vld_std_i   = np.expand_dims(z_vld_std_i, axis=0)
            if i == 0:
                z_vld_std = z_vld_std_i
            else:
                z_vld_std = np.concatenate([z_vld_std,z_vld_std_i],axis=0)
        lat_val_vld = save_root_ID + '\\latent_variances_vld_' + save_ID_network + '.csv'
        np.savetxt(lat_val_vld, z_vld_std, delimiter=",")

    # evaluate the reconstruction error statistics #########
    if eval_error_statistics:
        # training set ###########
        # compute the error statistics for the training set
        encoder_input  = X_tr[:]
        z            = encoder.predict(encoder_input)
        decoded_output = decoder.predict(z)            
        X_tr_t           = np.moveaxis(X_tr, 0, 1)
        decoded_output_t = np.moveaxis(decoded_output, 0, 1)
        for i in range(n_channels):  #cycle on the input channels
            to_save_tr_i = compute_errors(X_tr_t[:,:,i],decoded_output_t[:,:,i])
            if i == 0:
                to_save_tr = to_save_tr_i
            else:
                to_save_tr = np.concatenate([to_save_tr,to_save_tr_i],axis=1 )
        params1_tr  = np.expand_dims(params_tr[:,0], axis=1)
        params2_tr  = np.expand_dims(params_tr[:,1], axis=1)
        to_save_tr  = np.concatenate([to_save_tr,params1_tr,params2_tr],axis=1)
        #plot_dispersion_stat.m
        rec_stat_n_tr = save_root_ID + '\\reconstruction_statics_tr_' + save_ID_network + '.csv'
        np.savetxt(rec_stat_n_tr, to_save_tr, delimiter=",")
        # validation set ###########
        # compute the error statistics for the validation set
        encoder_input  = X_vld[:]
        z            = encoder.predict(encoder_input)
        decoded_output = decoder.predict(z)
        X_vld_t           = np.moveaxis(X_vld, 0, 1)
        decoded_output_t  = np.moveaxis(decoded_output, 0, 1)
        for i in range(n_channels):  #cycle on the input channels
            to_save_vld_i = compute_errors(X_vld_t[:,:,i],decoded_output_t[:,:,i])
            if i == 0:
                to_save_vld = to_save_vld_i
            else:
                to_save_vld = np.concatenate([to_save_vld,to_save_vld_i],axis=1 )
        params1_vld  = np.expand_dims(params_vld[:,0], axis=1)
        params2_vld  = np.expand_dims(params_vld[:,1], axis=1)
        to_save_vld  = np.concatenate([to_save_vld,params1_vld,params2_vld],axis=1)
        #plot_dispersion_stat.m
        rec_stat_n_vld = save_root_ID + '\\reconstruction_statics_vld_' + save_ID_network + '.csv'
        np.savetxt(rec_stat_n_vld, to_save_vld, delimiter=",")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

else:

# prediction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Load the network weights
    encoder.load_weights(restore_string_1_enc)
    decoder.load_weights(restore_string_1_dec)
    ae.load_weights(restore_string_1_ae)        

    # Multiple evaluation of the model %%%%%%%%%%%%%%%%%%%%%%%
    if multi_eval:

        if just_test == 1: # the reconstructed signals belong to the test set

            # plot signals ########################### (for test set)
            n_test     = 4
            n_sub_test = n_test ** 2 #number of signals plotted

            time_axis  = np.arange(t_axis_start,t_axis_end,signal_sampling*seq_sampling) # set the time axis

            if multi_channel == 1:
                for dof in range(n_channels):
                    plot_reconstruction_multi(time_axis,n_test,n_sub_test,dof,X_test,'test',params_test)
            else:
                plot_reconstruction_multi(time_axis,n_test,n_sub_test,0,X_test,'test',params_test)

            # evaluation of the error statistics###### (for test set)
            if eval_error_statistics:
                encoder_input  = X_test[:]
                z            = encoder.predict(encoder_input)
                decoded_output = decoder.predict(z)

                X_test_t         = np.moveaxis(X_test, 0, 1)
                decoded_output_t = np.moveaxis(decoded_output, 0, 1)
                for i in range(n_channels):
                    to_save_test_i = compute_errors(X_test_t[:,:,i],decoded_output_t[:,:,i])
                    if i == 0:
                        to_save_test = to_save_test_i
                    else:
                        to_save_test = np.concatenate([to_save_test,to_save_test_i],axis=1 )
                params1_test  = np.expand_dims(params_test[:,0], axis=1)
                params2_test  = np.expand_dims(params_test[:,1], axis=1)
                to_save_test  = np.concatenate([to_save_test,params1_test,params2_test],axis=1)
                rec_stat_n_test = save_root_ID + '\\reconstruction_statics_test_' + save_ID_network + '.csv'
                np.savetxt(rec_stat_n_test, to_save_test, delimiter=",")

            # save latent variable activation when the fnn regularization is used#### (for test set)
            if fnn_loss_if:
                encoder_input  = X_test[:]
                z            = encoder.predict(encoder_input)
                for i in range(latent_dim):
                    z_test_std_i   = np.std(z[:,i])
                    z_test_std_i   = np.expand_dims(z_test_std_i, axis=0)
                    if i == 0:
                        z_test_std = z_test_std_i
                    else:
                        z_test_std = np.concatenate([z_test_std,z_test_std_i],axis=0)
                lat_val_test = save_root_ID + '\\latent_variances_test_' + save_ID_network + '.csv'
                np.savetxt(lat_val_test, z_test_std, delimiter=",")

        else: # the reconstructed signals belong to the training and validation set


            # plot signals %%%%%%%%%%%%%%%%%%%%%%%%% (for training and valiation set)
            n_tr  = 4
            n_vld = 4
            n_sub_tr  = n_tr ** 2   # number of training signals plotted
            n_sub_vld = n_vld ** 2  # number of validation signals plotted

            time_axis = np.arange(t_axis_start,t_axis_end,signal_sampling*seq_sampling)

            # train %%%%%%%%%%%%%% (plot reconstructed signal training set)
            if multi_channel == 1:
                for dof in range(n_channels):
                    plot_reconstruction_multi(time_axis,n_tr,n_sub_tr,dof,X_tr,'train',params_tr)
            else:
                plot_reconstruction_multi(time_axis,n_tr,n_sub_tr,0,X_tr,'train',params_tr)

            # validation %%%%%%%%%% (plot reconstructed signal validation set)
            if multi_channel == 1:
                for dof in range(n_channels):
                    plot_reconstruction_multi(time_axis,n_vld,n_sub_vld,dof,X_vld,'validation',params_vld)
            else:
                plot_reconstruction_multi(time_axis,n_vld,n_sub_vld,0,X_vld,'validation',params_vld)                
            
            # evaluate error statistics %%%%%%%%%%%%%%%%%%%%%%%
            if eval_error_statistics:
                # for the training set %%%%%%%%%%%%% (evaluate error statistics)
                encoder_input  = X_tr[:]
                z              = encoder.predict(encoder_input)
                decoded_output = decoder.predict(z)                
                X_tr_t           = np.moveaxis(X_tr, 0, 1)
                decoded_output_t = np.moveaxis(decoded_output, 0, 1)
                for i in range(n_channels):
                    to_save_tr_i = compute_errors(X_tr_t[:,:,i],decoded_output_t[:,:,i])
                    if i == 0:
                        to_save_tr = to_save_tr_i
                    else:
                        to_save_tr = np.concatenate([to_save_tr,to_save_tr_i],axis=1 )
                params1_tr  = np.expand_dims(params_tr[:,0], axis=1)
                params2_tr  = np.expand_dims(params_tr[:,1], axis=1)
                to_save_tr  = np.concatenate([to_save_tr,params1_tr,params2_tr],axis=1)
                rec_stat_n_tr = save_root_ID + '\\reconstruction_statics_tr_' + save_ID_network + '.csv'
                np.savetxt(rec_stat_n_tr, to_save_tr, delimiter=",")

                # for the validation set %%%%%%%%%%% (evaluate error statistics)
                encoder_input  = X_vld[:]
                z            = encoder.predict(encoder_input)
                decoded_output = decoder.predict(z)                
                X_vld_t           = np.moveaxis(X_vld, 0, 1)
                decoded_output_t  = np.moveaxis(decoded_output, 0, 1)
                for i in range(n_channels):
                    to_save_vld_i = compute_errors(X_vld_t[:,:,i],decoded_output_t[:,:,i])
                    if i == 0:
                        to_save_vld = to_save_vld_i
                    else:
                        to_save_vld = np.concatenate([to_save_vld,to_save_vld_i],axis=1 )
                params1_vld  = np.expand_dims(params_vld[:,0], axis=1)
                params2_vld  = np.expand_dims(params_vld[:,1], axis=1)
                to_save_vld  = np.concatenate([to_save_vld,params1_vld,params2_vld],axis=1)
                #plot_dispersion_stat.m
                rec_stat_n_vld = save_root_ID + '\\reconstruction_statics_vld_' + save_ID_network + '.csv'
                np.savetxt(rec_stat_n_vld, to_save_vld, delimiter=",")

            # evaluate latent variable activation if fnn heuristics is used %%%%%%%%%%%%%
            if fnn_loss_if:
                # training set %%%%%%%%%%%%%% (evaluate latent variable activation)
                encoder_input  = X_tr[:]
                z            = encoder.predict(encoder_input)
                for i in range(latent_dim):
                    z_tr_std_i   = np.std(z[:,i])
                    z_tr_std_i   = np.expand_dims(z_tr_std_i, axis=0)
                    if i == 0:
                        z_tr_std = z_tr_std_i
                    else:
                        z_tr_std = np.concatenate([z_tr_std,z_tr_std_i],axis=0)
                lat_val_tr = save_root_ID + '\\latent_variances_tr_' + save_ID_network + '.csv'
                np.savetxt(lat_val_tr, z_tr_std, delimiter=",")

                # validation set %%%%%%%%%%%%% (evaluate latent variable activation)
                encoder_input  = X_vld[:]
                z            = encoder.predict(encoder_input)
                for i in range(latent_dim):
                    z_vld_std_i   = np.std(z[:,i])
                    z_vld_std_i   = np.expand_dims(z_vld_std_i, axis=0)
                    if i == 0:
                        z_vld_std = z_vld_std_i
                    else:
                        z_vld_std = np.concatenate([z_vld_std,z_vld_std_i],axis=0)
                lat_val_vld = save_root_ID + '\\latent_variances_vld_' + save_ID_network + '.csv'
                np.savetxt(lat_val_vld, z_vld_std, delimiter=",")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# further postprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if plot_box_whiskers or plot_lat_variable or plot_lat_var_all or plot_corr_heat:
        z = encoder.predict(X_test)

    # Plot box-whiskers representation of the latent variables (test set) %%%%%%%%%%%%%%%%%%%%%
    if plot_box_whiskers:
        matplotlib.rc('font',  size=4, family='sans-serif')
        matplotlib.rc('axes',  titlesize=8)
        matplotlib.rc('axes',  linewidth=0.01)
        matplotlib.rc('text',  usetex=True)
        matplotlib.rc('lines', linewidth=0.0001)
        matplotlib.rc('lines', markersize=0.00001)
        boxprops     = dict(linewidth=0.001)
        whiskerprops = dict(linewidth=0.00001, color='black')
        medianprops  = dict(linewidth=0.001)
        capprops     = dict(linewidth=0.0001)
        fig, ax = plt.subplots()
        ax.boxplot(z[:,:], showfliers=False, boxprops=boxprops,medianprops=medianprops,whiskerprops=whiskerprops,capprops=capprops)

        fig_save_string1  = save_root_ID + '\\box_plot_' + save_ID_network + '.png'
        fig_save_string2  = save_root_ID + '\\box_plot_' + save_ID_network + '.pdf'
        plt.savefig(fig_save_string1, bbox_inches='tight')
        plt.savefig(fig_save_string2, bbox_inches='tight')
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Plot box-whiskers and dispersion graphs for some latent variables (test set) %%%%%%%%%%%%%%
    if plot_lat_variable:
        lat_n = 4
        matplotlib.rc('font',  size=4, family='sans-serif')
        matplotlib.rc('axes',  titlesize=8)
        matplotlib.rc('axes',  linewidth=0.0001)
        matplotlib.rc('text',  usetex=True)
        matplotlib.rc('lines', linewidth=0.000001)
        matplotlib.rc('lines', markersize=0.000001)
        boxprops     = dict(linewidth=0.0001)
        flierprops   = dict(markersize=0.000001)
        medianprops  = dict(linewidth=0.00001)
        whiskerprops = dict(linewidth=0.00001, color='black')
        capprops     = dict(linewidth=0.000001)
        fig, axs = plt.subplots(lat_n, lat_n)
        for i in range(lat_n):
            for j in range(lat_n):
                if i==j:
                    axs[i,i].boxplot(z[:,i],boxprops=boxprops,flierprops=flierprops,medianprops=medianprops,whiskerprops=whiskerprops,capprops=capprops)
                else:
                    axs[i,j].scatter(z[:,i],z[:,j],s=0.0000001,c='k')
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])
        fig_save_string1  = save_root_ID + '\\scatter_covariance_' + save_ID_network + '.png'
        fig_save_string2  = save_root_ID + '\\scatter_covariance_' + save_ID_network + '.pdf'
        plt.savefig(fig_save_string1, bbox_inches='tight')
        plt.savefig(fig_save_string2, bbox_inches='tight')
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    # Plot box-whiskers and dispersion graphs for all the latent variables %%%%%%%%%%%%%%%%%%%%%%
    # test set
    if plot_lat_var_all:
        lat_n = 2 # = latent_dim
        matplotlib.rc('font',  size=4, family='sans-serif')
        matplotlib.rc('axes',  titlesize=8)
        matplotlib.rc('axes',  linewidth=0.0001)
        matplotlib.rc('text',  usetex=True)
        matplotlib.rc('lines', linewidth=0.000001)
        matplotlib.rc('lines', markersize=0.0000001)
        boxprops     = dict(linewidth=0.0001)
        flierprops   = dict(markersize=0.00001)
        medianprops  = dict(linewidth=0.00001)
        whiskerprops = dict(linewidth=0.00001, color='black')
        capprops     = dict(linewidth=0.000001)
        fig, axs = plt.subplots(lat_n, lat_n)
        for i in range(lat_n):
            for j in range(lat_n):
                if i==j:
                    axs[i,i].boxplot(z[:,i],boxprops=boxprops,flierprops=flierprops,medianprops=medianprops,whiskerprops=whiskerprops,capprops=capprops)
                else:
                    axs[i,j].scatter(z[:,i],z[:,j],s=0.0000001,c='k')
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])

        fig_save_string1  = save_root_ID + '\\scatter_covariance_all_' + save_ID_network + '.png'
        fig_save_string2  = save_root_ID + '\\scatter_covariance_all_' + save_ID_network + '.pdf'
        plt.savefig(fig_save_string1, bbox_inches='tight')
        plt.savefig(fig_save_string2, bbox_inches='tight')
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # plot correlation between latent variables as heat map (test set) %%%%%%%%%%%%%%%%%%%%%%%%%%
    if plot_corr_heat:
        fig, ax = plt.subplots()
        matplotlib.rc('font',  size=4, family='sans-serif')
        matplotlib.rc('axes',  titlesize=8)
        matplotlib.rc('axes',  linewidth=0.0001)
        matplotlib.rc('text',  usetex=True)

        z = np.transpose(z)
        corr_matrix = np.corrcoef(z)
        corr_mat_r  = np.around(corr_matrix,decimals=1)
        min_corr    = np.amin(corr_matrix)        
        avg_corr    = (1+min_corr)/2
        im = ax.imshow(corr_matrix, cmap='bone')
        ax.set_xticks(np.arange(latent_dim))
        ax.set_yticks(np.arange(latent_dim))
        ax.set_title("Correlation matrix")
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        for i in range(latent_dim):
            for j in range(latent_dim):
                if corr_matrix[i, j] > avg_corr:
                    text = ax.text(i, j, corr_mat_r[i, j], ha="center", va="center", color="k")
                else:
                    text = ax.text(i, j, corr_mat_r[i, j], ha="center", va="center", color="w")

        fig_save_string1  = save_root_ID + '\\correlation_matrix_' + save_ID_network + '.png'
        fig_save_string2  = save_root_ID + '\\correlation_matrix_' + save_ID_network + '.pdf'
        plt.savefig(fig_save_string1, bbox_inches='tight')
        plt.savefig(fig_save_string2, bbox_inches='tight')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%