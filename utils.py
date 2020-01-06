import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rc
import tensorflow as tf

def generate_simulation(model_parts, x_test, obj_test, frame_num = 100):
    """
    Using the trained model, returns the predicted LBM distribution over the
    next time steps

    Parameters
    ----------
    x_test : First LBM frame of the simulation, with shape (1, N, N, 9)
    obj_test : Boolean object matrix present in the simulation, with shape
        (1, N, N, 1)
    frame_num : integer, optional
        The ammount of simulation frames to be generated

    Returns
    ----------
    predic : The predicted LBM distribution with shape (frame_num, N, N, 9)

    """
    model_encoder_state, model_decoder_state, model_compression_mapping = model_parts
    imgSize = x_test.shape[1]    
    predic = np.zeros((frame_num,imgSize,imgSize,9))
    predic[0] = x_test[0:1]
    obj = obj_test[0:1]
    compressed, b_add, b_mul = model_encoder_state.predict([predic[0:1], obj])
    for i in range(frame_num-1):
        predic[i+1] = model_decoder_state.predict([compressed])
        compressed, _, _ = model_compression_mapping.predict([compressed, b_add, b_mul]) 
    return predic
	
def get_velocity(fin):
    """
    Computes the velocity of the fluid, given its LBM distribution

    Parameters
    ----------
    fin : The LBM density distribution, with shape (1, N, N, 9)

    Returns
    ----------
    vel : Two components of the velocity of the fluid, with shape (1, N, N, 2)

    """
    rho = np.sum(fin, axis=2)
    vel = np.zeros((1,fin.shape[0],fin.shape[1],2))
    vel[0,:,:,0] = (fin[:,:,1] + fin[:,:,5] + fin[:,:,8] - fin[:,:,3] - fin[:,:,6] - fin[:,:,7]) / rho  
    vel[0,:,:,1] = (fin[:,:,2] + fin[:,:,5] + fin[:,:,6] - fin[:,:,4] - fin[:,:,8] - fin[:,:,7]) / rho
    return vel

def get_divergence(fin):
    """
    Computes the divergence of the velocity of the fluid, given its LBM 
    distribution

    Parameters
    ----------
    fin : The LBM density distribution, with shape (1, N, N, 9)

    Returns
    ----------
    div : The divergence of the velocity of the fluid, with shape (N, N)
    
    """
    div_filter = np.zeros((3, 3, 2, 1))
    div_filter[1,0,0,0] = -1; div_filter[1,2,0,0] = 1 #div x
    div_filter[0,1,1,0] = 1; div_filter[2,1,1,0] = -1 #div y
    vel = get_velocity(fin)
    div = np.asarray(tf.nn.conv2d(vel, div_filter, strides=[1,1,1,1], padding='VALID'))
    return div[0,:,:,0]

def get_curl(fin):
    """
    Computes the curl of the velocity of the fluid, given its LBM distribution

    Parameters
    ----------
    fin : The LBM density distribution, with shape (1, N, N, 9)

    Returns
    ----------
    curl : The curl of the velocity of the fluid, with shape (N, N)
    
    """
    vel = get_velocity(fin)
    return (np.roll(vel[0,:,:,1],-1,axis=1) - np.roll(vel[0,:,:,1],1,axis=1) - np.roll(vel[0,:,:,0],-1,axis=0) + np.roll(vel[0,:,:,0],1,axis=0))
	
def make_animation(y_hat, x_test = [], plot = 'vel'):
    """
    Returns an animation, given a LBM distribution

    Parameters
    ----------
    y_hat : An array of a LBM distribution (m, N, N, 9)
    x_test : optional
        Whether to visualize the original animation side by side with the
        generated one
    plot : {'vel', 'div', 'curl'}, optional
        The desired velocity property to be visualized

    Returns
    ----------
    anim : matplotlib.animation object

    """
    x_test = np.asarray(x_test)
    if x_test:    
        predicConc = np.concatenate((y_hat, x_test), axis=2)
    else:
        predicConc = y_hat

    current_x = predicConc[15] #setup color normalization frame
    
    if plot == 'curl':
        u_plot = get_curl(current_x)
    elif plot == 'div':
        u_plot = get_divergence(current_x)
    else:        
        vel = get_velocity(current_x)
        u_plot = np.abs(vel[0,:,:,0], vel[0,:,:,1])

    fig, ax = plt.subplots()
    img = plt.imshow(u_plot, cmap='viridis')
    plt.close()

    def animate(i):
        current_x = predicConc[i]

        if plot == 'curl':
            u_plot = get_curl(current_x)
        elif plot == 'div':
            u_plot = get_divergence(current_x)
        else:
            vel = get_velocity(current_x)
            u_plot = np.abs(vel[0,:,:,0], vel[0,:,:,1])
        
        img.set_array(u_plot)
        
        return (img,)

    anim = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=True)
    rc('animation', html='jshtml')
    return anim