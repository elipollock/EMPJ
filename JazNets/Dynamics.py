'''
Dynamics.py: Script with functions for analyzing the dynamics of the JazNet
'''

import matplotlib.pyplot as plt
import time
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

def fixed_points(rnn, inp, num_points=1, eps = 0.01, opt_iters=10000, thresh=1, max_tries=100, 
                      rand_init=1, init_scale=5, plot_loss=0):
    '''This function uses the trained parameters to find num_points fixed points. It does a gradient
    descent to minimize q(x), which is analagous to the energy of the system. To just plot the gradient descent loss
    and step size for finding a single fixed point,  set the plot_loss flag to 1.
    Inputs:
        rnn: Should be a JazNet class object.
        inp: A fixed value for the input(s). Can just be a list (e.g. [1,0])
        num_points: Number of points to find (if plot_loss=0)
        eps: Epsilon value that scales the step size
        opt_iters: How many iterations to run to try to converge on a fixed point
        thresh: Threshold for the norm of the network activity before calling it a fixed point
        rand_init: Randomly pick a starting point if 1 (default), otherwise go with the network's current activity.
        plot_loss: Will result in only finding one fixed point. Shows how loss function/step size changes. Default 0

    Outputs:
        all_points: Gives activity for all fixed points found in a num_points-by-N array
        fp_outputs: Network output at each fixed point. Note: Should change this depending on
            whether network uses tanh of activities for outpus, or if it has biases.
        trajectories: List with num_points elements, where each element is a TxN array, where T is the number of 
        steps it took to find the fixed point and N is the number of neurons.
        '''
    def output(x):
        return np.dot(np.tanh(x),rnn_par['out_weights'])
    
    def F(x):
        return (-x + np.dot(np.tanh(x),rnn_par['rec_weights']) + 
                np.dot(inp, rnn_par['inp_weights']) + rnn_par['bias'])
    
    def q(x):
        return 1/2 * np.linalg.norm(F(x))**2
    
    def find_point(inp, opt_iters, eps):
        loss = []
        stepsize = []
        x_traj = []
        if rand_init:
            x = np.random.randn(rnn.act.size)*init_scale   # The randomized initial activity needs to be big enough to relax to interesting points
        else:
            x = np.squeeze(rnn.act)
        for i in range(opt_iters):
            loss.append(q(x))
            if loss[i]<thresh:
                break
            step = eps*loss_grad(x)
            stepsize.append(np.linalg.norm(step))
            x = x-step
            x_traj.append(x)
        return x, loss, stepsize, x_traj
    
    start = time.time()
    rnn_par = rnn.rnn_par # Extract the parameters
    loss_grad = grad(q)

    if plot_loss:  # To see the optimization process to find one fixed point
        x, loss, stepsize, x_traj = find_point(inp, opt_iters, eps)
        plt.figure()
        plt.subplot(1,3,1)
        plt.plot(loss[-100:-1])
        plt.title('Loss, last 100')
        plt.subplot(1,3,2)
        plt.plot(loss)
        plt.xlabel('Iteration')
        plt.title('Loss, all')
        plt.subplot(1,3,3)
        plt.plot(stepsize)
        plt.xlabel('Iteration')
        plt.title('Step size')
        plt.show()
        print('Last loss:',loss[-1])
    else:    # For finding a bunch of fixed points
        all_points = np.zeros((num_points,np.size(rnn.act)))
        fp_outputs = np.zeros((num_points,rnn_par['out_weights'].shape[1]))
        trajectories = []
        for p in range(num_points):
            endloss = 1000 # Some big value above the threshold
            tries = 0
            while endloss>thresh:  
                if tries<max_tries:
                    x, loss, stepsize, x_traj = find_point(inp, opt_iters, eps)
                    endloss = loss[-1]
                    tries+=1
                else:
                    print('Unsuccessful run; error=%g' % endloss)
                    raise TimeoutError('No fixed points found in %d tries' % max_tries)
            all_points[p,:] = x
            fp_outputs[p] = output(x)
            trajectories.append(np.array(x_traj))
            print('.',end="")
        finish = time.time()
        print('Done with fixed points in %d seconds' % (finish-start))
        return all_points, fp_outputs, trajectories



