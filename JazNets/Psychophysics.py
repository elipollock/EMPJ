'''
Psychophysics.py: Script containing functions to measure relevant psychophysical metrics for various tasks.
'''

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import time

def dRSG_metronomic(RNN, ntrials=1, threshold=0.1, showtrialplots=1, **kwargs):
    '''
    Function designed for the delayed ready-set-go (dRSG) task. The network should be 
    trained using the RHY task, with n_ints=1 and cont=1 (one interval, varying continuously).
	The function will plot the "metronomic" curve for the task, showing the relationship between the 
	sampled interval (t_s) and the produced interval (t_p). The function draws values of t_s from the
	default range for the RHY function.
    Inputs:
        RNN:				The trained JazNet.
        ntrials:			Number of trials to use per t_s.
        threshold:			Error threshold for considering a trial to be a "success"
        showtrialplots:		Determines if a plot is created showing the network's output for each condition. 

	Outputs:
		ts:					Array with ntrials columns, where each row is a different interval, containing sample intervals
		tp:					Same as above for produced intervals
		Also produces a scatter plot showing the "metronomic curve" (ts vs tp)
    '''
    from JazNets.Tasks import RHY
    import inspect
    
    # Initialize network
    RNN.initialize_act()
    init_trials = 3
    print('Initializing',end="")
    for init_trial in range(init_trials):
        inp = RHY(n_ints=1, cont=1, **kwargs)[0]
        RNN.run(inp)
        print('.',end="")
    print("")
    
    # Set values of the intervals (based on defaults from Tasks.RHY)
    argspecs = inspect.getfullargspec(RHY)

    int_min = [argspecs.defaults[i] for i in range(len(argspecs)) 
                if argspecs.args[i]=='int_min']
    int_max = [argspecs.defaults[i] for i in range(len(argspecs)) 
                if argspecs.args[i]=='int_max']
    int_times = np.linspace(int_min,int_max,11)
    npatterns = len(int_times)
    
    # Initialize outputs
    ts = np.zeros((npatterns,ntrials))
    tp = np.zeros((npatterns,ntrials))

    dt = RNN.p['dt']
    
    if showtrialplots:
        out_fig = plt.figure()
        out_ax = out_fig.add_subplot(111)
    
    for pat_idx in range(npatterns):   # Iterate over interval patterns
        pat = int_times[pat_idx]
        if showtrialplots:
            inp, targ = RHY(ints=[pat], **kwargs)[0:2]
            trig = np.argmax(inp[:,1])
            out_ax.plot(inp[trig:], 'g--')
            out_ax.plot(1+targ[trig:], 'r--')
        else:
            print('Interval: %gs' % pat, end="")
        for trial in range(ntrials):
            s = 0
            nopes=0
            while not s:  # If you aren't successful at training
                inp, targ = RHY(ints=[pat], **kwargs)[0:2]
                out = RNN.run(inp)[0]
                error = np.mean((targ-out)**2)/np.mean((targ)**2)
                #print(error) # Use this line to see the error if the network is failing a lot
                if error<threshold:  # Consider it a success if you are below some threshold
                    s=1

                    ts[pat_idx,trial] = (np.argmax(targ[:,1]) - np.argmax(targ[:,0]))*dt
                    tp[pat_idx,trial] = (np.argmax(out[:,1])-np.argmax(out[:,0]))*dt
                    
                    if showtrialplots:
                        trig = np.argmax(inp[:,1])
                        out_ax.plot(1+out[trig:], 'b', alpha=0.2) 
                        out_ax.set_title(pat)
                        out_fig.canvas.draw()
                    elif not trial % (ntrials/50):
                        print('.',end="")
                else:
                    print(',',end="")
                    nopes += 1
                    if nopes>100:
                    	raise RuntimeError('Cannot get a successful run! (error too high)')
        if not showtrialplots:
            print("")
    
    # Make metronomic curve
    plt.figure()
    plt.scatter(ts.flatten(), tp.flatten(), alpha=0.1, marker='.')
    plt.plot(ts[:,0],ts[:,0],'--')
    plt.title('Metronomic curve')
    plt.xlabel('$t_s$')
    plt.ylabel('$t_p$')
    plt.show()
    return ts, tp