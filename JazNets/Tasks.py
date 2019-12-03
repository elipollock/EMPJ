'''
Tasks.py: Script with functions that give time series for common tasks.
'''

import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
import time

def RSG(dt = 0.001, ts_min=0.5, ts_max=2, ts_time=[], iti_time=[], showplots=0, algorithm=[]):
    '''
    Generates time series for RSG task. Will randomly generate time series based ts_min and ts_max
    Inputs:
    	dt:			Time step
    	ts_min:		Minimum sample interval time (for random selection)
    	ts_max:		Maximum ""
    	ts_time:	Define the sample interval by hand
    	iti_time:	Inter-trial interval. Goes at the end of the time series
    	showplots:	If 1, creates a plot showing target, inputs, and hints
        algorithm:  Networks trained by different algorithms may require different inputs. Specify here
                    'full-FORCE':  Will be for the full-FORCE training algorithm
                    'grad':       For the gradient-based approach

    Outputs:
    	inp:		For input into network
    	targ:		Target output
    	hints: 		Hints for full-FORCE training
    	t:			A time vector
    	ts_time:	The sample interval. Useful for keeping track of what gets randomly generated
        targ_idx:   Indices where target is specified
    
    '''

    # Check that the algorithm is valid
    if algorithm!='full-FORCE' and algorithm!='grad':
        raise ValueError('Please choose a valid training algorithm setting. See documentation for details')

    t_fix = 0.2+0.2*npr.rand() # "Fixation time" at start of trial
    pulse_time = 0.02  # Width of the input pulses
    t_decay = 0.3  # Time to decay in the target
    hint_slope = 0.5  # Slope of the hint function
    mean_iti = 1  # Mean inter-trial interval
    pulse_scale = 0.4 # Magnitude of pulses
    tonic_mean = 0.2 + 0.05*(npr.rand()*2-1)
    tonic_std = 0.005
    
    if not ts_time:
        ts_time = npr.rand()*(ts_max-ts_min) + ts_min   # Sample period
    if not iti_time and iti_time!=0:
        if algorithm=='grad':
            iti_time=dt
        else:
            iti_time = dt + npr.exponential(scale=mean_iti)  # Inter-trial interval
    total_time = t_fix + 2*ts_time +t_decay+ iti_time
    total_steps = int(round(total_time/dt))
    t = np.expand_dims(np.arange(0,total_steps,1),axis=1)*dt
    
    pulses = np.zeros((total_steps,1))
    '''
    if algorithm=='full-FORCE':
        pulses = np.zeros((total_steps,1))
    elif algorithm=='grad':
        pulses = np.zeros((total_steps,2))
    '''
    
    targ = np.zeros((total_steps,1))
    hints = np.zeros((total_steps,1))
    
    ts_steps = int(round(ts_time/dt))
    fix_steps = int(round(t_fix/dt))
    decay_steps = int(round(t_decay/dt))
    pulse_steps = int(round(pulse_time/dt))
    ready = fix_steps
    set1 = ready + ts_steps
    go = set1 + ts_steps
    
    pulses[ready:(ready+pulse_steps),0] = pulse_scale
    pulses[set1:(set1+pulse_steps),0] = pulse_scale
    #pulses[set1:(set1+pulse_steps),1] = pulse_scale
    rampstart = set1+pulse_steps
    targ[rampstart:go,0] = np.linspace(0,0.5,go-rampstart)
    targ[go:go+decay_steps,0] = (np.exp(-t[0:decay_steps,0]*20))/2
    
    hints[ready:set1] = np.expand_dims(np.arange(0,ts_steps)*dt*hint_slope, axis=1)
    hints[set1:go] = hints[set1-1]

    if algorithm=='full-FORCE':
        inp = pulses
        targ = targ-0.5
        targ_idx = np.arange(len(targ))
    elif algorithm=='grad':
        targ = targ*2
        tonic = tonic_mean + npr.randn(total_steps,1)*tonic_std
        inp = np.hstack((pulses,tonic))
        targ_idx = np.arange(set1,go)
    
    if showplots:
        plt.figure()
        plt.plot(t,inp, 'b')
        plt.plot(t,targ, '-r')
        plt.title('$t_s$ = %g' % ts_time)
        if algorithm=='full-FORCE':
            plt.plot(t,hints, '--g')
        plt.show()

    inps_and_targs = {'inps':inp, 'targs':targ, 'hints':hints, 't':t, 'ts_time':ts_time, 'targ_idx':targ_idx}
    return inps_and_targs


def RHY(dt = 0.001, ints=[], n_ints=0, cont=0, int_min = 0.5, int_max = 1, 
    tw = [], tw_min = 0.5, tw_max = 1.5, iti_time=[], showplots=0, algorithm=[]):
    
    '''
    Generates time series for rhythm replication. Rhythms can be specified with ints (short for intervals), or randomly selected using n_ints. By default, the random selection will generate rhythms with integer-ratioed intervals, with either 1, 2, or 3 beats in between each tap.
        Inputs:
            dt:         Time step (default 0.001)
            ints:       List containing the lengths (in seconds) of the intervals for the rhythm. Default empty.
            n_ints:     Number of intervals to randomly generate, if you leave ints blank. Default 0.
            cont:       If 1, draw intervals from a continuous uniform distribution. Default 0
            int_min:	Minimum of random interval
            int_max: 	Maximum of random interval
            tw:         Trigger wait. Time between the last tap and the trigger. Random if empty (default)
			tw_min:		Minimum of random trigger wait
			tw_max: 	Maximum of random trigger wait
            showplots:  If 1, creates a plot showing target, inputs, and hints

        Outputs:
            inp:        For input into network. Will have two columns (rhythm and trigger)
            targ:       Target output. Four columns (one for each tap)
            hints:      Hints for full-FORCE training. Three columns (one for each interval)
            t:          A time vector (useful for plotting).
            targ_idx:   Indices where target is specified
    '''
    def add_pulse(time, x):
        # Inserts a pulse at specified time
        x[time:time+int(round(tap_width/dt))] = pulse_scale
        return x
    
    def add_ramp(start, end, x):
        # Inserts a ramp up to 1 between start and end in x
        slope = 1/(end-start)
        downtime = int(round(reaction_time/dt))
        x[start:end,0] = slope*np.arange(end-start)
        x[end:(end+downtime),0] = 1-1/downtime*np.arange(downtime)
        return x
    
    # Check that the algorithm is valid
    if algorithm!='full-FORCE' and algorithm!='grad':
        raise ValueError('Please choose a valid training algorithm setting. See documentation for details')

    # Parameters not set in inputs:
    reaction_time = 0.25#Time after the trigger before replay
    beat = 0.25         #Length of one beat
    tap_width = 0.02    #Length of a tap
    t_fix = 0.3         #"Fixation" time at beginning of trial
    pulse_scale = 0.4 # Magnitude of pulses
    tonic_mean = 0.1
    tonic_std = 0.005
    
    '''If the intervals themselves are not specified, we can generate random ones using n_ints
    and cont. Without n_ints, we cannot do anything, so we need an error'''
    if not ints and not n_ints:
        raise ValueError('You need to specify either the intervals or the number of intervals!')
    elif not ints:
        # Randomly generate intervals if intervals not specified
        
        if cont:
            ints = int_min + (int_max - int_min)*npr.rand(n_ints)
        else:
            ints = npr.choice([round(beat*x,3) for x in [1,2,3]], size=n_ints)
    else:
        n_ints = len(ints)

    if not iti_time and iti_time!=0:
        if algorithm=='grad':
            iti_time=dt
        else:
            iti_time = (1+npr.rand())*0.5 # Inter-trial interval

    # Convert times to array indices
    beat_length = int(round(beat/dt))
    if not tw:
        tw_length = int(round((tw_min + (tw_max-tw_min)*npr.rand())/dt)) # Trigger wait
    else:
        tw_length = int(round(tw/dt))
    fix_length = int(round(t_fix/dt))
    inp_length = int(round(sum(ints)/dt))
    rt_length = int(round(reaction_time/dt))   # Reaction time
    iti_length = int(round(iti_time/dt)) # Inter-trial interval
    total_length = fix_length+2*inp_length + tw_length + rt_length + iti_length
    t = np.expand_dims(np.arange(0,total_length,1),axis=1) * dt

    taps = [fix_length]
    taptimes = np.cumsum(ints)
    for tt in taptimes:
        taps.append(fix_length+int(round(tt/dt)))
    tap_ints = np.diff(taps)

    # Initialize outputs
    rhythm = np.zeros((total_length,1))
    int_code = np.zeros((total_length, n_ints))
    trigger = np.zeros((total_length,1))
    replay = np.zeros((total_length,n_ints))
    
    trigger_time = fix_length+inp_length+tw_length
    replay_start = trigger_time

    trigger = add_pulse(trigger_time, trigger)
    
    for i in range(len(taps)):
        rhythm = add_pulse(taps[i],rhythm)
    
    for i in range(len(tap_ints)):
        int_code[:,[i]] = add_ramp(taps[i],taps[i+1],int_code[:,[i]])*tap_ints[i]/(4*beat_length)
        int_code[taps[i+1]:taps[i+1]+inp_length+tw_length ,[i]] = int_code[taps[i+1],i]
        replay[:,[i]] = add_ramp(taps[i]+replay_start-fix_length,
                             taps[i+1]+replay_start-fix_length, replay[:,[i]])
    targ = replay
    hints = int_code

    if algorithm=='grad':
        tonic = tonic_mean + npr.randn(total_length,1)*tonic_std
        inp = np.hstack((rhythm, trigger, tonic))
        targ_idx = np.arange(replay_start,len(targ)-rt_length)
    elif algorithm=='full-FORCE':
        targ = targ-1 # Shift the replay down (helps for full-FORCE learning)
        inp = np.hstack((rhythm, trigger))
        targ_idx = np.arange(len(targ))
    
    if showplots==1:

        plt.figure()
        plt.plot(t,inp,'b'); 
        plt.plot(t,targ,'r');  
        plt.plot(t,hints,'--g'); 
        plt.title('All time series')
        plt.xlabel('Time (s)')
        plt.show()
    

    inps_and_targs = {'inps':inp,'targs':targ,'hints':hints,'t':t,'targ_idx':targ_idx}
    
    return inps_and_targs



def WMR(dt=0.001, theta=float('nan'), delay_time=[], delay_max=2, delay_min=1,
                   showplots=0, prior='uniform',algorithm=[]):

    '''
    Generates time series for the Working Memory Ring task. Network must remember a stimulus that lies on a ring, and 
    reproduce it at a specified time.
        Inputs:
            dt:         Time step (default 0.001)
            theta:      Angle of stimulus. Should be between -pi and pi. Randomly generated if not specified
            delay_time: How long to hold the interval in memory. If not specified, randomly generated.
            delay_max:  Used to randomly generate a delay
            delay_min:  Same.
            showplots:  If 1, creates a plot showing target, inputs, and hints
            prior:      Specifies type of distribution from which theta is drawn.
                        'uniform': Default, uniform from -pi to pi
                        'four':    Mixture of 4 Gaussians between -pi and pi, periodic BCs
                        'six':     Same as above, but with 6 slightly narrower Gaussians
            algorithm:  Specify which you need inputs for, as the time series might very
                        'grad'
                        'full-FORCE'

        Outputs:
            inp:        For input into network. Will have two columns (rhythm and trigger)
            targ:       Target output. Four columns (one for each tap)
            hints:      Hints for full-FORCE training. Three columns (one for each interval)
            theta:      Stimulus angle
            targ_idx:   Indices where target is specified
            response_idx:   Index where response should be recorded
            trigger_idx:    Start of response trigger
            stim_idx:       Where the stimulus begins
            delay_idx:      Where the delay begins
    '''
    
    def TTS(T,dt):
    # Convert Time to Steps
        return int(round(T/dt))

    # Check that the algorithm is valid
    if algorithm!='full-FORCE' and algorithm!='grad':
        raise ValueError('Please choose a valid training algorithm setting. See documentation for details')

    # If theta isn't specified, choose it from some distribution
    if np.isnan(theta):
        if prior=='uniform':
            theta = np.pi*(np.random.rand()*2-1)
        
        # Biased:
        elif prior=='four':
            q = np.random.choice(np.arange(0,1,1/4))
            theta = (np.pi*(2*np.mod(np.random.normal(q,0.06),1)-1))
        
        elif prior=='six':
            q = np.random.choice(np.arange(0,1,1/6))
            theta = (np.pi*(2*np.mod(np.random.normal(q,0.04),1)-1))
    
    # Pick a delay time, if it's not specified
    if not delay_time:
        delay_time = npr.rand()*(delay_max-delay_min) + delay_min
            
    x = np.cos(theta)
    y = np.sin(theta)
    
    fix_time = 0.3
    sample_time = 0.2
    trigger_time = 0.1
    reaction_time = 0.2
    response_time = 0.2
    if algorithm=='full-FORCE':
        iti_time = 0.3
    else:
        iti_time=0
    
        
    fix_steps = TTS(fix_time,dt)
    sample_steps = TTS(sample_time,dt)
    delay_steps = TTS(delay_time,dt)
    trigger_steps = TTS(trigger_time,dt)
    reaction_steps = TTS(reaction_time,dt)
    response_steps = TTS(response_time,dt)
    iti_steps = TTS(iti_time,dt)
    total_steps = (fix_steps+sample_steps+delay_steps+trigger_steps+
                   reaction_steps+response_steps+iti_steps)
    
    show_stim = fix_steps
    show_trigger = show_stim+sample_steps+delay_steps
    show_response = show_trigger+trigger_steps+reaction_steps
    
    
    x_input = np.zeros((total_steps,1))
    y_input = np.zeros((total_steps,1))
    trigger = np.zeros((total_steps,1))
    
    x_targ = np.zeros((total_steps,1))
    y_targ = np.zeros((total_steps,1))
    x_hint = np.zeros((total_steps,1))
    y_hint = np.zeros((total_steps,1))
    
    
    x_input[show_stim:show_stim+sample_steps,0] = x
    y_input[show_stim:show_stim+sample_steps,0] = y
    trigger[show_trigger:show_trigger+trigger_steps,0] = 1
    
    x_targ[show_trigger+trigger_steps:show_response,0] = np.linspace(0,x,reaction_steps)
    x_targ[show_response:show_response+response_steps,0] = np.linspace(x,0,response_steps)
    y_targ[show_trigger+trigger_steps:show_response,0] = np.linspace(0,y,reaction_steps)
    y_targ[show_response:show_response+response_steps,0] = np.linspace(y,0,response_steps)
    
    x_hint[show_stim:show_trigger,0] = x/2
    y_hint[show_stim:show_trigger,0] = y/2
    
    
    inputs = np.hstack((x_input,y_input,trigger))
    targets = np.hstack((x_targ,y_targ))
    hints = np.hstack((x_hint,y_hint))
    
    targ_idx = np.arange(show_trigger+trigger_steps,show_response+response_steps)
    
    if showplots:
        plt.figure()
        plt.plot(inputs,'b')
        plt.plot(targets,'r--')
        plt.plot(hints,'g--')
        plt.plot()
        plt.show()
        
    inps_and_targs = {'inps':inputs,'targs':targets,'hints':hints,
                      'targ_idx':targ_idx, 'theta':theta, 'response_idx': show_response, 
                      'trigger_idx': show_trigger, 'stim_idx': show_stim, 
                      'delay_idx': show_stim+sample_steps}
    return inps_and_targs