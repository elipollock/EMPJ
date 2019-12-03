

def param_list_perms_2(p1, p2, reps=1):
	'''
	A small function to create a list of all permutations of 
	parameters from the lists p1 and p2. Can make similar
	functions to do the same for larger numbers of parameters.

	Can repeat each condition reps times, if desired
	'''
	import numpy as np
	params = np.zeros((len(p1)*len(p2)*reps,2))
	for i in range(len(p1)):
		for j in range(len(p2)):
			for r in range(reps):
				params[i*len(p2)*reps + j*reps + r] = [p1[i], p2[j]]
	return params

def param_list_perms_3(p1, p2, p3, reps=1):
	'''
	A small function to create a list of all permutations of 
	parameters from the lists p1, p2, and p3. Can make similar
	functions to do the same for larger numbers of parameters.

	Can repeat each condition reps times, if desired
	'''
	import numpy as np
	params = np.zeros((len(p1)*len(p2)*len(p3)*reps,3))
	for i in range(len(p1)):
		for j in range(len(p2)):
			for k in range(len(p3)):
				for r in range(reps):
					params[i*len(p2)*len(p3)*reps + j*len(p3)*reps + k*reps + r] = [p1[i], p2[j], p3[k]]
	return params

def saveload(mode,filename,*args):
    import pickle 
    
    if mode=='save':
        f = open(filename,'wb')
        pickle.dump(args, f)
        f.close()
    elif mode=='load':
        f = open(filename,'rb')
        var_tuple = pickle.load(f)
        f.close()
        return var_tuple
    else:
        print('Not a valid saveload option!')