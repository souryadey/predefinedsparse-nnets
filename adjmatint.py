#==============================================================================
#  Adjacency matrices (adjmat) for NNs
#  Sourya Dey, USC
#==============================================================================

import warnings
import numpy as np
from math import factorial
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)


#==============================================================================
# Random sparsity : Freely distributing connections randomly given a certain density
#==============================================================================
def adjmat_random(p=8,density=0.5,n=8):
    '''
    Returns an nxp adjacency matrix adjmat which has total d 1s, where d = density*p*n
    IMPORTANT: If density>=1, it's interpeted as fo, so then d = density*p
        But this is not regular fo, i.e. it's not like every column has fo 1s. It is random, but total 1s will be the same as the case of using fixed fo
    If d is not an integer, it is rounded up
    '''
    if density>=1:
        d = density*p
    else:
        d = density*p*n
    d = int(np.ceil(d))
    adjmat = np.zeros(n*p)
    locs = np.random.choice(adjmat.size,d,replace=False)
    adjmat[locs] = 1
    return adjmat.reshape(n,p)


#==============================================================================
# Structured sparsity : Fixed fi, fo
#==============================================================================
def adjmat_basic(p=8,fo=2,n=8):
    '''
    Returns an nxp adjacency matrix adjmat where each row is an output neuron with fi 1s and each column is an input neuron with fo 1s
    '''
    fi = p*fo//n
    adjmat = np.zeros((n,p))
    fo_counter = np.zeros(p) #count the number of 1s in each column
    fo_probs = np.asarray(p*[1.0/p]) #start with uniform probabilities to get a 1 anywhere
    for ni in range(n):
        if ni < (n-fo): #usual case
            fo_probs[fo_probs!=0] = np.count_nonzero(fo_probs)*[1.0/np.count_nonzero(fo_probs)] #nonzero elements should have uniform prob. this line is REUSED
            fi_pattern = np.random.choice(p,size=fi,replace=False,p=fo_probs)
        else: #last fo output neurons
            fi_pattern = np.where(fo_counter <= ni-(n-fo))[0] #input neurons with low fanout must be chosen now
            if len(fi_pattern)!=fi: #if equal, all required positions are in fi_pattern, no need for choice any more
                temp_fo_probs = np.copy(fo_probs)
                temp_fo_probs[fi_pattern] = 0 #since elements that must be chosen are already chosen, make their fo_probs 0 temporarily
                temp_fo_probs[temp_fo_probs!=0] = np.count_nonzero(temp_fo_probs)*[1.0/np.count_nonzero(temp_fo_probs)]
                fi_pattern = np.concatenate((fi_pattern,np.random.choice(p,size=fi-len(fi_pattern),replace=False,p=temp_fo_probs))) #get remaining 1s
        adjmat[ni][fi_pattern] = 1
        fo_counter[fi_pattern] += 1
        fo_probs[fo_counter==fo] = 0 #if a column has fo 1s, it has reached max and should never be chosen again
    return adjmat


#==============================================================================
# Clash freedom
#==============================================================================
def memmap_sweep(p=40,z=10, typ=1):
    ''' Generate all left memory addresses in a SINGLE sweep. For definitions, see wt_interleaver '''
    if typ==1:
        return np.asarray([np.random.permutation(p//z) for _ in range(z)]).T.reshape(p,)
    elif typ==2 or typ==3:
        s = np.random.choice(np.arange(p//z),z) #pick starting addresses of z memories as any number between 0 and p//z-1 (since #rows = p//z)
        return np.asarray([(s+i)%(p//z) for i in range(p//z)]).reshape(p,)

def memdither_sweep(p=40,z=10, memdith=0):
    ''' Generate vector for memory dither in a SINGLE sweep. For definitions, see wt_interleaver '''
    if memdith==0:
        return np.tile(np.arange(z), p//z) #no mem dither, just read mems in order. Make v the same size as t (p elements) for vectorized calculation of wt_interleaver
    elif memdith==1:
        return np.tile(np.random.permutation(z), p//z) #memory dither, read mems in some other order (held constant for all cycles in a sweep)


def wt_interleaver(p,fo,z, typ=2, memdith=0, deinter=None, inp=None):
    '''
    p: Number of neurons in left layer of junction
    fo: Fanout degree of left layer
    z: Degree of parallelism
    typ:
        Type 1: No restrictions
        Type 2: Subsequent addresses in each actmem are +1 modulo
        Type 3: Type 2, and memmap remains same for every sweep
    memdith:
        Introduce additional memory dither v for every sweep. Eg for z=10 :
            For memdith=0, memories would be accessed as [0123456789] in every sweep
            For memdith=1, 1st sweep may access memories as [5803926174], 2nd sweep as [8279306145]
    Returns:
        Interleaver pattern
        De-interleaver pattern if deinter!=None
        Interleaver applied to particular input 'inp' if inp!=None
    '''
    assert (float(p)/z)%1 == 0.0, 'p/z = {0}/{1} = {2} must be an integer'.format(p,z,float(p)/z)

    ## Initial sweep ##
    t = memmap_sweep(p,z,typ)
    v = memdither_sweep(p,z,memdith)
    inter = (t*z+v)*fo

    ## Following sweeps ##
    for i in range(1,fo):
        if typ!=3:
            t = memmap_sweep(p,z,typ) #generate new t
            if memdith!=0:
                v = memdither_sweep(p,z,memdith) #generate new v
        inter = np.append(inter, (t*z+v)*fo + i)

    assert set(range(p*fo)) == set(inter), 'Interleaver is not valid:\n{0}'.format(inter)
    inter = inter.astype('int32')

    if deinter!=None:
        deinter = np.asarray([np.where(inter==i)[0] for i in range(len(inter))])
        deinter = deinter.astype('int32')

    if inp!=None:
        assert len(inp)==p*fo, 'Input size = {0} must be equal to p*fo = {1}'.format(len(inp),p*fo)
        inp = inp[inter]
        inp = inp.astype('int32')

    return (inter, deinter, inp)


def inter_to_adjmat(inter,p,fo,n):
    ''' Convert a weight interleaver 'inter' to an adjacency matrix '''
    fi = p*fo//n
    adjmat = np.zeros((n,p))
    for i in range(len(inter)):
        adjmat[i//fi,inter[i]//fo] = 1
    return adjmat


def adjmat_check_validity(adjmat,fo=2):
    '''
    Checks any adjacency matrix for validity, i.e. each col has fo 1s and each row has fi 1s
    '''
    n,p = adjmat.shape
    fi = p*fo//n
    validity = 'VALID'
    for i in range(n):
        if np.sum(adjmat[i,:]) != fi:
            validity = 'INVALID'
            break
    for i in range(p):
        if np.sum(adjmat[:,i]) != fo:
            validity = 'INVALID'
            break
    if np.max(adjmat)>1:
        validity = 'VALID, but has repeated connections'
    print(validity)


def adjmat_clash_free(p=40,fo=2,n=10, z=10, typ=2, memdith=0, check=0):
    '''
    Generates weight interleaver and converts to adjmat
    These are GUARANTEED to be valid and clash-free, no need to check
    If check is still desired, set check=1:
        For clash freedom, cannot look at adjmat because an integral number of rows will NOT be read every cycle in the general case where z is not an integral multiple of fi
        The only way to check clash-freedom is to use the interleaver pattern to look at the left neurons read every cycle and check that they all come from different memories
    '''
    inter,_,_ = wt_interleaver(p,fo,z, typ, memdith)
    adjmat = inter_to_adjmat(inter, p,fo,n)
    if check==1: ## Check for validity and clash-freedom (only for the paranoid) ##
        adjmat_check_validity(adjmat,fo)
        cf = 'CLASH FREE'
        for i in range(0,len(inter),z):
            if set((inter[i:i+z]//fo)%z) != set(range(z)): #inter[i:i+z]//fo is the set of left neurons read in cycle i, doing %z gets the AM numbers they are in. These should all be different
                cf = 'CLASH in cycle {0}'.format(i)
        print(cf)
    return adjmat


#==============================================================================
# Count number of possible adjmats
#==============================================================================
def count_adjmats(iters=10000, p=40,fo=2,n=10, z=10, typ=2, memdith=0, interval=1000):
    '''
    Count the number of possible CLASH-FREE adjmats for a given config
    intervals: Display progress after every this many iterations
    Prints expected count = 0 if formula is unknown
    Cases:
        If fo=n, i.e. FC, then count=1
        For others, see journal paper
    Tests:
        p=9,fo=2,n=6,z=3, memdith=0 :
            Type 1: 46656
            Type 2: 729
            Type 3: 27
        Same, memdith=1 => fi/z=1 :
            Same results, since fi/z=1
        p=12,fo=2,n=12,z=4, memdith=0 => z/fi=2 :
            Type 1: 60M
            Type 2: 236196
            Type 3: 486
    Parasitic tests:
        p=6,fo=2,n=4,z=2, memdith=1 => fi/z=1.5 :
            If k = z!, Type 3 should give 5184, but gives 324
        p=4,fo=3,n=4,z=2, memdith=1 => fi/z=1.5 :
            If k = z!, Type 2 should give 512, but gives 64
        So k is NOT equal to z! when fi/z > 1 but is not an integer
    '''
    fi = float(p*fo//n)
    k = 0
    if memdith==0 or (fi/z)%1 == 0:
        k = 1
    elif (z/fi)%1 == 0:
        k = factorial(z)/(factorial(fi)**(z/fi))

    if typ==1:
        expected_count = (k*factorial(p//z)**z)**fo
    elif typ==2:
        expected_count = (k*(p//z)**z)**fo
    elif typ==3:
        expected_count = k*(p//z)**z
    print('Expected count = {0}'.format(expected_count))

    adjmats = np.zeros((iters,n*p)) #this will store the adjmats in flattened form
    for i in range(iters):
        if i%interval==0:
            print(i)
        adjmats[i] = adjmat_clash_free(p,fo,n, z, typ, memdith, check=0).flatten()
    actual_count = len(set([tuple(adjmat) for adjmat in adjmats])) #convert each flattened adjmat to a tuple so that Python can compare them for uniqueness
    print('Actual count = {0}'.format(actual_count))






#==============================================================================
#==============================================================================
# # SCATTER (Metric used in ITA 2018 paper to test goodness of adjmats)
#==============================================================================
#==============================================================================

# =============================================================================
# Windowing functions
# =============================================================================
def window1D (dwidth=1024, wwidth=16, swidth=8):
    '''
    Slide a window of some size and stride length over a 1D data vector (could be a rasterized image)
        dwidth: Number of elements in vector
        wwidth: Number of elements in window
        swidth: Stride
    Returns:
        A 2D array of shape (nwidth,wwidth)
        nwidth = Number of windows = (dwidth-wwidth)//swidth + 1
    Each row of the output array are all the indices of the pixels for that window
    Example: Input has size has 1024, window is size 16 and stride is 8
        nwidth = (1024-16)/8+1 = 127
        0th row of output array will be [0,1,2,...,14,15]
        1st row will be shifted by 8, i.e. [8,9,10,...,22,23]
    '''
    if (dwidth-wwidth)%swidth != 0:
        warnings.warn('Number of windows = {0} is not integral, so data will be ignored and there will be {1} windows'.format((dwidth-wwidth)/float(swidth)+1,(dwidth-wwidth)//swidth+1))
    nwidth = (dwidth-wwidth)//swidth + 1
    out = np.zeros((nwidth,wwidth))
    for i in range(nwidth):
        out[i] = np.arange(i*swidth,i*swidth+wwidth)
    return out.astype(int)


def window2D (dwidth=32, dheight=None, wwidth=4, wheight=None, swidth=1, sheight=None):
    '''
    Slide a window of some size and stride length over a 2D data matrix (like an image)
    ASSUMES IMAGE IS STORED IN ROW FORM, i.e. 1st row then 2nd row etc
        dwidth, dheight: Width and height of input data. If dheight = dwidth, set dheight = None
        wwidth, wheight: Width and height of window. If wheight = wwidth, set wheight = None
        swidth, sheight: Width and height of stride. If sheight = swidth, set sheight = None
    Returns:
        A 2D array of shape (nwidth*nheight,wwidth*wheight)
        nwidth, nheight: Number of windows in either direction
        nwidth = (dwidth-wwidth)//swidth + 1. Similarly for nheight
    Each row of the output array are all the indices of the pixels for that window
    Example: Image is 32x32 stored in row format, window is 4x4, stride is either direction is 2
        nwidth = nheight = (32-4)//2+1 = 15
        So there are 15x15 = 225 windows, each of 4x4=16 pixels
        0th row of output array will be left cornermost 16 pixels - [0,1,2,3,32,33,34,35,64,65,66,67,96,97,98,99]
        1st row of output array will be shifted by 2 to the right - [2,3,4,5,34,35,36,37,66,67,68,69,98,99,100,101]
        and so on
    '''
    if dheight==None: dheight=dwidth
    if wheight==None: wheight=wwidth
    if sheight==None: sheight=swidth
    if (dwidth-wwidth)%swidth != 0:
        warnings.warn('Number of width windows = {0} is not integral, so data will be ignored and there will be {1} width windows'.format((dwidth-wwidth)/float(swidth)+1,(dwidth-wwidth)//swidth+1))
    if (dheight-wheight)%sheight != 0:
        warnings.warn('Number of height windows = {0} is not integral, so data will be ignored and there will be {1} height windows'.format((dheight-wheight)/float(sheight)+1,(dheight-wheight)//sheight+1))
    nwidth = (dwidth-wwidth)//swidth + 1
    nheight = (dheight-wheight)//sheight + 1

    def findWindowFromStart(start, wwidth, wheight, dwidth):
        ''' Given a starting pixel index, find indices of all the other pixels in that window '''
        x = np.array([])
        for i in range(wheight):
            x = np.append(x, np.arange(start+(i*dwidth),start+(i*dwidth)+wwidth))
        return x

    out = np.zeros((nwidth*nheight,wwidth*wheight))
    for i in range(nheight):
        for j in range(nwidth):
            out[i*nwidth+j] = findWindowFromStart(i*sheight*dwidth+j*swidth, wwidth, wheight, dwidth)
    return out.astype(int)


def window3D (dwidth=4, dheight=None, ddepth=256, wwidth=2, wheight=None, wdepth=None, swidth=2, sheight=None, sdepth=None):
    '''
    Slide a window of some size and stride length over a 3D data matrix (like an image with different feature maps)
    ASSUMES IMAGE IS STORED IN FEATURE FORM, i.e. OPPOSITE OF DIMENSION ORDER (LIKE IN KERAS FLATTEN)
    i.e. all features for left corner, then all features for pixel MOVING DOWN, etc
        dwidth, dheight, ddepth: Width, height and number of features of input data. If dheight = dwidth, set dheight = None
        wwidth, wheight, wdepth: Width, height and depth of window. If wheight = wwidth, set wheight = None. Same for wdepth
        swidth, sheight, sdepth: Width, height and depth of stride. If sheight = swidth, set sheight = None. Same for sdepth
    Returns:
        A 2D array of shape (nwidth*nheight*ndepth,wwidth*wheight*wdepth)
        nwidth, nheight, ndepth: Number of windows in either direction
        nwidth = (dwidth-wwidth)//swidth + 1. Similarly for nheight, ndepth
    Each row of the output array are all the indices of the pixels for that window
    Example: Image is 4x4x128 stored in flattened format, i.e. opposite of dimensions. Say all strides are 2 and window sizes are 2
        nwidth = nheight = (4-2)//2+1 = 2
        ndepth = (128-2)//2+1 = 64
        So there are 2x2x64 = 256 windows, each of 2x2x2=8 pixels
        0th row of output array will be left cornermost 4 pixels from 1st 2 feature maps - [0,1,128,129,512,513,640,641]
    '''
    if dheight==None: dheight=dwidth
    if ddepth==None: ddepth=dwidth
    if wheight==None: wheight=wwidth
    if wdepth==None: wdepth=wwidth
    if sheight==None: sheight=swidth
    if sdepth==None: sdepth=swidth

    if (dwidth-wwidth)%swidth != 0:
        warnings.warn('Number of width windows = {0} is not integral, so data will be ignored and there will be {1} width windows'.format((dwidth-wwidth)/float(swidth)+1,(dwidth-wwidth)//swidth+1))
    if (dheight-wheight)%sheight != 0:
        warnings.warn('Number of height windows = {0} is not integral, so data will be ignored and there will be {1} height windows'.format((dheight-wheight)/float(sheight)+1,(dheight-wheight)//sheight+1))
    if (ddepth-wdepth)%sdepth != 0:
        warnings.warn('Number of depth windows = {0} is not integral, so data will be ignored and there will be {1} depth windows'.format((ddepth-wdepth)/float(sdepth)+1,(ddepth-wdepth)//sdepth+1))
    nwidth = (dwidth-wwidth)//swidth + 1
    nheight = (dheight-wheight)//sheight + 1
    ndepth = (ddepth-wdepth)//sdepth + 1

    def findWindowFromStart(start, wwidth, wheight, wdepth, dheight, ddepth):
        ''' Given a starting pixel index, find indices of all the other pixels in that window '''
        x = np.array([])
        for i in range(wwidth):
            for j in range(wheight):
                x = np.append(x, np.arange(start+(i*dheight*ddepth + j*ddepth) , start+(i*dheight*ddepth + j*ddepth)+wdepth))
        return x

    out = np.zeros((nwidth*nheight*ndepth,wwidth*wheight*wdepth))
    for i in range(nwidth):
        for j in range(nheight):
            for k in range(ndepth):
                out[i*nheight*ndepth+j*ndepth+k] = findWindowFromStart(i*swidth*dheight*ddepth + j*sheight*ddepth + k*sdepth, wwidth, wheight, wdepth, dheight, ddepth)
    return out.astype(int) 
# =============================================================================


# =============================================================================
# Actual scatter
# =============================================================================
def scatter (adjmat01,adjmat12, adjmat=0, window=window2D(), outmatter=None):
    '''
    Given 2 individual adjmats or the net IO adjmat for a 2-jn network, compute scatter of output neurons w.r.t input windows
    Inputs:
        adjmat01,adjmat12: Either input these and keep adjmat as 0, OR
        adjmat: Input only this and keep adjmat01=adjmat12=0
        window: From any of the windowing functions, NOT NECESSARILY 2D
        outmatter: Number of output neurons which matter (E.g. only 10 instead of 16 for MNIST and CIFAR hardware). If None, defaults to total number of output neurons
    Given window, construct a window->output adjacency matrix of shape (num_outputs,num_windows)
        Element (i,j) is how many connections go to output neuron i from window j
        This is compared with the ideal number of connections from a window of input neurons to an output neuron
            Ideal number = Number of input neurons considered / Number of total input neurons * Product of fanins
            Eg: Consider a window of 16 neurons. Total neurons are 1024 and product of fanins is 256
            Then we can expect 1 in every 4 neurons to have a connection to output, so for that window, ideal number = 16*1/4 = 4
    Output tuple:
        [0] Scatter for each output neuron
        [1] Average scatter for all output neurons which matter
        To get a single number, use output [1]
    '''
    assert type(adjmat01)==type(adjmat12)!=type(adjmat), 'Either adjmat OR both adjmat01,adjmat12 must be 0'
    if type(adjmat)==int and adjmat==0:
        adjmat = np.dot(adjmat12,adjmat01)
    if outmatter==None:
        outmatter = adjmat.shape[0]
    fiprod = np.sum(adjmat[0])

    woam = np.asarray([[np.sum(adjmat[i,window[j]]) for j in range(window.shape[0])] for i in range(adjmat.shape[0])], dtype=float) #actual window-output adjacency matrix
    ideal_sin_conn = fiprod/float(adjmat.shape[1]) #ideal number of connections to an output neuron from a single input neuron
    ideal_win_conn = window.shape[1]*ideal_sin_conn #ideal number of connections to an output neuron from a window of input neurons
    if ideal_win_conn%1 != 0:
        warnings.warn('Ideal number of connections from a window = {0} is not an integer'.format(ideal_win_conn))
    ideal_woam = ideal_win_conn*np.ones_like(woam) #ideal window-output adjacency matrix for this level of sparsity

    div_woam = woam/ideal_woam
    div_woam[div_woam>1] = 1
    avg_div_woam = np.average(div_woam, axis=1) #Individual value for each output neuron
    wholeavg_div_woam = np.average(avg_div_woam[:outmatter]) #1 value for all the output neurons which matter
    return (avg_div_woam, wholeavg_div_woam)

    ## Bad code I wrote to calculate scatter for fully-connected case ##
#==============================================================================
#         idealFC_woam = window.shape[1]*np.ones_like(woam) #ideal window-output adjacency matrix had it been fully connected
#         divFC_woam = woam/idealFC_woam
#         # Entries can never be > 1
#         avg_divFC_woam = np.average(divFC_woam, axis=1) #Individual value for each output neuron
#         wholeavg_divFC_woam = np.average(avg_divFC_woam[:outmatter]) #1 value for all the output neurons which matter
#==============================================================================


def scatter_net(windows=[window1D(64,8,8),window1D(1024,8,8), window1D(1024,8,8),window1D(64,8,8), window1D(64,1,1),window1D(64,1,1)],
              inmatter=None, outmatter=None, **kwargs):
    '''
    Calculate all scatter values for a network, i.e. for any layer of neurons w.r.t windows from any other layer
    Inputs:
        windows:
            A list of length = total number of adjmats:
                0th element is which input (p side) window to consider for junction 01
                1st element is which output (n side) window to consider for junction 01
                2nd element is which input (p side) window to consider for junction 12 ... and so on
                2nd last and last elements are which input (p side) and output (n side) windows to consider for entire network, i.e. input to output
            The default values are for Morse code, use that to understand better
        inmatter: Consider only 1st inmatter input neurons, such as 784 for MNIST. Leave as None to consider all
        outmatter: Consider only 1st outmatter output neurons. such as 10 for MNIST. Leave as None to consider all
        kwargs:
            EITHER adjmats: Loaded npz archive containing adjacency matrices stored as 'adjmat01', 'adjmat12', etc
            OR individual adjmats 'adjmat01', 'adjmat12', etc
    Outputs:
        Calculates scatter for each set of neurons w.r.t the other in a junction
        Either returns all values, or just the minimum
    Usage:
        print(scatter_net(adjmat01 = adjmat_basic(p=64,fo=128,n=1024), adjmat12 = adjmat_basic(p=1024,fo=8,n=64)) OR
        print(scatter_net(adjmats = np.load('./adjmats/20171014_morse_fo1288_bestzonality_window1.npz'))
    '''
    a = []
    if 'adjmats' in kwargs:
        adjmats = kwargs['adjmats']
        for i in range(len(adjmats.keys())):
            a.append(adjmats['adjmat{0}{1}'.format(i,i+1)])
    else:
        if 'adjmat01' in kwargs:
            a.append(kwargs['adjmat01'])
        if 'adjmat12' in kwargs:
            a.append(kwargs['adjmat12'])
        if 'adjmat23' in kwargs:
            a.append(kwargs['adjmat23'])
        #add more as necessary
    x = np.dot(a[1],a[0])
    for i in range(2,len(a)):
        x = np.dot(a[i],x)
    a.append(x) #appending final IO adjmat
    assert len(windows) == 2*len(a), 'number of windows must be twice the number of adjacency matrices'

    if inmatter==None:
        inmatter = a[0].shape[1] #total number of input neurons
    if outmatter==None:
        outmatter = a[-1].shape[0] #total number of output neurons

    fo = []
    fi = []
    for adj in a[:-1]: #exclude overall adjmat when computing individual fo and fi
        fo.append(np.count_nonzero(adj[:,0]))
        fi.append(np.count_nonzero(adj[0]))
    fo.append(np.prod(np.asarray(fo))) #overall adjmat fo
    fi.append(np.prod(np.asarray(fi))) #overall adjmat fi

    '''
    ideal_sn_conn:
        for i=0 it is ideal number of jn01 p neurons connecting to a single jn01 n neuron
        for i=1 it is ideal number of jn01 n neurons connecting to a single jn01 p neuron
        for i=2 it is ideal number of jn12 p neurons connecting to a single jn12 n neuron
    ideal_win_conn is ideal_sn_conn multiplied by number of elements in a single window of that respective windows element
    woam is actual window output adjacency matrix
    ideal_woam is ideal window output adjacency matrix to bets distribute connections
    s is scatter for each window
    '''
    s = np.zeros(len(windows))
    for i in range(len(windows)):
        if i%2==0: #evaluating n side w.r.t p side windows
            woam = np.asarray([[np.sum(a[i//2][k,windows[i][j]]) for j in range(windows[i].shape[0])] for k in range(a[i//2].shape[0])], dtype=float)
            ideal_sn_conn = fo[i//2] / float(a[i//2].shape[0])
        else: #evaluating p side w.r.t n side windows
            woam = np.asarray([[np.sum(a[i//2][windows[i][j],k]) for j in range(windows[i].shape[0])] for k in range(a[i//2].shape[1])], dtype=float)
            ideal_sn_conn = fi[i//2] / float(a[i//2].shape[1])
        ideal_win_conn = ideal_sn_conn * windows[i].shape[1]
        if ideal_win_conn%1 != 0:
            warnings.warn('Ideal number of connections from window[{0}] = {1} is not an integer'.format(i,ideal_win_conn))
        ideal_woam = ideal_win_conn * np.ones_like(woam)
        div_woam = woam/ideal_woam
        div_woam[div_woam>1] = 1
        avg_div_woam = np.average(div_woam, axis=1) #Individual value for each neuron

        if i==1 or i==len(windows)-1: #evaluating for input layer w.r.t 1st hidden layer OR w.r.t final output layer
            s[i] = np.average(avg_div_woam[:inmatter])
        elif i==len(windows)-4 or i==len(windows)-2: #evaluating for output layer w.r.t last hidden layer OR w.r.t initial input layer
            s[i] = np.average(avg_div_woam[:outmatter])
        else:
            s[i] = np.average(avg_div_woam)
    return s
#    return np.min(s)
#==============================================================================
#==============================================================================
