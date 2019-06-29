#==============================================================================
# Train pre-defined sparse nets using Keras
# Sourya Dey, USC
#==============================================================================


#==============================================================================
#==============================================================================
# # Imports
#==============================================================================
#==============================================================================
import numpy as np
import pickle
from scipy.stats import entropy

from data_loadstore import load_any_data
import data_processing as dp
import adjmatint
import keras_nets as nets
from utils import merge_dicts

from keras.regularizers import l2
from keras.initializers import Constant
import keras.callbacks
from keras.models import load_model
from keras.optimizers import Adam

data_folder = './'
preds_folder = './timit_FCs/'
#==============================================================================
#==============================================================================



#==============================================================================
#==============================================================================
# # Callback to sparsify weight matrices
#==============================================================================
#==============================================================================
class SparseWeights(keras.callbacks.Callback):
    def __init__(self,adjmats,numlayers):
        '''
        adjmats still holds the non-transposed matrices, so need to transpose them
        numlayers: EXCLUDING input. For example for [784,100,100,10], this is 3
        '''
        self.adjmats = []
        self.numlayers = numlayers
        for i in range(numlayers):
            self.adjmats.append(adjmats['adjmat{0}{1}'.format(i,i+1)].T)
    def on_batch_end(self,batch,logs={}):
        wb = self.model.get_weights()
        for i in range(self.numlayers):
            wb[-2*(self.numlayers-i)] *= self.adjmats[i]
        self.model.set_weights(wb)
#==============================================================================
#==============================================================================



#==============================================================================
#==============================================================================
# # Main function
#==============================================================================
#==============================================================================
def run_model(model, config, fo, z=None,
              loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'],
              valtest=0, batch_size = 128, total_epochs = 30, epoch_step = -1,
              preds_compare = 0,
              dataset_filename = data_folder + 'dataset_cifar/cifar10_keras.npz',
              input_pad = -1, output_pad = -1,
              preprocesses = (),
              results_filepath = './results_new/new'):
    '''
    model: Output from any net function
    loss, optimizer, metrics: Needed for model compilation
    config, fo: Config and fanout. These are required ONLY FOR MLP portion of the net
    z: Degree of parallelism. Only required for clash-free adjmats. Should be nparray same size as fo
    preds_compare:
        0: Do nothing
        1: Compute and save test predictions as 'test_preds' to npz file, i.e. ndarray of same shape as yte. Use this for FC to set a standard
            Average standards across different iters to create 1 standard and save it as a separate npz file with key 'test_preds'
        Some npz file which has standard 'test_preds' as a key:
            Compute and save test predictions as 'test_preds' to npz file
            Compare these to standard predictions (maybe from FC):
                1/xte.shape[0] * Sum across rows ( Sum across columns ( standard_preds_ij * ln( standard_preds_ij / computed_preds_ij ) ) )
            Use this for sparse to get a soft indicator of how much worse it is from FC. LOWER the BETTER (always between 0 and 1)
            Save this as 'test_preds_compare' to output dictionary
        ****** NOTE ****** : *** Always store test_preds as npz and test_preds_compare in output dict with everything else ***
    valtest:
        0: Model selection - train using tr, validate using va. Don't test.
            This saves train and val metrics, including epoch where validation metric was maximum
            Filename has _VAL appended
        1: Final model evaluation. Train using tr+va and test once at the end using te. Don't validate.
            This saves train (on extended set) and test metrics
            Filename has nothing appended
        2: Final model evaluation. Train using tr only and test once at the end using te. Don't even use validation data.
            This saves train (on normal, non-extended training set) and test metrics
            Filename has nothing appended
    epoch_step: Save model and accs after this many epochs.
        Default: -1, then epoch_step=total_epochs
        Otherwise must be <= total_epochs
    dataset_filename: Dataset path
    input_pad, output_pad: If not -1, xdata is padded with 0s to reach input_pad, likewise for ydata
        Example: MNIST 784,10 can have input_pad=1024, output_pad=16
        Do not use input_pad when the 1st layer is a CNN
    preprocesses: Any input preprocessing function from data_processing. Enter as tuple, example (dp.gaussian_normalize,)
    results_filepath: Path to store model and accuracies
    '''
    xtr,ytr,xva,yva,xte,yte = load_any_data(filename = dataset_filename)
    for preprocess in preprocesses:
        xtr,xva,xte = preprocess(xtr,xva,xte)

    if epoch_step==-1:
        epoch_step = total_epochs
        
    callbacks = []


#==============================================================================
#     Use for MNIST CNN, not for cl only
#==============================================================================
#==============================================================================
#     xtr = xtr.reshape(xtr.shape[0],28,28,1)
#     xva = xva.reshape(xva.shape[0],28,28,1)
#     xte = xte.reshape(xte.shape[0],28,28,1)
#==============================================================================
#==============================================================================


#==============================================================================
#     Pad xdata and ydata with 0s as applicable - Do not change
#==============================================================================
    if input_pad>xtr.shape[1]:
        print('Padding xdata from {0} to {1}'.format(xtr.shape[1],input_pad))
        xtr = np.concatenate((xtr, np.zeros((xtr.shape[0],input_pad-xtr.shape[1])) ), axis=1)
        xva = np.concatenate((xva, np.zeros((xva.shape[0],input_pad-xva.shape[1])) ), axis=1)
        xte = np.concatenate((xte, np.zeros((xte.shape[0],input_pad-xte.shape[1])) ), axis=1)
    if output_pad>ytr.shape[1]:
        print('Padding ydata from {0} to {1}'.format(ytr.shape[1],output_pad))
        ytr = np.concatenate((ytr, np.zeros((ytr.shape[0],input_pad-ytr.shape[1])) ), axis=1)
        yva = np.concatenate((yva, np.zeros((yva.shape[0],input_pad-yva.shape[1])) ), axis=1)
        yte = np.concatenate((yte, np.zeros((yte.shape[0],input_pad-yte.shape[1])) ), axis=1)
#==============================================================================


#==============================================================================
#     Configure data according to valtest - Do not change
#==============================================================================
    '''
    validation_data needs to be put into a tuple always. Make this None if not validating
    Other data stays as individual x and y. If not testing, set these individually to None
    '''
    if valtest==0:
        #train_data as is
        validation_data = (xva,yva)
        #test_data unused
        results_filepath += '_VAL' #this is to differentiate the validation records from the final test records
    elif valtest==1:
        xtr = np.concatenate((xtr,xva),axis=0)
        ytr = np.concatenate((ytr,yva),axis=0)
        validation_data = None
        #test_data as is
    elif valtest==2:
        #train data as is
        validation_data = None
        #test_data as is
#==============================================================================


#==============================================================================
#     Adjmats
#==============================================================================
    '''
    Adjmats are stored as a dictionary 'adjmats'. Alternatively they can be loaded from pre-saved npz file 'adjmats' which is also like a dict
    I size adjmats as (output_dim,input_dim), but keras works as (input_dim,output_dim). Hence need to transpose
    '''
    numlayers = len(config)-1
#==============================================================================
#     Load pre-saved adjmats
#==============================================================================
#==============================================================================
#     adjmats =
#==============================================================================

#==============================================================================
#     Otherwise create from scratch
#==============================================================================
    adjmats = {}
    for i in range(numlayers):
        
        #For basic or random:
        adjmats['adjmat{0}{1}'.format(i,i+1)] = adjmatint.adjmat_basic(config[i],fo[i],config[i+1])
        
        #For clash-free:
#        adjmats['adjmat{0}{1}'.format(i,i+1)] = adjmatint.adjmat_basic(config[i],fo[i],config[i+1],z[i]) #Can also set typ, memdith, etc
        
        #Adjmats are in my transposed format
        #They need to be transposed when used to set keras weights
    spwt = SparseWeights(adjmats,numlayers)
#==============================================================================


#==============================================================================
#     Set initial weights to be sparse - Do not change
#==============================================================================
    wb = model.get_weights()
    for i in range(numlayers):
        wb[-2*(numlayers-i)] *= adjmats['adjmat{0}{1}'.format(i,i+1)].T
    model.set_weights(wb)
#==============================================================================
    

#==============================================================================
#     Model compile and run
#==============================================================================
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.save(results_filepath+'.h5')
    recs = {} #keeps all records
    with open(results_filepath+'.pkl','wb') as f:
        pickle.dump(recs,f)

    for k in range(total_epochs//epoch_step):
        model = load_model(results_filepath+'.h5')
        with open(results_filepath+'.pkl','rb') as f:
            recs = pickle.load(f)

        if np.array_equal(config[1:],fo) == False: #Any sparse case: Sparsity is enforced ONLY IF callbacks=[spwt] is included in arguments to model.fit()
            callbacks.append(spwt)
        history = model.fit(xtr,ytr, batch_size=batch_size, epochs=epoch_step, validation_data=validation_data, callbacks=callbacks)

        ### Save model and records for resuming training ###
        model.save(results_filepath+'.h5')
        merge_dicts(recs,history.history)
        if k!=(total_epochs//epoch_step-1): #Don't save for the last iteration because testing follows
            with open(results_filepath+'.pkl','wb') as f:
                pickle.dump(recs,f)

    ### Training done ###
    if valtest==0:
        maxvalacc, maxvalpos = max(recs['val_acc']), recs['val_acc'].index(max(recs['val_acc']))+1
        print('\nBest validation accuracy = {0} obtained in epoch {1}'.format(maxvalacc,maxvalpos))
        merge_dicts(recs, {'val_best_epoch':maxvalpos})
    elif valtest in [1,2]:
        score = model.evaluate(xte,yte,batch_size=batch_size)
        test_dict = {'test_loss':score[0]}
        for i in range(len(metrics)):
            test_dict['test_'+metrics[i]] = score[i+1] #score[1] is the 1st metric, score[2] is the 2nd, and so on
            print('\nTest {0} = {1}'.format(metrics[i],score[i+1]))
        if preds_compare!=0:
            preds = model.predict(xte, batch_size=batch_size)
            np.savez_compressed(results_filepath+'.npz', test_preds=preds)
            if preds_compare!=1:
                test_preds_standard = np.load(preds_compare)['test_preds']
                entropy_final = np.average(np.asarray([entropy(test_preds_standard[i],preds[i]) for i in range(len(preds))]))
                test_dict['test_preds_compare'] = entropy_final
                print('Entropy of predictions compared to standard = {0}'.format(entropy_final))
        merge_dicts(recs,test_dict)
    with open(results_filepath+'.pkl','wb') as f:
        pickle.dump(recs,f)
    return recs, model
#==============================================================================



#==============================================================================
#==============================================================================
# # MAIN FUNCTION AND EXECUTION
#==============================================================================
#==============================================================================
def sim_net(
            config = np.array([800,100,10]), #neuron configuration
            fo = np.array([50,10]), #out-degrees
            l2_val = 8e-5, #L2 regularization coefficient
            z = None, #Set to z configuration when simulating clash-free adjmats, for example np.array([200,5])
            dataset_filename = data_folder + 'dataset_MNIST/mnist.npz', #path to dataset
            preds_compare = 0 #Set to path of standard test_preds if comparison is desired
           ):
    if z is None:
        z = [None for _ in range(len(config))]
    
    model = nets.any_cl_only(config, activation='relu', kernel_regularizer=l2(l2_val), kernel_initializer='he_normal', bias_initializer=Constant(0.1))
    recs, model = run_model(model, config, fo, z,
                            optimizer = Adam(decay=1e-5),
                            metrics=['accuracy'],
                            valtest=1, batch_size=256, total_epochs=50, epoch_step=-1,
                            preds_compare = preds_compare,
                            dataset_filename = dataset_filename,
                            input_pad=800,
#                           output_pad=16,
                            preprocesses=(),
                            results_filepath = './results_new/net{0}_fo{1}_l2{2}'.format(config,fo,l2_val))
    return recs,model


recs,model = sim_net(
                    config = np.array([800,100,10]),
                    fo = np.array([50,10]),
                    l2_val = 8e-5,
                    dataset_filename = data_folder + 'dataset_MNIST/mnist.npz'
                    )
