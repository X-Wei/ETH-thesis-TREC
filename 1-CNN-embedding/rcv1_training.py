from keras.layers import Dense, Input, Flatten, Dropout, Merge, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential, load_model, Model
from keras.callbacks import Callback, EarlyStopping, TensorBoard
import keras.backend as K


from settings import * 
np.random.seed(1)

prefix = 'rcv1'
pk_data = pk.load(open(RCV1_PK_FPATH, 'rb'))

embedding_glove     = pk_data['embedding_glove']
X_train, Y_train    = pk_data['X_train'], pk_data['Y_train']
X_val, Y_val        = pk_data['X_val'], pk_data['Y_val']

N_LABELS            = Y_train.shape[1]
INPUT_SEQ_LEN       = X_train.shape[1]
EMBEDDING_INPUT_DIM = embedding_glove.shape[0]

from utils import multlabel_prec, multlabel_recall, multlabel_F1, multlabel_acc 

def compile_fit_evaluate(model, quick_test=False, print_summary=True,
                         save_log=True, save_model=True, del_model=False):
    # this funciton wraps up operations on models
    model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=[multlabel_prec, multlabel_recall, multlabel_F1, multlabel_acc])
    if print_summary:
        print model.summary()
        
    if quick_test: # use tiny data for quick testing
        print '(quick test mode)'
        model.fit(X_train[:100], Y_train[:100], nb_epoch=1)
        return  
    _callbacks = [EarlyStopping(monitor='val_loss', patience=2)] 
    if save_log:
        logdir = os.path.join( LOG_PATH, DATE_TODAY+'_'+str(model.name) )
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        _callbacks.append(TensorBoard(log_dir=logdir))
        print 'run "tensorboard --logdir=%s" to launch tensorboard'%logdir
    model.fit( X_train, Y_train, 
              validation_data=(X_val, Y_val), 
              nb_epoch=N_EPOCHS, batch_size=BATCH_SZ, 
            #   sample_weight = sample_weight, 
              callbacks=_callbacks )
    if save_model: 
        model_fpath = os.path.join( MODEL_PATH, DATE_TODAY+'_%s.h5'% str(model.name) )
        model.save(model_fpath)
        print 'model is saved at %s' % model_fpath


model_2conv1d_2FC_glove = Sequential(
       [Embedding(input_dim=EMBEDDING_INPUT_DIM ,output_dim=EMBEDDING_DIM, 
              weights=[embedding_glove],input_length=INPUT_SEQ_LEN, trainable=False ),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Flatten(),
        Dropout(p=0.5),
        Dense(500),
        Activation('relu'),
        Dropout(p=0.5),
        Dense(N_LABELS, activation='sigmoid') 
       ], name = '%s_2conv1d_2FC_glove' % prefix)

compile_fit_evaluate(model_2conv1d_2FC_glove, 0)


model_2conv1d_2FC_glove_d200 = Sequential(
       [Embedding(input_dim=EMBEDDING_INPUT_DIM ,output_dim=EMBEDDING_DIM, 
              weights=[embedding_glove],input_length=INPUT_SEQ_LEN, trainable=False ),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Flatten(),
        Dropout(p=0.5),
        Dense(200),
        Activation('relu'),
        Dropout(p=0.5),
        Dense(N_LABELS, activation='sigmoid') 
       ], name = '%s_2conv1d_2FC_glove_d200' % prefix)

compile_fit_evaluate(model_2conv1d_2FC_glove, 0)