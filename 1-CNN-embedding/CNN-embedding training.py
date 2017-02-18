from keras.layers import Dense, Input, Flatten, Dropout, Merge, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential, load_model, Model
from keras.callbacks import Callback, EarlyStopping, TensorBoard
import keras.backend as K


from settings import * 
np.random.seed(1)

# ### load data

pk_data = pk.load(open(PK_FPATH, 'rb'))

embedding_w2v = pk_data['embedding_w2v']
embedding_glove = pk_data['embedding_glove']

X_train, Y_train = pk_data['X_train'], pk_data['Y_train']
X_val, Y_val = pk_data['X_val'], pk_data['Y_val']

INPUT_SEQ_LEN = X_train.shape[1]
EMBEDDING_INPUT_DIM = embedding_w2v.shape[0]


# ### Modify sample weight, and use larger batch size
inv_freq = 1e6*Y_train.sum(axis=0)**(-1.5)
sample_weight = (inv_freq * Y_train).sum(axis=1)

# ### helper functions 
import utils
reload(utils)
from utils import multlabel_prec, multlabel_recall, multlabel_F1, multlabel_acc 

def evaluate_model(model):
    print 'evaluation on training set:'
    print model.evaluate(X_train, Y_train, batch_size=128)
    print 'evaluation on validation set:'
    print model.evaluate(X_val, Y_val, batch_size=128)


def compile_fit_evaluate(model, quick_test=False, print_summary=True,
                         save_log=True, save_model=True, del_model=False):
    # this funciton wraps up operations on models
    model.compile(loss='binary_crossentropy', # loss for multilabel classification
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
              sample_weight = sample_weight, 
              callbacks=_callbacks )
    
    print 'evaluating model...'
    evaluate_model(model)
    
    if save_model: 
        model_fpath = os.path.join( MODEL_PATH, DATE_TODAY+'_%s.h5'% str(model.name) )
        model.save(model_fpath)
        print 'model is saved at %s' % model_fpath
    
    if del_model:
        del model # delete the model to save memory


# ## start testing different models 
flag_quick_test = 1

# ### model 1: 2 embedding layers, 2 conv1d layers, 2 FC layers
embed1_w2v = Embedding(input_dim=EMBEDDING_INPUT_DIM ,output_dim=EMBEDDING_DIM, 
              weights=[embedding_w2v],input_length=INPUT_SEQ_LEN, trainable=False )
embed2_glove = Embedding(input_dim=EMBEDDING_INPUT_DIM ,output_dim=EMBEDDING_DIM, 
              weights=[embedding_glove],input_length=INPUT_SEQ_LEN, trainable=False )

input_layer = Input(shape=(INPUT_SEQ_LEN,), dtype='int32', name='main_input')

embed1 = embed1_w2v(input_layer)
conv_embed1 = Conv1D(128, 5, activation='relu')(embed1)

embed2 = embed2_glove(input_layer)
conv_embed2 = Conv1D(128, 5, activation='relu')(embed2)

from keras.layers import merge # `Merge` is for model, while `merge` is for tensor.
merge_layer = merge([conv_embed1, conv_embed2], mode='sum')

x = MaxPooling1D(5)(merge_layer)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
x = Dropout(p=0.5)(x)
x = Dense(500)(x)
x = Activation('relu')(x)
x = Dropout(p=0.5)(x)
output_layer = Dense(N_LABELS, activation='sigmoid')(x)

model_2embed_2conv1d_2FC = Model(input=input_layer, output=output_layer, 
                                 name = 'model_2embed_2conv1d_2FC')

compile_fit_evaluate(model_2embed_2conv1d_2FC, flag_quick_test)

# ### model 2.1: use w2v embedding, 2 conv1d layers, 2 FC layers
model_2conv1d_2FC = Sequential(
       [Embedding(input_dim=EMBEDDING_INPUT_DIM ,output_dim=EMBEDDING_DIM, 
              weights=[embedding_w2v],input_length=INPUT_SEQ_LEN, trainable=False ),
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
       ], name = 'model_2conv1d_2FC')
compile_fit_evaluate(model_2conv1d_2FC, flag_quick_test)

# ### model 2.2: use glove instead of w2v embedding
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
       ], name = 'model_2conv1d_2FC_glove')
compile_fit_evaluate(model_2conv1d_2FC_glove, flag_quick_test)\

# ### model 3: 3 conv layers 
model_3conv1d =Sequential(
        [ Embedding(input_dim=EMBEDDING_INPUT_DIM ,output_dim=EMBEDDING_DIM, 
                  weights=[embedding_w2v],input_length=INPUT_SEQ_LEN, trainable=False ),
            Conv1D(256, 5, activation='relu'),
            MaxPooling1D(5),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(5),
            Conv1D(64, 2, activation='relu'),
            MaxPooling1D(5),
            Flatten(),
            Dropout(p=0.5),
            Dense(N_LABELS, activation='sigmoid') ],
        name = 'model_3conv1d')

compile_fit_evaluate(model_3conv1d, flag_quick_test)