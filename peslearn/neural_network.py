import numpy as np
import sklearn.metrics
from .model import Model
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval

class NeuralNetwork(Model):
    """
    Constructs a Neural Network Model using Keras with Tensorflow or Theano as a backend
    """
    def __init__(self, dataset_path, input_obj, mol_obj=None):
        super().__init__(dataset_path, input_obj, mol_obj)
        self.set_default_hyperparameters()

    def get_hyperparameters(self):
        """
        Returns hyperparameters of this model
        """
        return self.hyperparameter_space

    def set_hyperparameter(self, key, val):
        """
        Set hyperparameter 'key' to value 'val'.
        Parameters
        ---------
        key : str
            A hyperparameter name
        val : obj
            A HyperOpt object such as hp.choice, hp.uniform, etc.
        """
        self.hyperparameter_space[key] = val

    def set_default_hyperparameters(self):
        self.hyperparameter_space = {
                      'scale_X': hp.choice('scale_X',
                               [
                               # Scale input data to domain of chosen activation functions.
                               {'scale_X': 'std', #might need diff anme than scale_X
                                    'activation_0': hp.choice('activation_a0', ['sigmoid', 'tanh', 'linear']),
                                    'activation_1': hp.choice('activation_a1', ['sigmoid', 'tanh', 'linear']),
                                    'activation_2': hp.choice('activation_a2', ['sigmoid', 'tanh', 'linear']),
                                    'activation_3': hp.choice('activation_a3', ['sigmoid', 'tanh', 'linear'])},
                               {'scale_X': 'mm', #might need diff anme than scale_X
                                    'activation_0': hp.choice('activation_b0', ['sigmoid', 'linear']),
                                    'activation_1': hp.choice('activation_b1', ['sigmoid', 'linear']),
                                    'activation_2': hp.choice('activation_b2', ['sigmoid', 'linear']),
                                    'activation_3': hp.choice('activation_b3', ['sigmoid', 'linear'])},
                               {'scale_X': 'mm11', #might need diff anme than scale_X
                                    'activation_0': hp.choice('activation_c0', ['tanh', 'linear']),
                                    'activation_1': hp.choice('activation_c1', ['tanh', 'linear']),
                                    'activation_2': hp.choice('activation_c2', ['tanh', 'linear']),
                                    'activation_3': hp.choice('activation_c3', ['tanh', 'linear'])},
                              ]),
                      'scale_y': hp.choice('scale_y', ['std', 'mm', 'mm11']), 
                      'dense_0': hp.choice('dense_0', [16,32,64,128]),
                      'dense_1': hp.choice('dense_1', [16,32,64,128]),
                      'dense_2': hp.choice('dense_2', [16,32,64,128]),
                      'dense_3': hp.choice('dense_3', [16,32,64,128]),
                      'hlayers': hp.choice('hlayers', [1,2,3,4]),
                      'lr': hp.choice('lr',[0.001,0.005,0.01,0.05,0.1]),
                      'decay': hp.choice('decay',[1e-5,1e-6,0.0]),
                      'batch': hp.choice('batch', [16,32,64,128,256]),
                      'a_optimizer': hp.choice('a_optimizer',['SGD','RMSprop', 'Adagrad', 'Adadelta','Adam','Adamax']),
                     }


        if self.input_obj.keywords['pes_format'] == 'interatomics':
            self.hyperparameter_space['morse_transform'] = hp.choice('morse_transform',[{'morse': True,'morse_alpha': hp.uniform('morse_alpha', 1.0, 2.0)},{'morse': False}])
        else:
            self.hyperparameter_space['morse_transform'] = hp.choice('morse_transform',[{'morse': False}])
        if self.pip:
            val =  hp.choice('pip',[{'pip': True,'degree_reduction': hp.choice('degree_reduction', [True,False])}])
            self.set_hyperparameter('pip', val)
        else:
            self.set_hyperparameter('pip', hp.choice('pip', [False]))

    def build_model(self, params, data_path='PES.dat', fi_path='h2co_fi', ntrain=600):
        morse_transform = params['morse_transform']['morse']
        if morse_transform:
            morse_alpha = params['morse_transform']['morse_alpha']
        scale_X = params['scale_X']['scale_X']
        scale_y = params['scale_y']
    
        # load data 
        data = pd.read_csv(data_path) 
        data = data.values
        raw_X = data[:,0:-1]
        raw_y = data[:,-1].reshape(-1,1)
    
        # FUNDAMENTAL INVARIANT
        raw_X, degrees = interatomics_to_fundinvar(raw_X,fi_path)
        ## DEGREE REDUCTION AFTER FI
        if params['degree_red']:
            for i, degree in enumerate(degrees):
                raw_X[:,i] = np.power(raw_X[:,i], 1/degree)
        # do morse transformation, scaling as called by HyperOpt
        if morse_transform:
            raw_X = morse(raw_X, morse_alpha)
        if scale_X:
            X, Xscaler = super_scaler(scale_X, raw_X)
        else:
            X = raw_X
        if scale_y:
            y, yscaler = super_scaler(scale_y, raw_y)
        else:
            y = raw_y
    
        # use random state that was successful for GPs
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, 
                                                            train_size=ntrain, random_state=0)
    
        activ0 = params['scale_X']['activation_0']
        activ1 = params['scale_X']['activation_1']
        activ2 = params['scale_X']['activation_2']
        activ3 = params['scale_X']['activation_3']
    
    
        in_dim = tuple([X.shape[1]])
        out_dim = y.shape[1]
        model = Sequential()
        model.add(Dense(params['dense_0']))
        #model.add(Activation(params['activation_0']))
        model.add(Activation(activ0))
        if params['hlayers'] > 1:  
            model.add(Dense(params['dense_1']))
        #model.add(Activation(params['activation_1']))
            model.add(Activation(activ1))
            if params['hlayers'] > 2:  
                model.add(Dense(params['dense_2']))
                #model.add(Activation(params['activation_2']))
                model.add(Activation(activ2))
                if params['hlayers'] > 3:  
                    model.add(Dense(params['dense_3']))
                    #model.add(Activation(params['activation_3']))
                    model.add(Activation(activ3))
        model.add(Dense(out_dim))
        model.add(Activation('linear'))
    
        if params['a_optimizer'] == 'SGD':
            opt = optimizers.SGD(lr=params['lr'],
                                  decay=params['decay'])
        if params['a_optimizer'] == 'RMSprop':
            opt = optimizers.RMSprop(lr=params['lr'],
                                  decay=params['decay'])
        if params['a_optimizer'] == 'Adagrad':
            opt = optimizers.Adagrad(lr=params['lr'],
                                  decay=params['decay'])
        if params['a_optimizer'] == 'Adadelta':
            opt = optimizers.Adadelta(lr=params['lr'],
                                  decay=params['decay'])
        if params['a_optimizer'] == 'Adam':
            opt = optimizers.Adam(lr=params['lr'],
                                  decay=params['decay'],
                                  amsgrad=True)
        if params['a_optimizer'] == 'Adamax':
            opt = optimizers.Adamax(lr=params['lr'],
                                  decay=params['decay'])
    
        # if error does not improve by 1 mH in 500 epochs, kill
        #callbacks = [EarlyStopping(monitor='mean_absolute_error', min_delta=5.0e-4, patience=1000)]
        model.compile(loss='mse', optimizer=opt, metrics=['mae'])
        # train with the "test" set, so its only 1000
        model.fit(x=X_train,y=y_train,epochs=70000,batch_size=params['batch'],verbose=1)
    
        # test on full dataset
        pred_full = model.predict(X)
        if scale_y:
            unscaled_pred_full = yscaler.inverse_transform(pred_full)
            #loss = np.sqrt(sklearn.metrics.mean_squared_error(raw_y, unscaled_pred_full))
            loss = np.sqrt(np.mean(np.square(raw_y - unscaled_pred_full)))
        else:
            #loss = np.sqrt(sklearn.metrics.mean_squared_error(y, pred_full))
            loss = np.sqrt(np.mean(np.square(y - pred_full)))
        print("RMSE (eH): ", loss, end=' ')
        print("RMSE (cm-1): ", 219474.63 * loss, end='  ')
        print(str(sorted(params.items())))
        # deleting models to save computational time
        K.clear_session()
        return {'loss': loss, 'status': STATUS_OK, 'model':model}



