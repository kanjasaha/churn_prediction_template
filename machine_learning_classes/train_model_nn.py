from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
from keras.layers import LeakyReLU, Activation
from keras.callbacks import TensorBoard
import sys

from keras import callbacks
from keras import backend as K

#pytorch
#keras
#tensorflow





class train_model(object):
    def __init__(self, learner,samples,space,time_series, X_train=None, y_train=None, X_test=None, y_test=None):


        self.samples=samples
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.learner=learner
        self.space=space
        self.time_series=time_series


    #binary classification
    #multiclass classification
    #multi label classification
    @staticmethod
    def getBestModelfromTrials(trials):
        valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
        losses = [float(trial['result']['loss']) for trial in valid_trial_list]
        index_having_minumum_loss = np.argmin(losses)
        best_trial_obj = valid_trial_list[index_having_minumum_loss]
        return best_trial_obj['result']['Trained_Model']



    class MyLogger(callbacks.Callback):
        def __init__(self, n):
            self.n = n  # print loss & acc every n epochs

        def on_epoch_end(self, epoch, logs={}):
            if epoch % self.n == 0:
                curr_loss = logs.get('loss')
                curr_acc = logs.get('acc') * 100
                print("epoch = %4d  loss = %0.6f  acc = %0.2f%%" % (epoch, curr_loss, curr_acc))

    def swish(x):
        return (K.sigmoid(x) * x)

    my_logger = MyLogger(n=1)
    # my_logger = TensorBoard(log_dir="logs/{}".format(time()))
    # my_logger = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    # tensorboard --logdir output/Graph
    class_weight = {0: 1., 1: 9.}
    space = {'choice': hp.choice('num_layers',
                                 [{'layers': 'two', },
                                  {'layers': 'three',
                                   'units3': hp.uniform('units3', 64, 1024),
                                   'dropout3': hp.uniform('dropout3', .25, .75)}
                                  ]),

             'units1': hp.uniform('units1', 64, 1024),
             'units2': hp.uniform('units2', 64, 1024),

             'dropout1': hp.uniform('dropout1', .25, .75),
             'dropout2': hp.uniform('dropout2', .25, .75),

             'batch_size': hp.uniform('batch_size', 28, 128),

             'nb_epochs': 50,
             'optimizer': hp.choice('optimizer', ['adam']),
             'activation': hp.choice('activation', ['LeakyReLU'])
             }

    def f_nn(params):
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers import Adadelta, Adam, rmsprop

        print('Params testing: ', params)
        # print("shape testing: {}, {}".format(params['choice']['units3']))
        model = Sequential()

        model.add(Dense(units=round(params['units1']), input_dim=X_train.shape[1], kernel_initializer="glorot_uniform"))
        if params['activation'] == 'LeakyReLU':
            model.add(LeakyReLU())
        else:
            model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout1']))

        model.add(Dense(units=round(params['units2']), kernel_initializer="glorot_uniform"))
        if params['activation'] == 'LeakyReLU':
            model.add(LeakyReLU())
        else:
            model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout2']))

        if params['choice']['layers'] == 'three':
            model.add(Dense(units=round(params['choice']['units3']), kernel_initializer="glorot_uniform"))
            if params['activation'] == 'LeakyReLU':
                model.add(LeakyReLU())
            else:
                model.add(Activation(params['activation']))
            model.add(Dropout(params['choice']['dropout3']))

        model.add(Dense(units=1, kernel_initializer='normal'))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=round(params['nb_epochs']), batch_size=round(params['batch_size']),
                  verbose=0, callbacks=[my_logger], class_weight=class_weight)

        pred_auc = model.predict_proba(X_validate, batch_size=128, verbose=0)
        acc = roc_auc_score(y_validate, pred_auc)
        print('AUC:', acc)
        sys.stdout.flush()
        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f_nn, space, algo=tpe.suggest, max_evals=10, trials=trials)
    print('best: ')
    print(best)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import fbeta_score, accuracy_score, f1_score

    y_pred = model.predict_classes(X_validate, batch_size=32, verbose=1)
    labels = [0, 1]
    cm = confusion_matrix(y_validate, y_pred, labels)
    display(cm)
    class_error_fp = cm[0][1] / (cm[0][0] + cm[0][1])
    class_error_fn = cm[1][0] / (cm[1][0] + cm[1][1])

    Test_set_accuracy_rate = accuracy_score(y_validate, y_pred)

    print("Test set accuracy:{} Class_error_fp: {} Class_error_fn: {}".format(Test_set_accuracy_rate, class_error_fp,
                                                                              class_error_fn))

    def train_hyperopt(self,save_model=True):


        def objective(args):

            if args['learner'] == 'naivebayes': #1
                params = {'n_estimators': int(args['n_estimators']),
                          'max_depth': int(args['max_depth']),
                          'max_features': (args['max_features'])}
                clf = xgb.train(**params)

            elif args['learner'] == 'xgboost': #2
                params = {'n_estimators': int(args['n_estimators']),
                          'max_depth': int(args['max_depth']),
                          'max_features': (args['max_features'])}
                clf = xgb.train(**params)

            elif args['learner'] == 'lgbm': #3
                params = {'n_estimators': int(args['n_estimators']),
                          'max_depth': int(args['max_depth']),
                          'max_features': (args['max_features'])}
                clf = xgb.train(**params)

            elif args['learner'] == 'randomforest':#4

                params = {'n_estimators': int(args['n_estimators']),
                          'max_depth': int(args['max_depth']),
                          'max_features': (args['max_features'])}

                clf = RandomForestClassifier(**params)
            elif args['learner'] == 'knn': #5
                params = {'n_neighbors': int(args['param']['n_neighbors']),
                          'algorithm': int(args['param']['algorithm']),
                          'leaf_size': int(args['param']['leaf_size']),
                          'metric': int(args['param']['metric']),
                          }
                clf = KNeighborsClassifier(**params)

            elif args['learner'] == 'sgd': #6
                params = {'n_neighbors': int(args['param']['n_neighbors']),
                          'algorithm': int(args['param']['algorithm']),
                          'leaf_size': int(args['param']['leaf_size']),
                          'metric': int(args['param']['metric']),
                          }
                clf = KNeighborsClassifier(**params)
            elif args['learner'] == 'logistic': #7
                params = {'n_neighbors': int(args['param']['n_neighbors']),
                          'algorithm': int(args['param']['algorithm']),
                          'leaf_size': int(args['param']['leaf_size']),
                          'metric': int(args['param']['metric']),
                          }
                clf = KNeighborsClassifier(**params)
            elif args['learner'] == 'lda': #8
                params = {'n_neighbors': int(args['param']['n_neighbors']),
                          'algorithm': int(args['param']['algorithm']),
                          'leaf_size': int(args['param']['leaf_size']),
                          'metric': int(args['param']['metric']),
                          }
                clf = KNeighborsClassifier(**params)
            elif args['learner']=='svm':
                params = {'C': int(args['param']['C']),
                          'kernel': int(args['param']['kernel']),
                          'degree': int(args['param']['degree']),
                          'gamma': int(args['param']['gamma']),
                          }
                clf = svm.SVC(**params)

            #clf = RandomForestClassifier(n_jobs=4, class_weight='balanced', **params)
            #score = cross_val_score(clf, self.X_train, self.y_train).mean() 
            tscv = TimeSeriesSplit()
            if self.time_series:
                cv=tscv
            else:
                cv=None
            f1=cross_val_score(clf,self.X_train, self.y_train.values.ravel(), cv=tscv, scoring='f1_weighted').mean()
            #print("Gini {:.3f} params {}".format(score, params))
            #print('run fit')
            #clf.predict(test_features_2019)
            #print('run fit successful')

            '''clf.fit(self.X_train,self.y_train)
            pred_auc = clf.predict_proba(self.X_test)
            acc = roc_auc_score(self.y_test, pred_auc)
            print('AUC:', acc)
            sys.stdout.flush()'''
            print ("loss {}:, status: {}, Trained Model: {}".format(-f1,STATUS_OK,clf))
            return {'loss': -f1, 'status': STATUS_OK, 'Trained_Model': clf}
            #return score

        trials = Trials()
        best = fmin(fn=objective,space=self.space,algo=tpe.suggest,max_evals=10, trials=trials)
        #pickle.dump(trials, open("trials.p", "wb"))
        #trials = pickle.load(open("trials.p", "rb"))
        best_model = self.getBestModelfromTrials(trials)
        best_model.fit(self.X_train,self.y_train.values.ravel())
        pickle.dump(best_model, open("model_artifacts\saved_model.p", "wb"))
        print("Hyperopt best estimated optimum parameters{}".format(best))
        print("Hyperopt best estimated optimum model{}".format(best_model))
        # The trials database now contains 100 entries, it can be saved/reloaded with pickle or another method

        return best,best_model


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop

print ('Params testing: ', params)
#print("shape testing: {}, {}".format(params['choice']['units3']))
model = Sequential()

model.add(Dense(units=round(params['units1']), input_dim = X_train.shape[1], kernel_initializer = "glorot_uniform"))
if params['activation']== 'LeakyReLU':
    model.add(LeakyReLU())
else:
    model.add(Activation(params['activation']))
model.add(Dropout(params['dropout1']))

model.add(Dense(units=round(params['units2']), kernel_initializer = "glorot_uniform"))
if params['activation']== 'LeakyReLU':
    model.add(LeakyReLU())
else:
    model.add(Activation(params['activation']))
model.add(Dropout(params['dropout2']))

if params['choice']['layers']== 'three':
    model.add(Dense(units=round(params['choice']['units3']), kernel_initializer = "glorot_uniform"))
    if params['activation']== 'LeakyReLU':
        model.add(LeakyReLU())
    else:
        model.add(Activation(params['activation']))
    model.add(Dropout(params['choice']['dropout3']))

model.add(Dense(units=1, kernel_initializer='normal'))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=params['optimizer'],metrics=['accuracy'])

model.fit(X_train, y_train, epochs=round(params['nb_epochs']), batch_size=round(params['batch_size']),verbose = 0,callbacks=[my_logger], class_weight=class_weight)

pred_auc =model.predict_proba(X_validate, batch_size = 128, verbose = 0)
acc = roc_auc_score(y_validate, pred_auc)
