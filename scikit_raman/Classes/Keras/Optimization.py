import optuna
from collections import defaultdict
from tensorflow.keras.optimizers import Adam

from scikit_raman.Classes.Keras.SingleGPU.DLModel import DLModelKeras

VARIABLE_TYPE = {'int': 'suggest_int', 'boolean':'suggest_categorical', 'float':'suggest_float'}
MODEL_FUNCTION = {'pool': 'MaxPooling1D', 'cnn': 'Conv1D', 'bacth': 'BatchNormalization', }


def objective(trial, dictionary_arch, dictionary_hyper):
    result_arch = defaultdict(dict)
    result_hyper = defaultdict(dict)
    for el in dictionary_arch.keys():
        func_name = getattr(trial, VARIABLE_TYPE[dictionary_arch[el]['add']['type']])
        values = dictionary_arch[el]['add']['value']
        result_arch[el]['add'] = func_name(el, values)
        result_arch[el]['type'] = dictionary_arch[el]['type']
        for el_due in dictionary_arch[el]['params']:
            func_name = getattr(trial, VARIABLE_TYPE[dictionary_arch[el]['params'][el_due]['type']])
            if dictionary_arch[el]['params'][el_due]['type'] == 'boolean':
                values = dictionary_arch[el]['params'][el_due]['value']
                try:
                    result_arch[el]['params'][el_due] = func_name(el + '' + el_due, values)
                except KeyError:
                    result_arch[el]['params'] = defaultdict(dict)
                    result_arch[el]['params'][el_due] = func_name(el + '' + el_due, values)
            else:
                values = dictionary_arch[el]['params'][el_due]['value']
                try:
                    result_arch[el]['params'][el_due] = func_name(el + '' + el_due, values[0], values[1])
                except KeyError:
                    result_arch[el]['params'] = defaultdict(dict)
                    result_arch[el]['params'][el_due] = func_name(el + '' + el_due, values[0], values[1])
    for el in dictionary_hyper:
        func_name = getattr(trial, VARIABLE_TYPE[dictionary_hyper[el]['type']])
        if dictionary_hyper[el]['type'] == 'boolean':
            values = dictionary_hyper[el]['value']
            try:
                result_hyper[el] = func_name(el, values)
            except KeyError:
                result_hyper[el] = defaultdict(dict)
                result_hyper[el] = func_name(el, values)
        else:
            values = dictionary_hyper[el]['value']
            try:
                result_hyper[el] = func_name(el, values[0], values[1])
            except KeyError:
                result_hyper[el] = defaultdict(dict)
                result_hyper[el] = func_name(el, values[0], values[1])
    loss = 'categorical_crossentropy'
    metrics = ['categorical_accuracy']
    optimizer = Adam(learning_rate=0.00020441990333108206)
    model = DLModelKeras.load_from_json_optuna(result_arch, result_hyper, optimizer, loss, metrics, 991)
    print(model.model.summary())
    l = [10,15]
    return sum(l)


d_arch = {'cnn1':
         {'add':
              {'value':[True, False],
               'type':'boolean'},
          'params':
              {'stride':
                   {'value':[5,100],
                    'type':'int'},
               'kernel':
                   {'value':[5,100],
                    'type': 'int'}
               },
          'type':'cnn'
          },
     'pool1':
         {'add':
              {'value':[True, False],
               'type':'boolean'},
          'params':
              {'kernel':
                   {'value':[5,100],
                    'type':'int'}
               },
          'type':'pool'},
    'flatten':
         {'add':
              {'value':[True, True],
               'type':'boolean'},
            'params':{},
          'type':'flatten'},
    'dense1':
         {'add':
              {'value':[True, False],
               'type':'boolean'},
            'params':{'units':
                          {'value':[50,200],
                           'type':'int'
                           }
          },
          'type':'dense'},
     }


d_hyper = {
    'learning_rate':
        {'value':[0.001, 0.001],
         'type':'float'},
    'epochs':
        {'value':[10, 500],
         'type':'int'}}

model = None
func = lambda trial: objective(trial, d_arch, d_hyper)
study = optuna.create_study(direction='maximize')
study.optimize(func, n_trials=10)