sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'   
                },
                'parameters': {
                    'epochs': {
                        'values': [1e2, 5e2, 1e3]
                    },
                    'batch_size': {
                        'values': [64, 124]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [1e-2, 1e-3, 1e-4, 1e-5]
                    },
                    'learning_rate': {
                        'values': [1e-1, 1e-2, 1e-3]
                    },
                    'optimizer': {
                        'values': ['sgd', 'adadelta']
                    },
                    'criterion': {
                        'values': ['mse', 'cross_entropy']  
                    },
                    'use_SAM':{
                        'values' : [True]
                    },
                    'transformation':{
                        'values' : ["U"]
                    }
                }
            }
             