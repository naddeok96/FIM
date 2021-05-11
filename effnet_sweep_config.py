sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'   
                },
                'parameters': {
                    'epochs': {
                        'values': [5e2, 1e3]
                    },
                    'batch_size': {
                        'values': [64]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [1e-2, 1e-3, 1e-5]
                    },
                    'learning_rate': {
                        'values': [1e-2, 1e-3]
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
                    'pretrained':{
                        'values' : [False]
                    },
                    'transfomation':{
                        'values' : ["U"]
                    },
                    'model_name':{
                        'values' : ['efficientnet-b1']
                    }
                }
            }
             