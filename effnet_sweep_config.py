sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'   
                },
                'parameters': {
                    'epochs': {
                        'values': [5e5]
                    },
                    'batch_size': {
                        'values': [124]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [1e-3]
                    },
                    'learning_rate': {
                        'values': [1e-2]
                    },
                    'optimizer': {
                        'values': ['adadelta']
                    },
                    'criterion': {
                        'values': ['cross_entropy']  
                    },
                    'use_SAM':{
                        'values' : [True]
                    },
                    'pretrained':{
                        'values' : [False]
                    },
                    'model_name':{
                        'values' : ['efficientnet-b1']
                    }
                }
            }
             