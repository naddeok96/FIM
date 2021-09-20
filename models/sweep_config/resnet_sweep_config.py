sweep_config = {
                'method': 'grid',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize' 
                },
                'parameters': {
                    'epochs': {
                        'values': [150, 1000]
                    },
                    'batch_size': {
                        'values': [124]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [1e-5]
                    },
                    'learning_rate': {
                        'values': [1e-1]
                    },
                    'optimizer': {
                        'values': ['nesterov']
                    },
                    'scheduler': {
                        'values': ["Cosine Annealing"]
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
                    'transformation':{
                        'values' : ["U"]
                    },
                    'model_name':{
                        'values' : ['cifar10_mobilenetv2_x1_0']
                    }
                }
            }
             