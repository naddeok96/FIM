sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'   
                },
                'parameters': {
                    'epochs': {
                        'values': [1e2, 5e2, 1e3, 5e3]
                    },
                    'batch_size': {
                        'values': [64, 124, 256]
                    },
                    'momentum': {
                        'values': [0, 0.7, 0.8, 0.9]
                    },
                    'weight_decay': {
                        'values': [0, 1e-2, 5e-2, 1e-3, 5e-3]
                    },
                    'learning_rate': {
                        'values': [1e-1, 1e-2, 1e-3]
                    },
                    'optimizer': {
                        'values': ['sgd', 'adadelta']
                    },
                    'criterion': {
                        'values': ["mse", 'cross_entropy']  
                    },
                    'num_kernels_layer1':{
                        'values' : [2**3, 2**4, 2**5, 2**6]
                    },
                    'num_kernels_layer2':{
                        'values' : [2**4, 2**5, 2**6, 2**7]
                    },
                    'num_kernels_layer3':{
                        'values' : [2**7, 2**8, 2**9, 2**10]
                    },
                    'num_nodes_fc_layer':{
                        'values' : [2**5, 2**6, 2**7, 2**8]
                    }
                }
            }
            