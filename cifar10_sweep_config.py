sweep_config = {
                'method': 'grid',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize' 
                },
                'parameters': {
                    'epochs': {
                        'values': [150]
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
                        'values' : ["models/pretrained/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt"]
                    },
                    'model_name':{
                        'values' : ['cifar10_mobilenetv2_x1_0']
                    }
                }
            }
             