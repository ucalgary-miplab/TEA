EXPS_REGISTRY = {}


class Exps:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


moin_bias = Exps()
moin_bias.exp_name = 'moin_bias'
moin_bias.dim_red = 'PCA'
moin_bias.n_comps = 500
moin_bias.n_causes = 2
moin_bias.n_evecs = 50
moin_bias.batch_size = 64
moin_bias.img_size = (211, 173)
moin_bias.debug = True
moin_bias.path = f'/data/Data/SimBA-MACAW/{moin_bias.exp_name}'
moin_bias.device = 'cuda'
moin_bias.flow = {'nl': 4, 'hm': [4, 6, 6, 4], 'batch_norm': True}
moin_bias.training = {'epochs': 200, 'batch_size': 64, 'patience': 10, 'min_delta': 50}
moin_bias.optim = {'weight_decay': 0.00005, 'optimizer': "Adam", 'lr': 0.001, 'beta1': 0.9, 'amsgrad': False,
                   'scheduler': True}
moin_bias.sfcn = {'batch_size': 32, 'workers': 4, 'epochs': 200, 'max_images': -1, 'lr': 0.001, 'patience': 5}
EXPS_REGISTRY[moin_bias.exp_name] = moin_bias

no_bias = Exps()
no_bias.exp_name = 'no_bias'
no_bias.dim_red = 'PCA'
no_bias.n_comps = 500
no_bias.n_causes = 2
no_bias.n_evecs = 50
no_bias.batch_size = 64
no_bias.img_size = (211, 173)
no_bias.debug = True
no_bias.path = f'/data/Data/SimBA-MACAW/{no_bias.exp_name}'
no_bias.device = 'cuda'
no_bias.flow = {'nl': 4, 'hm': [4, 6, 6, 4], 'batch_norm': True}
no_bias.training = {'epochs': 200, 'batch_size': 64, 'patience': 10, 'min_delta': 50}
no_bias.optim = {'weight_decay': 0.00005, 'optimizer': "Adam", 'lr': 0.001, 'beta1': 0.9, 'amsgrad': False,
                   'scheduler': True}
no_bias.sfcn = {'batch_size': 32, 'workers': 4, 'epochs': 200, 'max_images': -1, 'lr': 0.001, 'patience': 5}
EXPS_REGISTRY[no_bias.exp_name] = no_bias

far_bias = Exps()
far_bias.exp_name = 'far_bias'
far_bias.dim_red = 'PCA'
far_bias.n_comps = 500
far_bias.n_causes = 2
far_bias.n_evecs = 50
far_bias.batch_size = 64
far_bias.img_size = (211, 173)
far_bias.debug = True
far_bias.path = f'/data/Data/SimBA-MACAW/{far_bias.exp_name}'
far_bias.device = 'cuda'
far_bias.flow = {'nl': 4, 'hm': [4, 6, 6, 4], 'batch_norm': True}
far_bias.training = {'epochs': 200, 'batch_size': 64, 'patience': 10, 'min_delta': 50}
far_bias.optim = {'weight_decay': 0.00005, 'optimizer': "Adam", 'lr': 0.001, 'beta1': 0.9, 'amsgrad': False,
                  'scheduler': True}
far_bias.sfcn = {'batch_size': 32, 'workers': 4, 'epochs': 5, 'max_images': -1, 'lr': 0.0001, 'patience': 5}
EXPS_REGISTRY[far_bias.exp_name] = far_bias


def setup_experiments(name) -> Exps:
    return EXPS_REGISTRY[name]
