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
moin_bias.crop_size = 180
moin_bias.debug = True
moin_bias.path = f'/data/Data/SimBA-MACAW/{moin_bias.exp_name}'
moin_bias.device = 'cuda'
moin_bias.flow = {'nl': 4, 'hm': [4, 6, 6, 4], 'batch_norm': True}
moin_bias.training = {'epochs': 200, 'batch_size': 64, 'patience': 10, 'min_delta': 50}
moin_bias.optim = {'weight_decay': 0.00005, 'optimizer': "Adam", 'lr': 0.001, 'beta1': 0.9, 'amsgrad': False,
                   'scheduler': True}
EXPS_REGISTRY[moin_bias.exp_name] = moin_bias


def setup_experiments(name) -> Exps:
    return EXPS_REGISTRY[name]
