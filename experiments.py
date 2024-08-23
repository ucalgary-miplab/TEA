EXPS_REGISTRY = {}


class Exps:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


moin_bias = Exps()
moin_bias.exp_name = 'moin_bias'
moin_bias.dim_red = 'PCA'
moin_bias.n_comps = 500
moin_bias.batch_size = 64
moin_bias.crop_size = 180
moin_bias.debug = True
moin_bias.path = f'/data/Data/SimBA-MACAW/{moin_bias.exp_name}'
EXPS_REGISTRY[moin_bias.exp_name] = moin_bias


def setup_experiments(name) -> Exps:
    return EXPS_REGISTRY[name]
