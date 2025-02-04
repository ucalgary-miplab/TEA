EXPS_REGISTRY = {}


class Exps:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


exp = Exps()
exp.exp_name = "exp205"
exp.dim_red = "PCA"
exp.n_comps = 500
exp.n_causes = 2
exp.n_evecs = 50
exp.batch_size = 64
exp.img_size = (192, 192)
exp.debug = True
exp.path = f"data/{exp.exp_name}"
exp.device = "cuda"
exp.flow = {"nl": 5, "hm": [4, 6, 6, 4], "batch_norm": True}
exp.training = {"epochs": 200, "batch_size": 64, "patience": 4, "min_delta": 50}
exp.optim = {
    "weight_decay": 0.00005,
    "optimizer": "AdamW",
    "lr": 0.001,
    "beta1": 0.9,
    "amsgrad": False,
    "scheduler": True,
}
exp.sfcn = {
    "batch_size": 32,
    "workers": 4,
    "epochs": 5,
    "max_images": -1,
    "lr": 0.0001,
    "patience": 5,
}
EXPS_REGISTRY[exp.exp_name] = exp


def setup_experiments(name) -> Exps:
    return EXPS_REGISTRY[name]
