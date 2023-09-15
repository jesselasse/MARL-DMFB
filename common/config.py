import os

def config(name, version = None, train=True):
    if name == 'dmfb':
        os.chdir('data-dmfb/')
        if version == '0.1':
            from env.DMFB.dmfb import DMFBenv_v0_1, Chip
            return DMFBenv_v0_1, Chip
        else:
            from env.DMFB.dmfb import DMFBenv, Chip
            if train:
                from env.DMFB.dmfb import TrainingManager as Manager
            else:
                from env.DMFB.dmfb import AssayTaskManager as Manager
            return DMFBenv, Chip, Manager
    elif name == 'meda':
        os.chdir('data-meda/')
        if version == '0.1':
            from env.MEDA.meda import MEDAEnv_v0_1, Chip
            return MEDAEnv_v0_1, Chip
        from env.MEDA.meda import MEDAEnv, Chip
        return MEDAEnv, Chip