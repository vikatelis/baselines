def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )

def atari():
    return dict(
        nsteps=32, nminibatches=2,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
    )

def optimization():
    return dict(
        nsteps=512, nminibatches=20,
        #best = 512, nmini 10
        #lam=0.95, gamma=0.95, noptepochs=10, log_interval=10,
        lam=0.95, gamma=0.95, noptepochs=10, log_interval=10,
        ent_coef=0.00,
        #vf_coef=0.4,
        #lr=lambda f : f * 2.5e-8,
        lr=lambda f : f*2.5e-3,
        cliprange=lambda f : f*0.1,
        #cliprange=lambda f : f,
    )
