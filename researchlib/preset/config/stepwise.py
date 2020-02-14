def Stepwise(**kwargs):
    default_kwargs = dict(
        epochs=150, 
        lr=1e-1, 
        policy='fixed',
        warmup=5,
        multisteps=[0.5, 0.75], 
        batch_size=128,
        accum_grad=1,
        fp16=True,
        init='kaiming_normal',
    )
    default_kwargs.update(kwargs)
    return default_kwargs


def Stepwise300(**kwargs):
    default_kwargs = dict(
        epochs=300, 
        lr=1e-1, 
        policy='fixed',
        warmup=5,
        multisteps=[0.5, 0.75], 
        batch_size=128,
        accum_grad=1,
        fp16=True,
        init='kaiming_normal',
    )
    default_kwargs.update(kwargs)
    return default_kwargs