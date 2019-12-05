def Stepwise(**kwargs):
    default_kwargs = dict(
        epochs=150, 
        lr=1e-1, 
        policy='fixed',
        warmup=5,
        multisteps=[50, 100], 
        batch_size=128,
        accum_grad=1,
        fp16=True,
        monitor=['acc'],
        init='xavier',
    )
    default_kwargs.update(kwargs)
    return default_kwargs
