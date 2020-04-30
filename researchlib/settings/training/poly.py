def Poly(**kwargs):
    default_kwargs = dict(
        epochs=150, 
        lr=1e-1, 
        policy='poly2',
        warmup=5,
        batch_size=128,
        accum_grad=1,
        fp16=True,
        init='default',
    )
    default_kwargs.update(kwargs)
    return default_kwargs


def Poly300(**kwargs):
    default_kwargs = dict(
        epochs=300, 
        lr=1e-1, 
        policy='poly2',
        warmup=5,
        batch_size=128,
        accum_grad=1,
        fp16=True,
        init='default',
    )
    default_kwargs.update(kwargs)
    return default_kwargs

