def Poly(**kwargs):
    default_kwargs = dict(
        epochs=150, 
        lr=1e-1, 
        policy='poly2',
        warmup=5,
        batch_size=128,
        accum_grad=1,
        fp16=True,
        init='kaiming_normal',
    )
    default_kwargs.update(kwargs)
    return default_kwargs
