import requests
import PIL.Image
import io
import numpy as np


def _ExtendSequence(_y, max_length, value = 0):
    pad = np.zeros((_y.shape[0], max_length - _y.shape[1], *_y.shape[2:])) + value
    print(_y.shape, pad.shape)
    return np.concatenate([_y, pad], axis=1)


def _LoadEmoji(emoji, size=40):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u%s.png'%code
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((size, size), PIL.Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:]
    return img


class utils(object):
    ExtendSequence = _ExtendSequence
    LoadEmoji = _LoadEmoji