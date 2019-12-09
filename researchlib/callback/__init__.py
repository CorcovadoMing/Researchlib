from .inject import inject_after
from .restore_inject import restore_inject

class Callback(object):
    inject_after = inject_after
    restore_inject = restore_inject
    