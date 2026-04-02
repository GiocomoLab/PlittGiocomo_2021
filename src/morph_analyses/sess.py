import two_photon_utils as tpu
# from two_photon_utils.sess import Session


class CA1MorphSession(tpu.sess.Session):
    """Session subclass for CA1 morphology experiment (Plitt & Giocomo 2019)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
