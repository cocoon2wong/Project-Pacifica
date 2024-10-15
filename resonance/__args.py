"""
@Author: Conghao Wong
@Date: 2024-10-08 19:11:16
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-15 15:22:11
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class ResonanceArgs(EmptyArgs):

    @property
    def Kc(self) -> int:
        """
        The number of style channels in `Agent` model.
        """
        return self._arg('Kc', 20, argtype=STATIC,
                         desc_in_model_summary='Output channels')

    @property
    def partitions(self) -> int:
        """
        The number of partitions when computing the angle-based feature.
        """
        return self._arg('partitions', -1, argtype=STATIC,
                         desc_in_model_summary='Number of Angle-based Partitions')

    @property
    def T(self) -> str:
        """
        Transformation type used to compute trajectory spectrums
        on the ego agents.

        It could be:
        - `none`: no transformations
        - `fft`: fast Fourier transform
        - `haar`: haar wavelet transform
        - `db2`: DB2 wavelet transform
        """
        return self._arg('T', 'fft', argtype=STATIC, short_name='T',
                         desc_in_model_summary='Transform type (trajectory)')

    @property
    def Tr(self) -> str:
        """
        Transformation type used to compute trajectory spectrums
        on all neighbor agents for modeling social interactions.

        It could be:
        - `none`: no transformations
        - `fft`: fast Fourier transform
        - `haar`: haar wavelet transform
        - `db2`: DB2 wavelet transform
        """
        return self._arg('Tr', 'fft', argtype=STATIC, short_name='Tr',
                         desc_in_model_summary='Transform type (resonance)')

    @property
    def interp(self) -> str:
        """
        Type of interpolation method used to compute bias loss.
        It accepts `linear` (for linear interpolation) and `speed` (for linear
        speed interpolation).
        """
        return self._arg('interp', 'speed', argtype=STATIC)

    @property
    def no_interaction(self) -> int:
        return self._arg('no_interaction', 0, argtype=TEMPORARY)

    @property
    def learn_self_bias(self) -> int:
        """
        Choose whether to compute the self-bias term when training.
        """
        return self._arg('learn_self_bias', 1, argtype=STATIC,
                         other_names=['compute_non_social_bias'],
                         desc_in_model_summary='Predict self-bias trajectories')

    @property
    def learn_re_bias(self) -> int:
        """
        Choose whether to compute the re-bias term when training.
        """
        return self._arg('learn_re_bias', 1, argtype=STATIC,
                         desc_in_model_summary='Predict resonance-bias trajectories')

    @property
    def use_original_socialcircle(self) -> int:
        """
        Choose to use the `ResonanceCircle` (default) or the original
        `SocialCircle` when represent social interactions.
        """
        return self._arg('use_original_socialcircle', 0, argtype=STATIC,
                         desc_in_model_summary='Use SocialCircle rather than ResonanceCircle')

    @property
    def no_linear_bias(self) -> int:
        return self._arg('no_linear_bias', 0, argtype=TEMPORARY)

    @property
    def no_re_bias(self) -> int:
        return self._arg('no_re_bias', 0, argtype=TEMPORARY)

    def _init_all_args(self):
        super()._init_all_args()

        if self.interp not in ['linear', 'speed']:
            self.log(f'Wrong interpolation type `{self.interp}` reveived. ' +
                     'It accepts either `linear` or `speed`.',
                     level='error', raiseError=ValueError)

        if self.partitions <= 0:
            self.log(f'Illegal partition settings ({self.partitions})!',
                     level='error', raiseError=ValueError)
