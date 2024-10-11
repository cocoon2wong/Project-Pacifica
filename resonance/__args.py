"""
@Author: Conghao Wong
@Date: 2024-10-08 19:11:16
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-11 14:48:13
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
    def compute_non_social_bias(self) -> int:
        return self._arg('compute_non_social_bias', 1, argtype=STATIC)

    @property
    def temp_fix(self) -> int:
        return self._arg('temp_fix', 0, argtype=TEMPORARY)
    
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
