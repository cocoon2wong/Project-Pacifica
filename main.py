"""
@Author: Conghao Wong
@Date: 2022-06-20 15:28:14
@LastEditors: Conghao Wong
@LastEditTime: 2023-12-18 11:17:06
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import sys

import torch

import qpid

torch.autograd.set_detect_anomaly(True)


def main(args: list[str], run_train_or_test=True):

    min_args = qpid.args.Args(args, is_temporary=True)
    if min_args.help != 'null':
        qpid.print_help_info('all_args')
        exit()

    if (model := min_args.model) == 'linear':
        s = qpid.applications.Linear
    else:
        s = qpid.silverballers.SILVERBALLERS_DICT.get_structure(model)

    t = s(args)

    if run_train_or_test:
        t.train_or_test()

    # It is used to debug
    if t.args.verbose:
        t.print_info_all()

    return t

if __name__ == '__main__':
    main(sys.argv)
