#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Abstract for target distribution. 
"""


class Distribution(object):
    def __init__(self, *args, **kwargs):
        pass

    def get_lnprob(self, x):
        raise NotImplementedError
