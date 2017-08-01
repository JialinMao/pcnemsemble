#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Abstract for target distribution. 
"""


class Distribution(object):
    def __init__(self, **kwargs):
        """
        for simple distributions, can just pass in lnprob function
        with the keyword 'lnprob' & arguments.
        """
        self.f = kwargs.get('lnprob', None)
        self.args = kwargs
        try:
            del self.args['lnprob']
        except KeyError:
            pass

    def get_lnprob(self, x):
        if self.f is not None:
            return self.f(x, self.args)
        else:
            raise NotImplementedError

    def get_auto_corr_f(self, chain):
        return chain

    def set(self, k, v):
        self.args[k] = v

    def get(self, k):
        return self.args[k]

