#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class HomomorphicProvider:
    NAME = "HE"

    @staticmethod
    def generate_additive_triple(size0, size1, op, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    @staticmethod
    def square(size):
        """Generate square double of given size"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    @staticmethod
    def generate_xor_triple(size0, size1):
        """Generate xor triples of given size"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    @staticmethod
    def wrap_rng(size, num_parties):
        """Generate random shared tensor of given size and sharing of its wraps"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    @staticmethod
    def B2A_rng(size):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        raise NotImplementedError("HomomorphicProvider not implemented")
