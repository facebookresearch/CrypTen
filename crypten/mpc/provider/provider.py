#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class TupleProvider:
    TRACEABLE_FUNCTIONS = [
        "generate_additive_triple",
        "square",
        "generate_binary_triple",
        "wrap_rng",
        "B2A_rng",
    ]

    def __init__(self):
        self.tracing = False
        self.request_cache = []
        self.tuple_cache = {}

    def trace(self, tracing=True):
        self.tracing = tracing

    def trace_once(self):
        untraced = self.request_cache.empty()
        self.trace(tracing=untraced)

    def __getattribute__(self, func_name):
        if func_name not in TupleProvider.TRACEABLE_FUNCTIONS:
            return object.__getattribute__(self, func_name)

        # Trace requests while tracing
        if self.tracing:

            def func_with_trace(*args, **kwargs):
                request = (func_name, args, kwargs)
                self.request_cache.append(request)
                return object.__getattribute__(self, func_name)(*args, **kwargs)

            return func_with_trace

        # If the cache is empty, call function directly
        if len(self.tuple_cache) == 0:
            return object.__getattribute__(self, func_name)

        # Return results from cache if available
        def func_from_cache(*args, **kwargs):
            hashable_kwargs = frozenset(kwargs.items())
            request = (func_name, args, hashable_kwargs)
            # Read from cache
            if request in self.tuple_cache.keys():
                return self.tuple_cache[request].pop()
            # Cache miss
            return object.__getattribute__(self, func_name)(*args, **kwargs)

        return func_from_cache

    def fill_cache(self):
        # TODO: parallelize / async this
        for request in self.request_cache:
            func_name, args, kwargs = request
            result = object.__getattribute__(self, func_name)(*args, **kwargs)

            hashable_kwargs = frozenset(kwargs.items())
            hashable_request = (func_name, args, hashable_kwargs)
            if hashable_request in self.tuple_cache.keys():
                self.tuple_cache[hashable_request].append(result)
            else:
                self.tuple_cache[hashable_request] = [result]

    def generate_additive_triple(self, size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        raise NotImplementedError(
            "TupleProvider generate_additive_triple not implemented."
        )

    def square(self, size, device=None):
        """Generate square double of given size"""
        raise NotImplementedError("TupleProvider square not implemented.")

    def generate_binary_triple(self, size0, size1, device=None):
        """Generate xor triples of given size"""
        raise NotImplementedError(
            "TupleProvider generate_binary_triple not implemented."
        )

    def wrap_rng(self, size, device=None):
        """Generate random shared tensor of given size and sharing of its wraps"""
        raise NotImplementedError("TupleProvider wrap_rng not implemented.")

    def B2A_rng(self, size, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        raise NotImplementedError("TupleProvider B2A_rng not implemented.")
