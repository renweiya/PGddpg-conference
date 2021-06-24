import tensorflow as tf
import numpy as np
import algorithm.common.tf_utils as U
from tensorflow.python.ops import math_ops
import random

class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def logp(self, x):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

class PdType(object):
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)

class SoftCategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return SoftCategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return [self.ncat]

    def sample_dtype(self):
        return tf.float32

class SoftCategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return U.softmax(self.logits, axis=-1)

    def logp(self, x):
        return -tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=x)

    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        z1 = U.sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=1)

    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=1)

    def sample(self, deterministic=False, step_explore=0.0, explore_direction=1.0, n_steps_annealings=1.0,gama_gussian=0.0):# laker
        # explore_direction: main direction of the explore
        # step_explore: whether to explore
        # n_steps_annealings: a parameter to adjust explore
    
        # u = tf.random_uniform(tf.shape(self.logits))
        # return U.tanh(self.logits - (step_explore+0.0000000000000000000001)*explore_direction*(1.0/n_steps_annealings)*tf.log(-tf.log(u)))
        # return U.tanh(tf.random_normal(self.logits, stddev=step_explore/n_steps_annealings, seed=1))
        # return U.tanh(tf.random_normal(self.logits, stddev=step_explore/n_steps_annealings, seed=1))
        
        u = tf.random_uniform(tf.shape(self.logits),minval=0,maxval=1,)

        self.logits=tf.clip_by_value(self.logits, -10000, 10000)#laker
        if deterministic:
            # return U.tanh(self.logits), U.tanh(self.logits - explore_direction*(1.0/n_steps_annealings)*tf.log(-tf.log(u)))
            out=tf.random_normal(tf.shape(self.logits),U.tanh(self.logits), stddev=0.3)
            # out=tf.random_normal(tf.shape(self.logits),U.tanh(self.logits), stddev=1.0)
            #out=tf.random_normal(tf.shape(self.logits),U.tanh(self.logits), stddev=0.5-tf.abs(0.5-tf.abs(U.tanh(self.logits))))
            
            return U.tanh(self.logits), tf.clip_by_value(out, -1, 1)

            # return U.tanh(self.logits), tf.random_normal(tf.shape(self.logits),U.tanh(self.logits), stddev=0.3)
            # return U.tanh(self.logits), U.tanh(tf.random_normal(self.logits, stddev=step_explore/n_steps_annealings, seed=1))
        else: #no explore
            return U.tanh(self.logits) 
            # return U.tanh(self.logits - explore_direction*(1.0/n_steps_annealings)*tf.log(-tf.log(u))) #evaluate the noisy action perform better
            # return tf.random_normal(U.tanh(self.logits - explore_direction*tf.log(-tf.log(u))), stddev=gama_gussian, seed=1)   
            
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def make_pdtype(ac_space):
    return SoftCategoricalPdType(2)

