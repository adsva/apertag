# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
from collections import defaultdict
from collections import Counter
from itertools import izip
import cPickle
import logging
import random
import re
import numpy as np
cimport numpy as np

NOTAG = '_N_'

log = logging.getLogger('apertag')

cdef class AveragedPerceptron:
    cdef public int [:,:] weights
    cdef public int [:,:] totals
    cdef public int [:,:] iterations
    cdef public dict features
    cdef public dict tags
    cdef public list taglist
    cdef public int num_tags
    cdef public int num_feats
    cdef public int i

    def __cinit__(self):
        self.features = {}
        self.num_feats = 0
        self.tags = {NOTAG:0}
        self.taglist = [NOTAG]
        self.num_tags = 1
        self.weights = np.zeros((0,1), dtype=np.int32)
        self.totals = np.zeros((0,1), dtype=np.int32)
        self.iterations = np.zeros((0,1), dtype=np.int32)
        self.i = 0

    def __reduce__(self):
        w_arr = np.empty_like(self.weights)
        w_arr[:] = self.weights
        t_arr = np.empty_like(self.totals)
        t_arr[:] = self.totals
        i_arr = np.empty_like(self.iterations)
        i_arr[:] = self.iterations

        state = dict(
            features=self.features,
            num_feats = self.num_feats,
            tags=self.tags,
            taglist=self.taglist,
            num_tags=self.num_tags,
            weights = w_arr,
            totals=t_arr,
            iterations = i_arr,
            i=self.i
        )

        return (AveragedPerceptron,(),state)


    def __setstate__(self, state):
        for k,v in state.items():
            setattr(self, k, v)

    def tag_id(self, str tag):
        cdef int ti
        if tag in self.tags:
            return self.tags[tag]
        else:
            ti = self.num_tags
            self.tags[tag] = ti
            self.taglist.append(tag)
            self.num_tags += 1
            self.weights = np.concatenate(
                (self.weights, np.zeros((self.num_feats, 1), dtype=np.int32)),
                axis=1)
            self.totals = np.concatenate(
                (self.weights, np.zeros((self.num_feats, 1), dtype=np.int32)),
                axis=1)
            self.iterations = np.concatenate(
                (self.weights, np.zeros((self.num_feats, 1), dtype=np.int32)),
                axis=1)
            return ti

    def feature_id(self, str feature):
        cdef int fi
        if feature in self.features:
            return self.features[feature]
        else:
            fi = self.num_feats
            self.features[feature] = fi
            self.num_feats += 1
            self.weights = np.concatenate(
                (self.weights, np.zeros((1, self.num_tags), dtype=np.int32)),
                axis=0)
            self.totals = np.concatenate(
                (self.weights, np.zeros((1, self.num_tags), dtype=np.int32)),
                axis=0)
            self.iterations = np.concatenate(
                (self.weights, np.zeros((1, self.num_tags), dtype=np.int32)),
                axis=0)
            return fi

    def update(self, weights):
        """
        Adjust the weights for non-zero feature-tag pairs

        In order to average the weights later, the sum of all feature
        weights accross all updates need to be stored. To avoid adding
        the current weight of every unchanged feature on every update,
        we record the update iteration when a weight actually changes,
        and the next time the weight is about to be changed (or when
        training is completed) the current weight multiplied by the
        number of unrecorded updates is added to the total before
        proceeding.
        """
        cdef str feature
        cdef str tag
        cdef int ti
        cdef int fi
        cdef int weight
        cdef int new
        cdef int old

        self.i += 1
        for (feature, tag), weight in weights.iteritems():
            ti = self.tag_id(tag)
            if weight:
                fi = self.feature_id(feature)
                old = self.weights[fi][ti]
                new = old + weight
                # Update the weight sum with the last registered weight
                # for every iteration since it was updated
                self.totals[fi,ti] += ((self.i - self.iterations[fi,ti]) * old) + new

                # Update the weight
                self.weights[fi][ti] = new

                # Store the update iteration
                self.iterations[fi,ti] = self.i

    def best_tag(self, features):
        cdef int best_tag = 0
        cdef int best_score = 0
        cdef int score
        cdef int tag_id
        cdef int feature_id


        for tag_id in range(self.num_tags):
            score = 0
            for feature in features:
                if feature in self.features:
                    feature_id = self.features[feature]
                    score += self.weights[feature_id, tag_id]
            if score > best_score:
                best_tag = tag_id
                best_score = score

        return self.taglist[best_tag]

    def tag_scores(self, features):
        cdef int score
        cdef int tag_id
        cdef int feature_id
        cdef list scores = []

        for tag_id in range(self.num_tags):
            score = 0
            for feature in features:
                if feature in self.features:
                    feature_id = self.features[feature]
                    score += self.weights[feature_id, tag_id]
            scores.append((self.taglist[tag_id], score))
        return scores

    def average(self):
        """
        Average the weights across all updates and reset totals
        """
        cdef int f
        cdef int t
        cdef int w
        cdef int tot

        if self.i == 0:
            return

        for f in range(self.num_feats):
            for t in range(self.num_tags):
                w = self.weights[f,t]
                tot = self.totals[f,t] + (self.i - self.iterations[f,t]) * w
                self.weights[f,t] = tot/self.i

        self.totals = np.empty_like(self.weights)
        self.totals[:] = self.weights
        self.iterations = np.zeros_like(self.weights)

