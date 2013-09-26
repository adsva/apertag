from collections import defaultdict
from collections import Counter
from itertools import izip
import cPickle
import logging
import random

NOTAG = '_N_'

log = logging.getLogger('apertag')

class Weight(object):
    def __init__(self):
        self.weight = 0.0
        self.total = 0.0
        self.iterations = 0

    def __repr__(self):
        return '<Weight w: {:f} t: {:f} i: {:d}>'.format(
            self.weight, self.total, self.iterations)

def defaultweights():
    return defaultdict(Weight)

class AveragedPerceptron(object):

    def __init__(self):
        self.weights = defaultdict(defaultweights)
        self.tags = set([NOTAG])
        self.i = 0

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
        self.i += 1
        for (feature, tag), weight in weights.iteritems():
            self.tags.add(tag)
            if weight:
                w = self.weights[feature][tag]
                # Update the weight sum with the last registered weight
                # for every iteration since it was updated
                w.total += (self.i - w.iterations) * w.weight

                # Update the weight and the total
                w.weight += weight
                w.total += w.weight

                # Store the update iteration
                w.iterations = self.i

    def score(self, features, tag):
        score = 0
        for feature in features:
            if feature in self.weights:
                if tag in self.weights[feature]:
                    score += self.weights[feature][tag].weight
        return score

    def average(self):
        """
        Average the weights across all updates and reset totals
        """
        for feature, tags in self.weights.iteritems():
            for tag, w in tags.iteritems():
                # Make sure all updates are accounted for in the total
                w.total += (self.i - w.iterations) * w.weight
                w.weight = w.total/self.i if self.i else 0
                # Reset the feature so it's suitable for re-training.
                w.total = w.weight
                w.iterations = 0



class Tagger(object):
    """
    A sequence labeler using an averaged perceptron model

    To avoid making assumptions about what kind of sequence data you
    are labeling, or the format of your features, the input to the
    tagger is simply sequences of feature value sets. Each set of
    values represent an observation to receive a tag. A feature value
    can be any python type, as long as it can be hashed, but it's
    important to note that the the values are used only in a binary
    fashion, i.e. either they exist in the context of the item being
    tagged or not; the nature of the value has no impact on the
    decision.

    A simple example illustrating an NP-chunker:
    >>> t = Tagger()
    >>> t.train([[['POS=DT','WRD=the'],['POS=NN','WRD=dog']]],[['NP-B','NP-I']])
    >>> t.tag([['POS=DT','WRD=the'],['POS=NN','WRD=dog']])
    ['NP-B', 'NP-I']

    There is one crucial exception to all this featuratory freedom:
    Any features wishing to make use of the actual output tags need to
    signal this by formatting their value as a string with special
    tags that will be replaced by the corresponding tags from the
    current context during tagging. The tag format is "<Tn>", where n
    is the negative index of the tag relative to the current
    position. For example, if you are training a POS-tagger and you
    have a feature that looks at the current word and the previous
    output tag, and the current word is "dog", the feature could be
    encoded as "<T1>:dog". The tagger will expand this using its
    predicted label context into something like "DT:dog" (depending on
    your tag set and feature format, of course).

    An example illustrating a POS-tagger with output label features:
    >>> t = Tagger()
    >>> t.train([[['POS -1:<T-1>','W:the'],['POS -1:<T-1>','W:dog']]],[['DT','NN']])
    >>> t.tag([['POS -1:<T-1>','W:the'],['POS -1:<T-1>','W:dog']])
    ['DT', 'NN']

    It is most likely a good idea to use this format for training as
    well, even though you (hopefully) have the output tags yourself at
    that point, to ensure the features are identical across training
    and tagging.

    If you don't require output tags for any of your features, you can
    slightly increase performance (especially for non-string features)
    by setting expand_features=False.
    """

    def __init__(self, model=None, beam_size=3, iterations=10, expand_features=True):
        if isinstance(model, file):
            log.info('Loading model from {:s}'.format(model.name))
            model = cPickle.load(model)
        self.model = model or AveragedPerceptron()
        if beam_size < 1:
            raise Exception('Beam must be >= 1')
        self.beam_size = beam_size
        self.iterations = iterations
        self.expand_features = expand_features
        self.tag_p = re.compile(r'<T-?(\d+)>')

    def _expanded_features(self, features, prev_tags):
        """
        Generator that adds context tags to features

        A feature can be expanded with output tags from the current
        context using tags of the format <Tn>, where n is the negative
        index of the tag relative to the current position. Only
        negative indexes are allowed, but they may be specified
        signless.

        Example:
        >>> t = Tagger()
        >>> list(t._expanded_features(['POS -1:<T-1>','WRD=dog'],['VB','DT']))
        ['POS -1:DT', 'WRD=dog']
        >>> list(t._expanded_features(['POS-TRI:<T3>:<T2>:<T1>'],['VB','DT']))
        ['POS-TRI:_N_:VB:DT']
        """
        for feature in features:
            try:
                matches = self.tag_p.finditer(feature)
            except TypeError:
                # Only string features can be expanded
                pass
            else:
                for m in matches:
                    i = int(m.group(1))
                    try:
                        tag = prev_tags[-i]
                    except IndexError:
                        tag = NOTAG
                    feature = feature.replace(m.group(), tag)
            yield feature

    def _expanded_feature_seq(self, feature_seq, tags):
        """
        Generator of expanded features for an entire sequence
        """
        for i, features in enumerate(feature_seq):
            yield self._expanded_features(features, tags[:i])

    def _bag_sequence(self, feature_seq, tag_seq):
        """
        Return a counted set of feature-tag pairs for the given sequence

        Counted feature sets are used during training to determine
        which features differed between the training sequence and the
        predicted sequence, so their weights can be updated
        accordingly. Features requiring tag context are expanded using
        the provided tags first.
        """
        if self.expand_features:
            feature_seq = self._expanded_feature_seq(feature_seq, tag_seq)
        return Counter((f,t) for fs, t in izip(feature_seq, tag_seq) for f in fs)

    def train(self, feature_seqs, tag_seqs, iterations=None):
        """
        Set the model parameters and optimize the weights

        Arguments:
        feature_seqs -- A sequence of training sequences, each
          consisting of a sequence of feature value sequences.
        tag_seqs     -- A sequence of tag sequences, providing
          the labels for the feature sequences.

        Example:
        >>> t = Tagger()
        >>> t.train([[['POS=DT','WRD=the'],['POS=NN','WRD=dog']]],[['NP-B','NP-I']])

        Starting with an empty model, the trainer labels the supplied
        training data, evaluates the result and updates the model
        based on its mistakes. This process is repeated for a fixed
        number of iterations, then the feature weights are all
        averaged and the model is ready.
        """
        iterations = iterations or self.iterations

        # Combine the features and the correct tags into counted sets of
        # feature-tag combinations for each sequence
        log.info('Reading training sequences')

        seqs = zip(feature_seqs, tag_seqs)
        num_seqs = len(seqs)

        log.info('Start training using {:d} sequences'.format(num_seqs))
        for i in range(iterations):
            correct_seqs = 0
            # Using the same order for each iteration is bad.
            random.shuffle(seqs)
            for seq_idx, (feature_seq, gold_tags) in enumerate(seqs):
                log.debug('Tagging sequence {:d} of {:d}'.format(seq_idx, num_seqs))
                predicted_tags = self.tag(feature_seq)
                if predicted_tags != gold_tags:
                    # If the predicted tag sequence is not correct,
                    # the weights need to be adjusted.  This is done
                    # by generating a counted set (bag) of all
                    # active feature-tag pairs for both the training
                    # sequence and the predicted sequence, and then
                    # subtract each pair count in the prediction bag
                    # from the corresponding count in the training
                    # bag. This way feature-tag pairs from the
                    # training set that were missing in the prediction
                    # set gets upweighted, and pairs from the
                    # prediction set that were not in the training set
                    # get downweighted.
                    gold_bag = self._bag_sequence(feature_seq, gold_tags)
                    # Recreate the expanded features used to predict this
                    # particular sequence and bag them. Better to recreate
                    # the expanded features once per sequence during
                    # training than to add complexity to the tagger.
                    prediction_bag = self._bag_sequence(feature_seq, predicted_tags)
                    gold_bag.subtract(prediction_bag)
                    self.model.update(gold_bag)
                else:
                    correct_seqs += 1
            log.info('Finished training iteration: %d Correct seqs: %d', i+1, correct_seqs)

        log.info('Averaging weights')
        self.model.average()
        log.info('Done training')


    def tag(self, feature_seq):
        paths = [(0,[])]
        for features in feature_seq:
            candidates = []
            for path in paths:
                # Add local tag context from this path to features
                # that require it
                if self.expand_features:
                    features = list(self._expanded_features(features, path[1]))
                for tag in self.model.tags:
                    score = path[0] + self.model.score(features, tag)
                    candidates.append((score, path[1]+[tag]))

            # Prune the candidates
            candidates.sort(reverse=True)
            paths = candidates[:self.beam_size]

        paths.sort(reverse=True)
        return paths[0][1]

    def export_model(self, f):
        if isinstance(f, basestring):
            f = open(f, 'w')
        cPickle.dump(self.model, f, cPickle.HIGHEST_PROTOCOL)
        f.close()

if __name__ == '__main__':
    import argparse
    from itertools import tee
    import re

    def read_sequences(f):
        """
        Generate sequences as lists of columns from the input file
        """
        sequence = []
        for line in f:
            line = line.strip('\n')
            if not line and sequence:
                yield sequence
                sequence = []
            else:
                sequence.append(line.split('\t'))

    def split_sequences(f):
        """
        Create separate iterators for feature columns and tag column
        """
        s1, s2 = tee(read_sequences(f))
        features = [[row[:-1] for row in s] for s in s1]
        tags = [[row[-1] for row in s] for s in s2]
        return features, tags

    def train(args):
        t = Tagger(beam_size=args.beam_size, iterations=args.iterations)
        features, tags = split_sequences(args.training_sequences)
        t.train(features, tags)
        with args.model as f:
            t.export_model(args.model)

    def tag(args):
        t = Tagger(model=args.model)
        with args.tags as f:
            if args.eval:
                i,c = 0,0
                seqs, tag_seqs = split_sequences(args.sequences)
                for seq, gold_tags in izip(seqs, tag_seqs):
                    i += len(gold_tags)
                    tags = t.tag(seq)
                    c += sum(p == g for p,g in izip(tags, gold_tags))
                    f.write('\n'.join(tags) + '\n\n')
                print 'Accuracy: {:.2f}'.format(c/float(i))
            else:
                for seq in read_sequences(args.sequences):
                    tags = t.tag(seq)
                    f.write('\n'.join(tags) + '\n\n')

    def test(args):
        log.setLevel(logging.CRITICAL)
        import doctest
        doctest.testmod()

    parser = argparse.ArgumentParser()

    # Logging options
    g = parser.add_mutually_exclusive_group()
    g.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Log everything.')
    g.add_argument('-q', '--quiet', action='store_true', default=False,
                   help='Log nothing.')

    subparsers = parser.add_subparsers()

    # Training options
    p = subparsers.add_parser('train')
    p.add_argument('training_sequences', type=argparse.FileType('r'),
                   help='Read training sequences from this file. '
                   'A sequences consists of lines of tab delimited feature '
                   'columns with the output tag in the last column. '
                   'Sequences are separated by double newlines. '
                   'Use - for stdin.')
    p.add_argument('model', type=argparse.FileType('w'),
                   help='Write the resulting model to this file. Use - for stdout.')
    p.add_argument('-b', '--beam-size', type=int, default=3,
                   help='Number of best paths to keep at each prediction step.')
    p.add_argument('-i', '--iterations', type=int, default=10,
                   help='Number iterations used when training.')
    p.set_defaults(func=train)

    # Tagging options
    p = subparsers.add_parser('tag')
    p.add_argument('model', type=argparse.FileType('r'),
                   help='Read model from this file.')
    p.add_argument('sequences', type=argparse.FileType('r'),
                   help='Read feature sequences from this file. '
                   'A sequences consists of lines of tab delimited feature '
                   'columns. Sequences are separated by double newlines. '
                   'Use - for stdin.')
    p.add_argument('tags', type=argparse.FileType('w'),
                   help='Write the resulting output sequences in a single column '
                   'to this file. Use - for stdout.')
    p.add_argument('-b', '--beam-size', type=int, default=3,
                   help='Number of best paths to keep at each prediction step')
    p.add_argument('-e', '--eval', action='store_true',
                   help='Indicates that the input data is labeled. Uses last column to calculate accuracy.')
    p.set_defaults(func=tag)

    # Test options
    p = subparsers.add_parser('test')
    p.set_defaults(func=test)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig()
    if args.verbose:
        log.setLevel(logging.DEBUG)
    elif args.quiet:
        log.setLevel(logging.CRITICAL)
    else:
        log.setLevel(logging.INFO)

    args.func(args)
