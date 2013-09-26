Apertag: Averaged Perceptron Tagger
===================================

Apertag is a sequence tagger based on an averaged perceptron model. In
order to avoid making assumptions about what kind of sequence data you
are labeling, or the format of your features, the input to the tagger
is simply sequences of feature value sets. Each set of values
represent an observation to receive a tag. A feature value can be any
python type, as long as it can be hashed, but it's important to note
that the the values are used only in a binary fashion, i.e. either
they exist in the context of the item being tagged or not; the nature
of the value has no impact on the decision.

A simple example illustrating an NP-chunker:

.. code-block:: python

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

.. code-block:: python

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

Where do I put my columns?
==========================

The tagger module can be run as a standalone script, which takes its
input as good old files of tab delimited columns, where each row is an
observation consisting of feature values, followed by the tag in the
last column. For more info run:

.. code-block:: bash

    $ python apertag.py {train,tag} -h

References and acknowledgments
==============================

* http://people.csail.mit.edu/mcollins/papers/tagperc.pdf
* https://github.com/lmjohns3/py-tagger/
* https://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
