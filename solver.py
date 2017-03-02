#!/usr/bin/env python
# coding: utf-8

import typing, itertools
from operator import attrgetter

flatten = itertools.chain.from_iterable

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)

def assert_almost_sums_to_one(probabilities):
    epsilon = .00000001
    assert abs(1 - sum(probabilities)) < epsilon, 'Probability tables need to sum to (almost) 1'
    
class Reference(object):
    
    def __init__(self, name, table):
        self.name = name
        self.table = table
    
    def __repr__(self):
        if self.table._name is None:
            return self.name
        
        return '%s.%s' % (self.table._name, self.name)
    __str__ = __repr__

class ProbabilityTable(object):
    
    @classmethod
    def independent(cls, **kwargs):
        assert_almost_sums_to_one(kwargs.values())
        return cls(tuple(kwargs.keys()), kwargs, dependencies=())
    
    @classmethod
    def dependent(cls, labels, dependent_values):
        references, probability_rows = tuple(dependent_values.keys()), tuple(dependent_values.values())
        
        assert len(labels) == len(probability_rows[0]), 'Need one label for each probability %r, %r' % (labels, probability_rows[0])
        assert len(set(map(len, probability_rows))) == 1,\
            'Need the same number of probabilites for each row'
        
        
        if isinstance(references[0], Reference): # single dependency table
            assert all([isinstance(reference, Reference) for reference in references]),\
                'Needs consistent numbers of references to other tables.'
            
            # normalize structure
            references = tuple((reference, ) for reference in references)
        
        for reference in references:
            assert len(set(map(attrgetter('table'), reference))) == len(reference), \
                'All references for a row need to point to different tables'
            
        key = lambda reference: id(reference.table)
        by_table = itertools.groupby(
            sorted(flatten(references), key=lambda reference: id(reference.table)),
            key=lambda x: x.table
        )
        dependencies = []
        for table, keys in by_table:
            dependencies.append(table)
            keys = set(keys) # consume group
            assert keys == set(table._labels), \
                'Missing keys of table %s, only have %s, expecting %s' \
                % (table, keys, set(table._labels))
        
        values = dict()
        for keys, probabilities in zip(references, probability_rows):
            assert_almost_sums_to_one(probabilities)
            for self_key, value in zip(labels, probability_rows):
                values[keys + (self_key, )] = value
        
        cross_product_of_dependencies_keys = set(map(frozenset, itertools.product(*map(attrgetter('_labels'), dependencies))))
        assert set(map(frozenset, references)) == cross_product_of_dependencies_keys, \
            "References to other tables need to be a full product of their labels. Expected %r, \nbut got %r" \
            % (set(references), cross_product_of_dependencies_keys)
        
        return cls(labels, values, dependencies=tuple(dependencies))
    
    def __init__(self, labels, values, dependencies):
        self._network = None # to be set by network
        self._name = None # to be set by network
        # self._labels = [] # set in _set_references
        self._set_references(labels)
        self._dependencies = dependencies
        self._values = dict()
        
        self._values = { self._normalize_keys(key): value for key, value in values.items()}
    
    def _set_references(self, labels):
        self._labels = tuple(map(lambda key: Reference(key, self), labels))
        for reference in self._labels:
            setattr(self, reference.name, reference)
    
    def __getitem__(self, key_or_keys):
        keys = self._normalize_keys(key_or_keys)
        self._assert_keys_are_sufficient(keys)
        return self._values[keys]
    
    def _normalize_keys(self, key_or_keys):
        keys = (key_or_keys,) if isinstance(key_or_keys, (str, Reference)) else key_or_keys
        def to_reference(key):
            if isinstance(key, Reference): return key
            return getattr(self, key)
        return frozenset(map(to_reference, keys))
    
    def _assert_keys_are_sufficient(self, keys):
        print(tuple(self._values.keys())[0], keys)
        assert len(tuple(self._values.keys())[0]) == len(keys), 'Need the full set of keys to get a probability'
        assert any(filter(lambda x: x.table == self, keys)), 'Does not contain key to self'
    
    def __repr__(self):
        display_values = ', '.join(['%r: %s' % (set(key), value) for key, value in self._values.items()])
        name = self._name if self._name is not None else 'ProbabilityTable'
        return '%s(%s)' % (name, display_values)
    __str__ = __repr__
    
    def _suitable_key_subset(self, keys):
        return filter(lambda key: key == self or key in self._dependencies, keys)

class BayesianNetwork(object):
    
    def __init__(self):
        for name, table in self._tables().items():
            table._network = self
            table._name = name
    
    def _tables(self):
        # would be a desaster if ProbabilityTable's are added after construction - but that is currently
        # prevented by design
        if not hasattr(self, '__tables'):
            self.__tables = dict()
            for name, table in vars(self.__class__).items():
                if len(name) == 1 or name[0] == '_': continue # shortname, or private
                self.__tables[name] = table
        return self.__tables
    
    def probability(self, *for_, given=()):
        result = 1
        for table in self._tables().values():
            print(table, for_, list(table._suitable_key_subset(for_)))
            result * table[table._suitable_key_subset(for_)]
        return result

class Student(BayesianNetwork):
    i = intelligence = ProbabilityTable.independent(low=.7, high=.3)
    d = difficulty = ProbabilityTable.independent(low=.6, high=.4)
    
    s = sat = ProbabilityTable.dependent(
                ('bad', 'good'), {
        i.low:  (.95,   .05),
        i.high: (.2,    .8)
    })
    g = grade = ProbabilityTable.dependent(
                          ('good', 'ok', 'bad'), {
        (i.low, d.low):   (.3,     .4,   .3),
        (i.low, d.high):  (.05,    .25,  .7),
        (i.high, d.low):  (.9,     .08,  .02),
        (i.high, d.high): (.5,     .3,   .2),
    })
    l = letter = ProbabilityTable.dependent(
                ('bad', 'glowing'), {
        g.good: (.1,    .9),
        g.ok:   (.4,    .6),
        g.bad:  (.99,   .01),
    })

n = network = Student()
# print(n.intelligence.low, n.i[n.i.low])
# print(n.difficulty.low, n.d[n.d.low])
#
# print(n.sat.bad, n.intelligence.low, n.sat[n.s.bad, n.i.low])
#
# print(n.intelligence.low, n.difficulty.low, n.grade.good, n.g[n.i.low, n.d.low, n.g.good])
# print(n.intelligence.low, n.difficulty.low, n.grade.good, n.g[n.i.low, n.g.good, n.d.low])
#
# print(n.letter.bad, n.grade.good, n.l[n.l.bad, n.g.good])
#
n.probability(n.i.low, n.d.low, n.s.bad, n.g.good, n.l.glowing)
# print('i1, d0, g2, l0, s1', probability('i1', 'd0', 'g2', 'l0', 's1'), 'should be', 0.004608)
