#!/usr/bin/env python
# coding: utf-8

from pyexpect import expect
import itertools
from operator import attrgetter
from fluent import *

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

class Distribution(object):
    # REFACT consider to ask the distribution for partial distributions perhaps by creating one on the fly?
    
    @classmethod
    def independent(cls, **kwargs):
        assert_almost_sums_to_one(kwargs.values())
        return cls(tuple(kwargs.keys()), kwargs, dependencies=())
    
    @classmethod
    def dependent(cls, labels, dependent_values):
        references = tuple(dependent_values.keys())
        probability_rows = tuple(dependent_values.values())
        
        assert _(probability_rows).map(len).call(set).len() == 1,\
            'Need the same number of probabilites for each row'
        
        if isinstance(references[0], Reference): # single dependency table
            # normalize structure
            references = tuple((reference, ) for reference in references)
        
        dependencies = _(references) \
            .iflatten() \
            .imap(lambda x: x.table) \
            .call(set).call(tuple)
        
        values = dict()
        for keys, probabilities in zip(references, probability_rows):
            assert_almost_sums_to_one(probabilities)
            for self_key, value in zip(labels, probabilities):
                values[keys + (self_key, )] = value
        
        cross_product_of_dependencies_keys = _(dependencies) \
            .imap(attrgetter('_labels')) \
            .star_call(itertools.product) \
            .imap(frozenset) \
            .call(set)
        
        assert _(references).map(frozenset).call(set) == cross_product_of_dependencies_keys, \
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
        
        assert _(values.values()).map(lambda x: isinstance(x, float)).all(), 'Need all probabilities to be floats'
        self._values = { self._normalize_keys(key): value for key, value in values.items() }
    
    def _set_references(self, labels):
        self._labels = _(labels).map(lambda key: Reference(key, self))
        for reference in self._labels:
            setattr(self, reference.name, reference)
    
    # REFACT consider to ignore all keys which do not apply
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
        assert len(tuple(self._values.keys())[0]) == len(keys), 'Need the full set of keys to get a probability'
        assert any(filter(lambda x: x.table == self, keys)), 'Does not contain key to self'
    
    def __repr__(self):
        display_values = ', '.join(['%r: %s' % (set(key), value) for key, value in self._values.items()])
        name = self._name if self._name is not None else 'Distribution'
        return '%s(%s)' % (name, display_values)
    __str__ = __repr__
    
    def _suitable_subset_of(self, keys):
        return filter(lambda key: key.table == self or key.table in self._dependencies, keys)

class BayesianNetwork(object):
    
    def __init__(self):
        for name, table in self._tables().items():
            table._network = self
            table._name = name
    
    def _tables(self):
        # would be a desaster if Distribution's are added after construction - but that is currently
        # prevented by design
        if not hasattr(self, '__tables'):
            self.__tables = dict()
            for name, table in vars(self.__class__).items():
                if len(name) == 1 or name[0] == '_': continue # shortname, or private
                self.__tables[name] = table
        return self.__tables
    
    def probability_of_event(self, *atomic_event):
        probability = 1
        # REFACT rename table -> distributions
        for table in self._tables().values():
            probability *= table[table._suitable_subset_of(keys=atomic_event)]
        return probability
    
    # REFACT not sure this is the right name for this?
    def joint_probability(self, *givens): # REFACT rename events -> givens
        probability = 0
        by_table = self._events_by_table(self._sure_event()) # REFACT rename _sure_event -> _all_events
        for event in givens:
            by_table[event.table] = [event]
        # dict(intelligence = [intelligence.low], difficulty=$alle, ...)
        for atomic_event in itertools.product(*by_table.values()):
            probability += self.probability_of_event(*atomic_event)
        return probability
    
    def conditional_probability(self, *events, given):
        return self.joint_probability(*events, *given) / self.joint_probability(*given)
    
    def _sure_event(self):
        return set(flatten(map(attrgetter('_labels'), n._tables().values())))
    
    def _events_by_table(self, events):
        grouped_iterator = _(events) \
            .isorted(key=lambda reference: id(reference.table)) \
            .groupby(key=lambda x: x.table)
        by_table = dict()
        for table, events in grouped_iterator:
            by_table[table] = events
        return by_table

class Student(BayesianNetwork):
    d = difficulty = Distribution.independent(easy=.6, hard=.4)
    i = intelligence = Distribution.independent(low=.7, high=.3)
    
    s = sat = Distribution.dependent(
                ('bad', 'good'), {
        i.low:  (.95,   .05),
        i.high: (.2,    .8)
    })
    g = grade = Distribution.dependent(
                          ('good', 'ok', 'bad'), {
        (i.low, d.easy):   (.3,     .4,   .3),
        (i.low, d.hard):   (.05,    .25,  .7),
        (i.high, d.easy):  (.9,     .08,  .02),
        (i.high, d.hard):  (.5,     .3,   .2),
    })
    l = letter = Distribution.dependent(
                ('bad', 'glowing'), {
        g.good: (.1,    .9),
        g.ok:   (.4,    .6),
        g.bad:  (.99,   .01),
    })

n = network = Student()

# print(n.i)
# print(n.i.low)
# print(n.s)

expect(n.intelligence[n.i.high]) == .3
expect(n.difficulty[n.d.easy]) == .6
expect(n.grade[n.g.ok, n.i.high, n.d.easy]) == .08, 
expect(n.letter[n.l.bad, n.g.ok]) == .4

# print(n.intelligence.low, n.i[n.i.low])
# print(n.difficulty.easy, n.d[n.d.easy])
#
# print(n.sat.bad, n.intelligence.low, n.sat[n.s.bad, n.i.low])
#
print(n.intelligence.low, n.difficulty.easy, n.grade.good, n.g[n.i.low, n.d.easy, n.g.good])
print(n.intelligence.low, n.difficulty.easy, n.grade.good, n.g[n.i.low, n.g.good, n.d.easy])
#
# print(n.letter.bad, n.grade.good, n.l[n.l.bad, n.g.good])

expect(n.probability_of_event(n.i.high, n.d.easy, n.g.ok, n.l.bad, n.s.good)) == 0.004608
expect(n.joint_probability()).close_to(1, 1e-6)
expect(n.joint_probability(n.l.glowing)).close_to(.502, 1e-3)

expect(n.conditional_probability(n.l.glowing, given=(n.i.low,))).close_to(.38, 1e-2)
expect(n.conditional_probability(n.l.glowing, given=(n.i.low, n.d.easy))).close_to(.513, 1e-2)
expect(n.conditional_probability(n.i.high, given=(n.g.good,))).close_to(.613, 1e-2)
expect(n.conditional_probability(n.i.high, given=(n.g.good, n.d.easy))).close_to(.5625, 1e-4)

# print('P(d0 | g1)', conditional_probability('difficulties', ['d0'], grades=['g1']))
# P(d0 | g1) 0.7955801104972375
# print('P(d0 | g1, i1)', conditional_probability('difficulties', ['d0'], grades=['g1'], intelligences=['i1']))
# P(d0 | g1, i1) 0.7297297297297298
#
#
# print('P(i1 | g3)', conditional_probability('intelligences', ['i1'], grades=['g3']))
# P(i1 | g3) 0.07894736842105264
# print('P(i1 | g3, d1)', conditional_probability('intelligences', ['i1'], grades=['g3'], difficulties=['d1']))
# P(i1 | g3, d1) 0.10909090909090914
# print('P(d1 | g3)', conditional_probability('difficulties', ['d1'], grades=['g3']))
# P(d1 | g3) 0.6292906178489701
# print('P(d1 | g3, i1)', conditional_probability('difficulties', ['d1'], grades=['g3'], intelligences=['i1']))
# P(d1 | g3, i1) 0.8695652173913044
