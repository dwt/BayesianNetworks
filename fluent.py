#!/usr/bin/env python3
# encoding: utf8
# license: ISC (MIT/BSD compatible) https://choosealicense.com/licenses/isc/

# This library is principally created for python 3. However python 2 support may be doable and is welcomed.

"""Use python in a more object oriented, saner and shorter way.

First: A word of warning. This library is an experiment. It is based on a wrapper that aggressively 
wraps anything it comes in contact with and tries to stay invisible from then on (apart from adding methods).
However this means that this library is probably quite unsuitable for use in bigger projects. Why? 
Because the wrapper will spread in your runtime image like a virus, 'infecting' more and more objects 
causing strange side effects. That being said, this library is perfect for short scripts and especially 
'one of' shell commands. Use it's power wisely!

This library is heavily inspired by jQuery and underscore / lodash in the javascript world. Or you 
could say that it is inspired by SmallTalk and in extension Ruby and how they deal with collections 
and how to work with them.

In JS the problem is that the standard library sucks very badly and is missing many of the 
most important convenience methods. Python is better in this regard, in that it has (almost) all 
those methods available somewhere. BUT: quite a lot of them are available on the wrong object or 
are free methods where they really should be methods. Examples: `str.join` really should be on iterable.
`map`, `zip`, `filter` should really be on iterable. Part of this problem comes from the design 
choice of the python language to provide a strange kind of minimal duck typing interface with the __*__ 
methods that the free methods like `map`, `zip`, `filter` then use. This however has the unfortunate
side effect in that writing python code using these methods often requires the reader to mentally skip 
back and forth in a line to parse what it does. While this is not too bad for simple usage of these 
functions, it becomes a nightmare if longer statements are built up from them.

Don't believe me? Try to parse this simple example as fast as you can:

>>> map(print, map(str.upper, sys.stdin.read().split('\n')))

How many backtrackings did you have to do? To me this code means, finding out that it starts in the 
middle at `sys.stdin.read().split('\n')`, then I have to backtrack to `map(str.upper, …)`, then to 
`map(print, …)`. Then while writing, I have to make sure that the number of parens at the end are 
correct, which is something I usually have to use editor support for as it's quite hard to accurately 
identify where the matching paren is.

The problem with this? This is hard! Hard to write, as it doesn't follow the way I think about this 
statement. Literally, this means I usually write these statements from the inside out and wrap them
using my editor as I write them. As demonstrated above, it's also hard to read - requireing quite a 
bit of backtracking.

So, what's the problem you say? Just don't do it, it's not pythonic you say! Well, Python has two 
main workarounds available for this mess. One is to use list comprehension / generator 
statements like this:

>>> [print(line.upper()) for line in sys.stdin.read().split('\n')]

This is clearly better. Now you only have to skip back and forth once instead of twice Yay! Win! 
To me that is not a good workaround. Sure it's nice to easily be able to create generators this 
way, but it still requires of me to find where the statement starts and to backtrack to the beginning 
to see what is happening. Oh, but they support filtering too!

>>> [print(line.upper()) for line in sys.stdin.read().split('\n') if line.upper().startswith('FNORD')]

Well, this is little better. For one thing, this doesn't solve the backtracking problem, but more 
importantly, if the filtering has to be done on the processed version (here artificially on 
`line.upper().startswith()`) then the operation has to be applied twice - which sucks because you have to write it twice, but also because it is computed twice.

The solution? Nest them!

[print(line) for line in (line.upper() for line in sys.stdin.read().split('\n')) if line.startswith('FNORD')]

Do you start seing the problem?

Compare it to this:

>>> for line in sys.stdin.read().split('\n'):
>>>     uppercased = line.upper()
>>>     if uppercased.startswith('FNORD'):
>>>         print(uppercased)

Almost all my complaints are gone. It reads and writes almost completely in order it is computed.
Easy to read, easy to write - but one drawback. It's not an expression - it's a statement.
Which means that it's not easily combinable and abstractable with higher order methods. Also 
(to complain on a high level), you had to invent two variable names `line` and `uppercased`. 
While that is not bad - it's also not really helping _and_ (drummroll) it requires some 
backtracking to read. Oh well.

Wouldn't it be cool if you could write this instead:

>>> sys.stdin.read().split('\n').map(str.upper).filter(_.each.startswith('FNORD')).map(print)

All my complaints are gone. It's an expression (yay!). It reads and writes exactly in the order 
stuff is computed, so it matches exactly the way I think. It doesn't require additional variables 
for stuff that is not essential.

There is one other way to solve this problem in python and that is using intermediate variables.

Consider this code:

>>> cross_product_of_dependency_labels = \
>>>     set(map(frozenset, itertools.product(*map(attrgetter('_labels'), dependencies))))

That certainly is hard to read (and write). Pulling out explaining variables, makes it better. Like so:

>>> labels = map(attrgetter('_labels'), dependencies)
>>> cross_product_of_dependency_labels = set(map(frozenset, itertools.product(*labels)))

Better, but still hard to read. Sure, those explaining variables are nice and sometimes 
essential to understand the code. - but it does take up space in lines, and space in my head 
while parsing this code. The question would be - is this really easier to read than something 
like this?

>>> cross_product_of_dependency_labels = _(dependencies) \
>>>     .map(attrgetter('_labels')) \
>>>     .star_call(itertools.product) \
>>>     .map(frozenset) \
>>>     .call(set)

Sure you are not used to this at first, but consider the advantages. The intermediate variable 
names are abstracted away - the data flows through the methods completely naturally. No jumping 
back and forth to parse this at all. It just reads and writes exactly in the order it is computed.

So what is the essence of all of this?

Python is an object oriented language - but it doesn't really use what object orientation has tought 
us about how we can work with collections and higher order methods in the languages that came before it
(especially SmallTalk, but more recently Ruby). Why can't I make those beautiful fluent call chains 
in Python?

Well, now you can.

While I know that this is not something you want to use in big projects (see warning at the beginning) 
I envision this to be very usefull in quick python scripts and shell one liner filters.

Some examples:

$ curl -sL 'https://www.iblocklist.com/lists.php' | egrep -A1 'star_[345]' | python3 -c "import sys, re; from xml.sax.saxutils import unescape; print('\n'.join(map(unescape, re.findall(r'value=\'(.*)\'', sys.stdin.read()))))"

And compare it to this:

$ curl -sL 'https://www.iblocklist.com/lists.php' | egrep -A1 'star_[345]' | python3 -m fluent "lib.sys.stdin.read().findall(r'value=\'(.*)\'').map(lib.xml.sax.saxutils.unescape).map(print)"

Which do you think is easier to read or write?

To enable this style of coding having to import every symbol used is quite counter productive as this 
always requires a separate statement in python. To shorten this, this library also contains the `lib` 
object, which is a wrapper around the python import machinery and allows to import 
anything that is accessible by import to be imported as an expression for inline use.

So instead of

>>> import sys
>>> input = sys.stdin.read()

You can do

>>> input = lib.sys.stdin.read()

As a bonus, everything imported via lib is already pre-wrapped, so you can chain off of it immediately.

`lib` is also available on `_` which is itself just an alias for `wrap`. This is usefull if you want 
to import fewer symbols from fluent or want to import the library undera custom name

>>> from fluent import _ # alias for wrap
>>> _.lib.sys.stdin.split('\n').map(str.upper).map(print)

>>> from fluent import _ as fluent # alias for wrap
>>> fluent.lib.sys.stdin.split('\n').map(str.upper).map(print)

>>> import fluent
>>> fluent.lib.sys.stdin.split('\n').map(str.upper).map(print)

This library tries to do a little of what underscore does for javascript. Just provide the missing glue to make the standard library nicer to use. Have fun!
"""

"""Future Ideas:

    _.attrgetter = _.lib.operator.attrgetter seems like a nice shortcut and would be pre-wrapped
just using lib.operator.attrgetter is also not so bad and requires nothing special
    _([1,2,3]).map(lambda x: (x['foo'], x['bar']))
    _([1,2,3]).map(_.attrgetter('foo', 'bar')) # this seems to be a nice shortcut
    _([1,2,3]).map(lib.operator.attrgetter('foo', 'bar'))
    _([1,2,3]).map(_.each.foo)
    _([1,2,3]).map(_.itemgetter('foo', 'bar')) # this seems to be a nice shortcut
    _([1,2,3]).map(lib.operator.itemgetter('foo', 'bar')) # return tuples of items
    _([1,2,3]).map(_.each['foo', 'bar'])
    _([1,2,3]).map(lambda x: x.method_name(arg1, kwarg='value')) # just do a lambda...
    _([1,2,3]).map(_.methodcaller('method_name', arg1, kwarg='value')) # this seems to be a nice shortcut
    _([1,2,3]).map(lib.operator.methodcaller('method_name', arg1, kwarg='value'))
    _([1,2,3]).map(_.each.method_name(arg1, kwarg='value'))


    _([1,2,3]).map(_.each + 3)

I cold also put all the operator methods on `wrap` to make them easily accessible?
would make them easily available to curry too
Could provide some auto_curry feature where
    `_ % 3` creates a function that modulos it's argument, quite a nice shortcut...
    `3 % _` is this also possible?
    _(3).call(3 % _)
    _(3).call(_.mod(3, _))
    _(_, 'foo', bar='baz') could become a callable that accepts one argument?
    _(a_function)(_, 'foo', bar='baz') could become a callable that accepts one argument?
    _(_, 'foo', _, _, bar='baz') could become a callable that accepts three arguments? (plus any kwargs of course)
Consider if auto-currying for all methods here would be a cool idea?
one or multiple placeholders?

    _.something('foo') 
becomes an auto curried method that has a placeholder as the first argument (usually self)
That would mean an easy version to write most lambdas. Also _ could be used as a placeholder at a specific position.
Maybe have this on it's own sub-symbol?
    _.curry.something('foo', _, 'bar')
    _.each.something('foo', _, 'bar') == lambda x: x.something('foo', _, 'bar')
    lib.sys.stdin.read().split('\n').filter(_.startswith('fnord')).map(print)
    lib.sys.stdin.read().split('\n').filter(_.curry.startswith('fnord')).map(print)
    lib.sys.stdin.read().split('\n').filter(_.each.startswith('fnord')).map(print)
    lib.sys.stdin.read().split('\n').filter(_.methodcaller('startswith', 'fnord')).map(print)

Support SmallTalk style return value handling. I.e. if a method returns None, wrapper could act as if it had returned 'self' to allow further chaining.
"""

# REFACT rename wrap -> fluent? perhaps as an alias?
__all__ = [
    'wrap', # generic wrapper factory that returns the appropriate subclass in this package according to what is wrapped
    '_', # _ is an alias for wrap
    'lib', # wrapper for python import machinery, access every importable package / function directly on this via attribute access
]

import typing
import re
import math
import types
import functools
import itertools
import operator
import collections.abc

CollectionType = collections.abc.Container
if hasattr(typing, 'Collection'): # strangely not available on ipad
    CollectionType = typing.Collection

TextType = str
if hasattr(typing, 'Text'): # strangely not available on ipad
    TextType = typing.Text


def wrap(wrapped, *, previous=None):
    """Factory method, wraps anything and returns the appropriate Wrapper subclass.
    
    This is the main entry point into the fluent wonderland. Wrap something and 
    everything you call off of that will stay wrapped in the apropriate wrappers.
    """
    by_type = (
        (types.ModuleType, Module),
        (TextType, Text),
        (typing.Mapping, Mapping),
        (typing.AbstractSet, Set),
        (typing.Iterable, Iterable),
        (typing.Callable, Callable),
    )
    for clazz, wrapper in by_type:
        if isinstance(wrapped, clazz):
            return wrapper(wrapped, previous=previous)
    
    return Wrapper(wrapped, previous=previous)

# sadly _ is pretty much the only valid python identifier that is sombolic and easy to type. Unicode would also be a candidate, but hard to type $, § like in js cannot be used
_ = wrap

# using these decorators will take care of unwrapping and rewrapping the target object.
# thus all following code is written as if the methods live on the wrapped object
def wrapped(wrapped_function, wrap_result=None):
    @functools.wraps(wrapped_function)
    def wrapper(self, *args, **kwargs):
        result = wrapped_function(self.chain, *args, **kwargs)
        if callable(wrap_result):
            result = wrap_result(result)
        return wrap(result, previous=self)
    return wrapper

def unwrapped(wrapped_function):
    @functools.wraps(wrapped_function)
    def forwarder(self, *args, **kwargs):
        return wrapped_function(self.chain, *args, **kwargs)
    return forwarder

def wrapped_forward(wrapped_function, wrap_result=None):
    """Forwards a call to a different object
    
    This makes its method available on the wrapper.
    This specifically models the case where the method forwarded to, 
    takes the current object as its first argument.
    
    This also deals nicely with the fact that the method is just on the wrong object.
    """
    @functools.wraps(wrapped_function)
    def wrapper(self, *args, **kwargs):
        result = wrapped_function(args[0], self.chain, *args[1:], **kwargs)
        if callable(wrap_result):
            result = wrap_result(result)
        return wrap(result, previous=self)
    return wrapper

def tupleize(wrapped_function):
    "Wrap the returned obect in a tuple to force execution of iterators"
    @functools.wraps(wrapped_function)
    def wrapper(self, *args, **kwargs):
        return wrap(tuple(wrapped_function(self, *args, **kwargs)), previous=self)
    return wrapper

class Wrapper(object):
    """Universal wrapper.
    
    This class ensures that all function calls and attribute accesses 
    that can be caught in python will be wrapped with the wrapper again.
    
    This ensures that the fluent interface will persist and everything 
    that is returned is itself able to be chaned from again.
    
    Using this wrapper changes the behaviour of python soure code in quite a big way.
    
    a) If you wrap something, if you want to get at the real object from any 
       function call or attribute access off of that object, you will have to 
       explicitly unwrap it.
    
    b) All returned objects will be enhanced by behaviour that matches the 
       wrapped type. I.e. iterables will gain the collection interface, 
       mappings will gain the mapping interface, strings will gain the 
       string interface, etc.
    
    c) Operators like ==, +, *, will just proxy to the wrapped object and 
       return an unwrapped object. If you want to chain from these, use the
       chainable alternatives (.eq, .mul, .add, …)
    """
    
    def __init__(self, wrapped, previous):
        assert wrapped is not None or previous is not None, 'Cannot chain off of None'
        self.__wrapped = wrapped
        self.__previous = previous
    
    # Proxied methods
    
    __getattr__ = wrapped(getattr)
    
    __str__ = unwrapped(str)
    __repr__ = unwrapped(repr)
    
    __eq__ = unwrapped(operator.eq)
    
    # TODO add all remaining __$__ methods - consider if this can be done generically?
    # Problem is that python doesn't access __ attributes via __getattr__. :-/
    
    # Breakouts
    
    @property
    def unwrap(self):
        return self.__wrapped
    _ = unwrap # alias
    
    @property
    def previous(self):
        return self.__previous
    
    @property
    def chain(self):
        if self.unwrap is None:
            return self.previous.unwrap
        return self.unwrap
    
    # Chainable versions of operators
    
    @wrapped
    def call(self, function, *args, **kwargs):
        "Call function with self as first argument"
        # Different from __call__! Calls function(self, …) instead of self(…)
        return function(self, *args, **kwargs)
    
    # REFACT eq - keep or toss?
    eq = wrapped(operator.eq)
    
    setattr = wrapped(setattr)
    getattr = wrapped(getattr)
    hasattr = wrapped(hasattr)
    
    isinstance = wrapped(isinstance)
    issubclass = wrapped(issubclass)
    
    @wrapped
    def tee(self, function):
        function(wrap(self)) # REFACT consider to hand in unwrapped object here to make it easier to work with stdlib methods here, might be unneccessary if wrapper is complete
        return self
    
    # TODO vars, dir

# REFACT consider to use wrap as the placeholder to have less symbols? Probably not worth it...
virtual_root_module = object()
class Module(Wrapper):
    """Importer shortcut.
    
    All attribute accesses to instances of this class are converted to
    an import statement, but as an expression that returns the wrapped imported object.
    
    Example:
    
    >>> lib.sys.stdin.read().map(print)
    
    Is equivalent to
    
    >>> import importlib
    >>> wrap(importlib.import_module('sys').stdin).read().map(print)
    
    But of course without creating the intermediate symbol 'stdin' in the current namespace.
    # TODO replace example with call to importlib implementation
    
    All objects returned from lib are pre-wrapped, so you can chain off of them immediately.
    """
    
    def __getattr__(self, name):
        if hasattr(self.chain, name):
            return wrap(getattr(self.chain, name))
        
        import importlib
        module = None
        if self.chain is virtual_root_module:
            module = importlib.import_module(name)
        else:
            module = importlib.import_module('.'.join((self.chain.__name__, name)))
        
        return wrap(module)

wrap.lib = lib = Module(virtual_root_module, previous=None)

class Callable(Wrapper):
    
    @wrapped
    def __call__(self, *args, **kwargs):
        return self(*args, **kwargs)
    
    # REFACT rename to partial for consistency with stdlib?
    # REFACT consider if there could be more utility in supporting placeholders for more usecases.
    # examples:
    #   Switching argument order?
    #   multiple placeholders?
    @wrapped
    def curry(self, *args, **kwargs):
        """"Like functools.partial, but with a twist.
        
        If you use wrap (best imported as _) as a positional argument,
        upon the actual call, arguments will be left-filled for those placeholders.
        For example:
        
        >>> from fluent import wrap as _
        >>> _(operator.add).curry(_, 'foo')('bar') == 'barfoo'
        """
        # would be so nice to just use functools.partial(self, *args, **kwargs)
        # but they don't support placeholders
        placeholder = wrap
        def merge_args(curried_args, fargs):
            new_args = list(curried_args)
            if placeholder in curried_args:
                assert curried_args.count(placeholder) == len(fargs), \
                    'Need the right ammount of arguments for the placeholders'
                index = 0
                for arg in fargs:
                    index = new_args.index(placeholder, index)
                    new_args[index] = arg
            return new_args
        @functools.wraps(self)
        def wrapper(*fargs, **fkeywords):
            # TODO do I want the curried arguments to overwrite the 
            # direct ones or should they define defaults?
            # Currently they define defaults. This feels similar to how python handles
            #  keyword arguments in function definitions, but i wonder if the other 
            # way around would be more usefull here?
            new_kwargs = dict(kwargs, **fkeywords)
            new_args = merge_args(args, fargs)
            return self(*new_args, **new_kwargs)
        return wrapper
    
    @wrapped
    def compose(self, outer):
        return lambda *args, **kwargs: outer(self(*args, **kwargs))
    # REFACT consider aliasses wrap = chain = cast = compose
    # Also not so sure if this is really so helpfull as the i* versions of the iterators 
    # basically allow the same thing

class Iterable(Wrapper):
    """Add iterator methods to any iterable.
    
    Most iterators in python3 return an iterator by default, which is very interesting 
    if you want to build efficient processing pipelines, but not so hot for quick and 
    dirty scripts where you have to wrap the result in a list() or tuple() all the time 
    to actually get at the results (e.g. to print them) or to actually trigger the 
    computation pipeline.
    
    Thus all iterators on this class are by default immediate, i.e. they don't return the 
    iterator but instead consume it immediately and return a tuple. Of course if needed, 
    there is also an i{map,zip,enumerate,...} version for your enjoyment.
    """
    
    __iter__ = unwrapped(iter)
    
    @wrapped
    def star_call(self, function, *args, **kwargs):
        "Call function with *self as first argument"
        return function(*args, *self, **kwargs)
    
    # This looks like it should be the same as 
    # starcall = wrapped(lambda function, wrapped, *args, **kwargs: function(*wrapped, *args, **kwargs))
    # but it's not. Why?
    
    @wrapped
    def join(self, with_what):
        "Like str.join, but the other way around. Bohoo!"
        return with_what.join(map(str, self))
    
    ## Reductors .........................................
    
    len = wrapped(len)
    max = wrapped(max)
    min = wrapped(min)
    sum = wrapped(sum)
    any = wrapped(any)
    all = wrapped(all)
    reduce = wrapped_forward(functools.reduce)
    
    ## Iterators .........................................
    
    imap = wrapped_forward(map)
    map = tupleize(imap)
    
    istarmap = wrapped_forward(itertools.starmap)
    starmap = tupleize(istarmap)
    
    ifilter = wrapped_forward(filter)
    filter = tupleize(ifilter)
    
    ienumerate = wrapped(enumerate)
    enumerate = tupleize(ienumerate)
    
    ireversed = wrapped(reversed)
    reversed = tupleize(ireversed)
    
    isorted = wrapped(sorted)
    sorted = tupleize(isorted)
    
    @wrapped
    def igrouped(self, group_length):
        "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
        return zip(*[iter(self)]*group_length)
    grouped = tupleize(igrouped)
    
    @wrapped
    def izip(self, *args):
        return zip(self, *args)
    zip = tupleize(izip)
    
    @wrapped
    def iflatten(self, level=math.inf):
        "Modeled after rubys array.flatten @see http://ruby-doc.org/core-1.9.3/Array.html#method-i-flatten"
        for element in self:
            if level > 0 and isinstance(element, typing.Iterable):
                for subelement in _(element).iflatten(level=level-1):
                    yield subelement
            else:
                yield element
        return
    flatten = tupleize(iflatten)
    
    igroupby = wrapped(itertools.groupby)
    def groupby(self, *args, **kwargs):
        # Need an extra wrapping function to consume the deep iterators in time
        result = []
        for key, values in self.igroupby(*args, **kwargs):
            result.append((key, tuple(values)))
        return wrap(tuple(result))
    
    @wrapped
    def tee(self, function):
        "This override tries to retain iterators, as a speedup"
        if not isinstance(self, CollectionType): # probably a consuming iterator
            first, second = itertools.tee(self, 2)
            function(wrap(first)) # REFACT here too, do I want to hand out unwrapped objects for better interoperability?
            return second
        else: # can't call super from here, as self is not the real self
            function(wrap(self))
            return self

class Mapping(Iterable):
    
    # REFACT rename: kwargs_call?
    @wrapped
    def splat_call(self, function, *args, **kwargs):
        "Calls function(**self), but allows to override kwargs"
        # REFACT curry sets defaults - maybe I want this here too for consistency?
        return function(*args, **dict(self, **kwargs))

class Set(Iterable): pass

# REFACT consider to inherit from Iterable?
class Text(Wrapper):
    
    # Regex Methods ......................................
    
    findall = wrapped_forward(re.findall)
    
    # finditer
    # fullmatch
    # match
    # search
    
    split = wrapped_forward(re.split)
    
    # sub, subn
    

# Roundable (for all numeric needs?)
    # round, times, repeat, if, else
"""commented
not sure this is very usefull, can just .call('foo'.format) stuff
# list.format = lambda self, format_string: str(format_string.format(*self))
#
#
# str.findall = lambda self, pattern: list(re.findall(pattern, self))
# # str.split = lambda self, *args, **kwargs: list(__str.split(self, *args, **kwargs))
# # str.split = list.cast(__str.split)
# # str.split = str.split.cast(list)
# # str.split = str.split.chain(list)
# # str.split = func.compose(str.split, list)
# str.split = func.wrap(str.split, list)
# str.upper = lambda self: str(__str.upper(self))
# str.prepend = lambda self, other: str(other + self)
# str.format = lambda self, format_string: str(format_string.format(self))

# REFACT accept regex as first argument and route to re.split then instead

# REFACT add imp auto importer, that pre-wraps everything he imports. End effect should be that python is seamlessly usable like this.
# REFACT add python -m fluent 'code…' support which auto injects module importer and 
# TODO add flatten to listlikes
# TODO add sort, groupby, grouped
# TODO add convenience keyword arguments to map etc.
# map(attr='attrname') as shortcut for map(attrgetter('attrname'))
# map(item='itemname') as shortcut for map(itemgetter('itemname'))
# TODO consider numeric type to do stuff like wrap(3).times(...)
    or wrap([1,2,3]).call(len).times(yank_me)
"""

import unittest
from pyexpect import expect
import pytest

class FluentTest(unittest.TestCase): pass

class WrapperTest(FluentTest):
    
    def test_should_wrap_callables(self):
        counter = [0]
        def foo(): counter[0] += 1
        expect(_(foo)).is_instance(Wrapper)
        _(foo)()
        expect(counter[0]) == 1
    
    def test_should_wrap_attribute_accesses(self):
        class Foo(): bar = 'baz'
        expect(wrap(Foo()).bar).is_instance(Wrapper)
    
    def test_should_error_when_accessing_missing_attribute(self):
        class Foo(): pass
        expect(lambda: wrap(Foo().missing)).to_raise(AttributeError)
    
    def test_should_explictly_unwrap(self):
        foo = 1
        expect(wrap(foo).unwrap).is_(foo)
    
    def test_should_wrap_according_to_returned_type(self):
        expect(wrap('foo')).is_instance(Text)
        expect(wrap([])).is_instance(Iterable)
        expect(wrap({})).is_instance(Mapping)
        expect(wrap({1})).is_instance(Set)
        
        expect(wrap(lambda: None)).is_instance(Callable)
        class CallMe(object):
            def __call__(self): pass
        expect(wrap(CallMe())).is_instance(Callable)
        
        expect(wrap(object())).is_instance(Wrapper)
    
    def test_should_remember_call_chain(self):
        def foo(): return 'bar'
        expect(wrap(foo)().unwrap) == 'bar'
        expect(wrap(foo)().previous.unwrap) == foo
    
    def test_should_delegate_equality_test_to_wrapped_instance(self):
        expect(wrap(1)) == 1
        expect(wrap('42')) == '42'
        callme = lambda: None
        expect(wrap(callme)) == callme
    
    def test_hasattr_getattr_setattr(self):
        expect(wrap((1,2)).hasattr('len'))
        expect(wrap('foo').getattr('__len__')()) == 3
        class Attr(object):
            foo = 'bar'
        expect(wrap(Attr()).setattr('foo', 'baz').foo) == 'baz'
    
    def test_isinstance_issubclass(self):
        expect(wrap('foo').isinstance(str)) == True
        expect(wrap('foo').isinstance(int)) == False
        expect(wrap(str).issubclass(object)) == True
        expect(wrap(str).issubclass(str)) == True
        expect(wrap(str).issubclass(int)) == False

class CallableTest(FluentTest):
    
    def test_star_call(self):
        expect(wrap([1,2,3]).star_call(str.format, '{} - {} : {}')) == '1 - 2 : 3'
    
    def test_should_call_callable_with_wrapped_as_first_argument(self):
        expect(wrap([1,2,3]).call(min)) == 1
        expect(wrap([1,2,3]).call(min)) == 1
        expect(wrap('foo').call(str.upper)) == 'FOO'
        expect(wrap('foo').call(str.upper)) == 'FOO'
    
    def test_tee_breakout_a_function_with_side_effects_and_disregard_return_value(self):
        side_effect = {}
        def tee(a_list): side_effect['tee'] = a_list.join('-')
        expect(wrap([1,2,3]).tee(tee)) == [1,2,3]
        expect(side_effect['tee']) == '1-2-3'
        
        def fnording(ignored): return 'fnord'
        expect(wrap([1,2,3]).tee(fnording)) == [1,2,3]
    
    def _test_tee_should_work_fine_with_functions_not_expecting_a_wrapper(self):
        pass
    
    def test_curry(self):
        expect(wrap(lambda x, y: x*y).curry(2, 3)()) == 6
        expect(wrap(lambda x=1, y=2: x*y).curry(x=3)()) == 6
    
    def test_curry_should_support_placeholders_to_curry_later_positional_arguments(self):
        expect(_(operator.add).curry(_, 'foo')('bar')) == 'barfoo'
        expect(_(lambda x, y, z: x + y + z).curry(_, 'baz', _)('foo', 'bar')) == 'foobazbar'
        # expect(_(operator.add).curry(_2, _1)('foo', 'bar')) == 'barfoo'
    
    def test_compose_cast_wraps_chain(self):
        expect(_(lambda x: x*2).compose(lambda x: x+3)(5)) == 13
        expect(_(str.strip).compose(str.capitalize)('  fnord  ')) == 'Fnord'

class IterableTest(FluentTest):
    
    def test_should_call_callable_with_star_splat_of_self(self):
        expect(wrap([1,2,3]).star_call(lambda x, y, z: z-x-y)) == 0
    
    def test_join(self):
        expect(wrap(['1','2','3']).join(' ')) == '1 2 3'
        expect(wrap([1,2,3]).join(' ')) == '1 2 3'
    
    def test_any(self):
        expect(wrap((True, False)).any()) == True
        expect(wrap((False, False)).any()) == False
    
    def test_all(self):
        expect(wrap((True, False)).all()) == False
        expect(wrap((True, True)).all()) == True
    
    def test_len(self):
        expect(wrap((1,2,3)).len()) == 3
    
    def test_min_max_sum(self):
        expect(wrap([1,2]).min()) == 1
        expect(wrap([1,2]).max()) == 2
        expect(wrap((1,2,3)).sum()) == 6
    
    def test_map(self):
        expect(wrap([1,2,3]).imap(lambda x: x * x).call(list)) == [1, 4, 9]
        expect(wrap([1,2,3]).map(lambda x: x * x)) == (1, 4, 9)
    
    def test_starmap(self):
        expect(wrap([(1,2), (3,4)]).istarmap(lambda x, y: x+y).call(list)) == [3, 7]
        expect(wrap([(1,2), (3,4)]).starmap(lambda x, y: x+y)) == (3, 7)
    
    def test_filter(self):
        expect(wrap([1,2,3]).ifilter(lambda x: x > 1).call(list)) == [2,3]
        expect(wrap([1,2,3]).filter(lambda x: x > 1)) == (2,3)
    
    def test_zip(self):
        expect(wrap((1,2)).izip((3,4)).call(tuple)) == ((1, 3), (2, 4))
        expect(wrap((1,2)).izip((3,4), (5,6)).call(tuple)) == ((1, 3, 5), (2, 4, 6))
        
        expect(wrap((1,2)).zip((3,4))) == ((1, 3), (2, 4))
        expect(wrap((1,2)).zip((3,4), (5,6))) == ((1, 3, 5), (2, 4, 6))
    
    def test_reduce(self):
        # no iterator version of reduce as it's not a mapping
        expect(wrap((1,2)).reduce(operator.add)) == 3
    
    def test_grouped(self):
        expect(wrap((1,2,3,4,5,6)).igrouped(2).call(list)) == [(1,2), (3,4), (5,6)]
        expect(wrap((1,2,3,4,5,6)).grouped(2)) == ((1,2), (3,4), (5,6))
        expect(wrap((1,2,3,4,5)).grouped(2)) == ((1,2), (3,4))
    
    def test_group_by(self):
        actual = {}
        for key, values in _((1,1,2,2,3,3)).igroupby():
            actual[key] = tuple(values)
        
        expect(actual) == {
            1: (1,1),
            2: (2,2),
            3: (3,3)
        }
        
        actual = {}
        for key, values in _((1,1,2,2,3,3)).groupby():
            actual[key] = tuple(values)
        
        expect(actual) == {
            1: (1,1),
            2: (2,2),
            3: (3,3)
        }
    
    def test_tee_should_not_break_iterators(self):
        recorder = []
        def record(generator): recorder.extend(generator)
        expect(wrap([1,2,3]).imap(lambda x: x*x).tee(record).call(list)) == [1,4,9]
        expect(recorder) == [1,4,9]
    
    def test_enumerate(self):
        expect(wrap(('foo', 'bar')).ienumerate().call(list)) == [(0, 'foo'), (1, 'bar')]
        expect(wrap(('foo', 'bar')).enumerate()) == ((0, 'foo'), (1, 'bar'))
    
    def test_reversed_sorted(self):
        expect(wrap([2,1,3]).ireversed().call(list)) == [3,1,2]
        expect(wrap([2,1,3]).reversed()) == (3,1,2)
        expect(wrap([2,1,3]).isorted().call(list)) == [1,2,3]
        expect(wrap([2,1,3]).sorted()) == (1,2,3)
        expect(wrap([2,1,3]).isorted(reverse=True).call(list)) == [3,2,1]
        expect(wrap([2,1,3]).sorted(reverse=True)) == (3,2,1)
    
    def test_flatten(self):
        expect(wrap([(1,2),[3,4],(5, [6,7])]).iflatten().call(list)) == \
            [1,2,3,4,5,6,7]
        expect(wrap([(1,2),[3,4],(5, [6,7])]).flatten()) == \
            (1,2,3,4,5,6,7)
        
        expect(wrap([(1,2),[3,4],(5, [6,7])]).flatten(level=1)) == \
            (1,2,3,4,5,[6,7])
    
    def _tee_should_work_fine_with_functions_that_dont_expect_wrappers(self):
        pass

class MappingTest(FluentTest):
    
    def test_should_call_callable_with_double_star_splat_as_keyword_arguments(self):
        def foo(*, foo): return foo
        expect(wrap(dict(foo='bar')).splat_call(foo)) == 'bar'
        expect(wrap(dict(foo='baz')).splat_call(foo, foo='bar')) == 'bar'

class StrTest(FluentTest):
    
    def test_findall(self):
        expect(wrap("bazfoobar").findall('ba[rz]')) == ['baz', 'bar']
    
    def test_split(self):
        expect(wrap('foo\nbar\nbaz').split(r'\n')) == ['foo', 'bar', 'baz']
        expect(wrap('foo\nbar/baz').split(r'[\n/]')) == ['foo', 'bar', 'baz']

class ImporterTest(FluentTest):
    
    def test_import_top_level_module(self):
        import sys
        expect(lib.sys) == sys
    
    def test_import_symbol_from_top_level_module(self):
        import sys
        expect(lib.sys.stdin) == sys.stdin
    
    def test_import_submodule_that_is_also_a_symbol_in_the_parent_module(self):
        import os
        expect(lib.os.name) == os.name
        expect(lib.os.path.join) == os.path.join
    
    def test_import_submodule_that_is_not_a_symbol_in_the_parent_module(self):
        import dbm
        expect(lambda: dbm.dumb).to_raise(AttributeError)
        
        def delayed_import():
            import dbm.dumb
            return dbm.dumb
        expect(lib.dbm.dumb) == delayed_import()
    
    def test_imported_objects_are_pre_wrapped(self):
        lib.os.path.join('/foo', 'bar', 'baz').findall(r'/(\w*)') == ['foo', 'bar', 'baz']

class IntegrationTest(FluentTest):
    
    
    def test_extrac_and_decode_URIs(self):
        from xml.sax.saxutils import unescape
        line = '''<td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
            <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='ydxerpxkpcfqjaybcssw' readonly='readonly' onClick="select_text('ydxerpxkpcfqjaybcssw');" value='http://list.iblocklist.com/?list=ydxerpxkpcfqjaybcssw&amp;fileformat=p2p&amp;archiveformat=gz'></td>'''

        actual = wrap(line).findall(r'value=\'(.*)\'').imap(unescape).call(list)
        expect(actual) == ['http://list.iblocklist.com/?list=ydxerpxkpcfqjaybcssw&fileformat=p2p&archiveformat=gz']
    
    def test_call_module_from_shell(self):
        from subprocess import check_output
        output = check_output(
            ['python', '-m', 'fluent', "_.lib.sys.stdin.read().split('\\n').imap(str.upper).imap(print).call(list)"],
            input=b'foo\nbar\nbaz')
        expect(output) == b'FOO\nBAR\nBAZ\n'


# REFACT remove or make an integration test
def _test():
    from xml.sax.saxutils import unescape
    
    """
    rico:~ dwt$ curl -sL 'https://www.iblocklist.com/lists.php' | egrep -A1 'star_[345]' | python -c "from __future__ import print_function; import sys, re; from from xml.sax.saxutils import unescape; print(map(unescape, re.findall(r'value=\'(.*)\'', sys.stdin.read())))"
    
    # CHECK wenn man das commando eh an -m fluent übergibt kann man auch das global objekt überschreiben und im getattr darin die imports dynamisch auflösen
    python -m fluent "str(sys.stdin).split('\n').map(xml.sax.unescape).map(print)"
    python -m fluent "[print(line) for line in [xml.sax.unescape(line) for line in sys.stdin.split('\n')]]
    """


    line = '''<td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='ydxerpxkpcfqjaybcssw' readonly='readonly' onClick="select_text('ydxerpxkpcfqjaybcssw');" value='http://list.iblocklist.com/?list=ydxerpxkpcfqjaybcssw&amp;fileformat=p2p&amp;archiveformat=gz'></td>'''

    str(line).findall(r'value=\'(.*)\'').map(unescape).map(print)
    str(line).findall(r'value=\'(.*)\'').map(unescape).join('\n').call(print)
    str(line).findall(r'value=\'(.*)\'').map(unescape).join('\n')(print)
    str(line).findall(r'value=\'(.*)\'').map(unescape).apply(print)
    str('lalala').upper().call(print)
    str('fnord').upper()(print)
    str('fnord').upper().prepend('Formatted: ')(print)
    str('fnord').upper().format('Formatted: {}')(print)
    list(['foo', 'bar', 'baz']).map(str.upper).tee(print).join(' ')(print)
    str('foo,bar,baz').split(',').map(print)
    # def to_curry(one, two, three):
    #     print(one, two, three)
    # functools.partial(to_curry, 1, 2, 3)()

    lines = '''<td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='ydxerpxkpcfqjaybcssw' readonly='readonly' onClick="select_text('ydxerpxkpcfqjaybcssw');" value='http://list.iblocklist.com/?list=ydxerpxkpcfqjaybcssw&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='gyisgnzbhppbvsphucsw' readonly='readonly' onClick="select_text('gyisgnzbhppbvsphucsw');" value='http://list.iblocklist.com/?list=gyisgnzbhppbvsphucsw&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_4.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='uwnukjqktoggdknzrhgh' readonly='readonly' onClick="select_text('uwnukjqktoggdknzrhgh');" value='http://list.iblocklist.com/?list=uwnukjqktoggdknzrhgh&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='imlmncgrkbnacgcwfjvh' readonly='readonly' onClick="select_text('imlmncgrkbnacgcwfjvh');" value='http://list.iblocklist.com/?list=imlmncgrkbnacgcwfjvh&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_3.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='plkehquoahljmyxjixpu' readonly='readonly' onClick="select_text('plkehquoahljmyxjixpu');" value='http://list.iblocklist.com/?list=plkehquoahljmyxjixpu&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='gihxqmhyunbxhbmgqrla' readonly='readonly' onClick="select_text('gihxqmhyunbxhbmgqrla');" value='http://list.iblocklist.com/?list=gihxqmhyunbxhbmgqrla&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='dgxtneitpuvgqqcpfulq' readonly='readonly' onClick="select_text('dgxtneitpuvgqqcpfulq');" value='http://list.iblocklist.com/?list=dgxtneitpuvgqqcpfulq&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='llvtlsjyoyiczbkjsxpf' readonly='readonly' onClick="select_text('llvtlsjyoyiczbkjsxpf');" value='http://list.iblocklist.com/?list=llvtlsjyoyiczbkjsxpf&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_4.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='xoebmbyexwuiogmbyprb' readonly='readonly' onClick="select_text('xoebmbyexwuiogmbyprb');" value='http://list.iblocklist.com/?list=xoebmbyexwuiogmbyprb&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_4.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='cwworuawihqvocglcoss' readonly='readonly' onClick="select_text('cwworuawihqvocglcoss');" value='http://list.iblocklist.com/?list=cwworuawihqvocglcoss&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='xshktygkujudfnjfioro' readonly='readonly' onClick="select_text('xshktygkujudfnjfioro');" value='http://list.iblocklist.com/?list=xshktygkujudfnjfioro&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='mcvxsnihddgutbjfbghy' readonly='readonly' onClick="select_text('mcvxsnihddgutbjfbghy');" value='http://list.iblocklist.com/?list=mcvxsnihddgutbjfbghy&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='usrcshglbiilevmyfhse' readonly='readonly' onClick="select_text('usrcshglbiilevmyfhse');" value='http://list.iblocklist.com/?list=usrcshglbiilevmyfhse&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_5.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='xpbqleszmajjesnzddhv' readonly='readonly' onClick="select_text('xpbqleszmajjesnzddhv');" value='http://list.iblocklist.com/?list=xpbqleszmajjesnzddhv&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_3.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='ficutxiwawokxlcyoeye' readonly='readonly' onClick="select_text('ficutxiwawokxlcyoeye');" value='http://list.iblocklist.com/?list=ficutxiwawokxlcyoeye&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_4.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='ghlzqtqxnzctvvajwwag' readonly='readonly' onClick="select_text('ghlzqtqxnzctvvajwwag');" value='http://list.iblocklist.com/?list=ghlzqtqxnzctvvajwwag&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_3.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='bcoepfyewziejvcqyhqo' readonly='readonly' onClick="select_text('bcoepfyewziejvcqyhqo');" value='http://list.iblocklist.com/?list=bcoepfyewziejvcqyhqo&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_3.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='cslpybexmxyuacbyuvib' readonly='readonly' onClick="select_text('cslpybexmxyuacbyuvib');" value='http://list.iblocklist.com/?list=cslpybexmxyuacbyuvib&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_4.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='pwqnlynprfgtjbgqoizj' readonly='readonly' onClick="select_text('pwqnlynprfgtjbgqoizj');" value='http://list.iblocklist.com/?list=pwqnlynprfgtjbgqoizj&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_3.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='jhaoawihmfxgnvmaqffp' readonly='readonly' onClick="select_text('jhaoawihmfxgnvmaqffp');" value='http://list.iblocklist.com/?list=jhaoawihmfxgnvmaqffp&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    --
    <td><img src='/sitefiles/star_3.png' height='15' width='75' alt=''></td>
    <td><input style='width:200px; outline:none; border-style:solid; border-width:1px; border-color:#ccc;' type='text' id='zbdlwrqkabxbcppvrnos' readonly='readonly' onClick="select_text('zbdlwrqkabxbcppvrnos');" value='http://list.iblocklist.com/?list=zbdlwrqkabxbcppvrnos&amp;fileformat=p2p&amp;archiveformat=gz'></td>
    '''

    # str(lines).findall(r'value=\'(.*)\'').map(unescape).apply(print)

    blocklists = [u'http://list.iblocklist.com/?list=ydxerpxkpcfqjaybcssw&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=gyisgnzbhppbvsphucsw&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=uwnukjqktoggdknzrhgh&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=imlmncgrkbnacgcwfjvh&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=plkehquoahljmyxjixpu&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=gihxqmhyunbxhbmgqrla&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=dgxtneitpuvgqqcpfulq&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=llvtlsjyoyiczbkjsxpf&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=xoebmbyexwuiogmbyprb&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=cwworuawihqvocglcoss&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=xshktygkujudfnjfioro&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=mcvxsnihddgutbjfbghy&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=usrcshglbiilevmyfhse&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=xpbqleszmajjesnzddhv&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=ficutxiwawokxlcyoeye&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=ghlzqtqxnzctvvajwwag&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=bcoepfyewziejvcqyhqo&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=cslpybexmxyuacbyuvib&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=pwqnlynprfgtjbgqoizj&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=jhaoawihmfxgnvmaqffp&amp;fileformat=p2p&amp;archiveformat=gz', u'http://list.iblocklist.com/?list=zbdlwrqkabxbcppvrnos&amp;fileformat=p2p&amp;archiveformat=gz']

if __name__ == '__main__':
    import sys
    assert len(sys.argv) == 2, \
        "Usage: python -m fluent 'some code that can access fluent functions without having to import them'"
    
    exec(sys.argv[1], dict(wrap=wrap, _=_, lib=lib))
