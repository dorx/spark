#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from random import getrandbits

from pyspark.rdd import RDD
from pyspark.mllib._common import _deserialize_double, _deserialize_double_vector
from pyspark.serializers import NoOpSerializer


def uniformRDD(sc, size, numPartitions=None, seed=None):
    """
    Generates an RDD comprised of i.i.d. samples from the
    uniform distribution on [0.0, 1.0].

    To transform the distribution in the generated RDD from U[0.0, 1.0]
    to U[a, b], use
    C{uniformRDD(sc, n, p, seed).map(lambda v: a + (b - a) * v)}

    >>> x = uniformRDD(sc, 100).collect()
    >>> len(x)
    100
    >>> max(x) <= 1.0 and min(x) >= 0.0
    True
    >>> uniformRDD(sc, 100, 4).getNumPartitions()
    4
    >>> parts = uniformRDD(sc, 100, seed=4).getNumPartitions()
    >>> parts == sc.defaultParallelism
    True
    """
    numPartitions, seed = _getDefaultArgs(sc, numPartitions, seed)
    jrdd = sc._jvm.PythonMLLibAPI().uniformRDD(sc._jsc, size, numPartitions, seed)
    uniform =  RDD(jrdd, sc, NoOpSerializer())
    return uniform.map(lambda bytes: _deserialize_double(bytearray(bytes)))

def normalRDD(sc, size, numPartitions=None, seed=None):
    """
    Generates an RDD comprised of i.i.d samples from the standard normal
    distribution.

    To transform the distribution in the generated RDD from standard normal
    to some other normal N(mean, sigma), use
    C{normal(sc, n, p, seed).map(lambda v: mean + sigma * v)}

    >>> x = normalRDD(sc, 1000, seed=1L).collect()
    >>> from pyspark.statcounter import StatCounter
    >>> stats = StatCounter(x)
    >>> stats.count()
    1000L
    >>> abs(stats.mean() - 0.0) < 0.1
    True
    >>> abs(stats.stdev() - 1.0) < 0.1
    True
    """
    numPartitions, seed = _getDefaultArgs(sc, numPartitions, seed)
    jrdd = sc._jvm.PythonMLLibAPI().normalRDD(sc._jsc, size, numPartitions, seed)
    normal =  RDD(jrdd, sc, NoOpSerializer())
    return normal.map(lambda bytes: _deserialize_double(bytearray(bytes)))

def poissonRDD(sc, mean, size, numPartitions=None, seed=None):
    """
    Generates an RDD comprised of i.i.d samples from the Poisson
    distribution with the input mean.

    >>> mean = 100.0
    >>> x = poissonRDD(sc, mean, 1000, seed=1L).collect()
    >>> from pyspark.statcounter import StatCounter
    >>> stats = StatCounter(x)
    >>> stats.count()
    1000L
    >>> abs(stats.mean() - mean) < 0.5
    True
    >>> from math import sqrt
    >>> abs(stats.stdev() - sqrt(mean)) < 0.5
    True
    """
    numPartitions, seed = _getDefaultArgs(sc, numPartitions, seed)
    jrdd = sc._jvm.PythonMLLibAPI().poissonRDD(sc._jsc, mean, size, numPartitions, seed)
    poisson =  RDD(jrdd, sc, NoOpSerializer())
    return poisson.map(lambda bytes: _deserialize_double(bytearray(bytes)))

def uniformVectorRDD(sc, numRows, numCols, numPartitions=None, seed=None):
    """
    Generates an RDD comprised of vectors containing i.i.d samples drawn
    from the uniform distribution on [0.0 1.0].

    >>> import numpy as np
    >>> mat = np.matrix(uniformVectorRDD(sc, 10, 10).collect())
    >>> mat.shape
    (10, 10)
    >>> mat.max() <= 1.0 and mat.min() >= 0.0
    True
    >>> uniformVectorRDD(sc, 10, 10, 4).getNumPartitions()
    4
    """
    numPartitions, seed = _getDefaultArgs(sc, numPartitions, seed)
    jrdd = sc._jvm.PythonMLLibAPI()\
        .uniformVectorRDD(sc._jsc, numRows, numCols, numPartitions, seed)
    uniform =  RDD(jrdd, sc, NoOpSerializer())
    return uniform.map(lambda bytes: _deserialize_double_vector(bytearray(bytes)))

def normalVectorRDD(sc, numRows, numCols, numPartitions=None, seed=None):
    """
    Generates an RDD comprised of vectors containing i.i.d samples drawn
    from the standard normal distribution.

    >>> import numpy as np
    >>> mat = np.matrix(normalVectorRDD(sc, 100, 100, seed=1L).collect())
    >>> mat.shape
    (100, 100)
    >>> abs(mat.mean() - 0.0) < 0.1
    True
    >>> abs(mat.std() - 1.0) < 0.1
    True
    """
    numPartitions, seed = _getDefaultArgs(sc, numPartitions, seed)
    jrdd = sc._jvm.PythonMLLibAPI() \
        .normalVectorRDD(sc._jsc, numRows, numCols, numPartitions, seed)
    normal =  RDD(jrdd, sc, NoOpSerializer())
    return normal.map(lambda bytes: _deserialize_double_vector(bytearray(bytes)))

def poissonVectorRDD(sc, mean, numRows, numCols, numPartitions=None, seed=None):
    """
    Generates an RDD comprised of vectors containing i.i.d samples drawn
    from the Poisson distribution with the input mean.

    >>> import numpy as np
    >>> mean = 100.0
    >>> rdd = poissonVectorRDD(sc, mean, 100, 100, seed=1L)
    >>> mat = np.mat(rdd.collect())
    >>> mat.shape
    (100, 100)
    >>> abs(mat.mean() - mean) < 0.5
    True
    >>> from math import sqrt
    >>> abs(mat.std() - sqrt(mean)) < 0.5
    True
    """
    numPartitions, seed = _getDefaultArgs(sc, numPartitions, seed)
    jrdd = sc._jvm.PythonMLLibAPI() \
        .poissonVectorRDD(sc._jsc, mean, numRows, numCols, numPartitions, seed)
    poisson =  RDD(jrdd, sc, NoOpSerializer())
    return poisson.map(lambda bytes: _deserialize_double_vector(bytearray(bytes)))

def _getDefaultArgs(sc, numPartitions, seed):
    """
    Use sc.defaultParallelism for numPartitions and
    a randomly generated long for seed if either has a value of C{None}

    >>> _getDefaultArgs(sc, 3, 2)
    (3, 2)
    >>> _getDefaultArgs(sc, None, 2) == (sc.defaultParallelism, 2)
    True
    >>> from math import pow
    >>> _getDefaultArgs(sc, None, None)[1] < pow(2, 63)
    True
    """
    if not numPartitions:
        numPartitions = sc.defaultParallelism
    if not seed:
        seed = _nextLong()
    return numPartitions, seed

def _nextLong():
    """
    Returns a random long to be used as RNG seed in the Java APIs.

    Note: only 63 random bits are used here since Long.MAX_VALUE = 2 ^ 63 - 1
    """
    return long(getrandbits(63))


def _test():
    import doctest
    from pyspark.context import SparkContext
    globs = globals().copy()
    # The small batch size here ensures that we see multiple batches,
    # even in these small test examples:
    globs['sc'] = SparkContext('local[2]', 'PythonTest', batchSize=2)
    (failure_count, test_count) = doctest.testmod(globs=globs, optionflags=doctest.ELLIPSIS)
    globs['sc'].stop()
    if failure_count:
        exit(-1)


if __name__ == "__main__":
    _test()
