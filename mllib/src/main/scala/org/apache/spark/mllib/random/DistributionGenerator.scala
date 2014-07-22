/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.random

import cern.jet.random.Poisson
import cern.jet.random.engine.DRand

import org.apache.spark.util.random.{XORShiftRandom, Pseudorandom}

/**
 * Trait for random number generators that generate i.i.d values from a distribution.
 */
trait DistributionGenerator extends Pseudorandom with Serializable {

  /**
   * @return An i.i.d sample as a Double from an underlying distribution.
   */
  def nextValue(): Double

  /**
   * @return A copy of the DistributionGenerator with a new instance of the rng object used in the
   *         class when applicable. Each partition has a unique seed and therefore requires its
   *         own instance of the DistributionGenerator.
   */
  def newInstance(): DistributionGenerator
}

/**
 * Generates i.i.d. samples from U[0.0, 1.0]
 */
class UniformGenerator() extends DistributionGenerator {

  // XORShiftRandom for better performance. Thread safety isn't necessary here.
  private val random = new XORShiftRandom()

  /**
   * @return An i.i.d sample as a Double from U[0.0, 1.0].
   */
  override def nextValue(): Double = {
    random.nextDouble()
  }

  /** Set random seed. */
  override def setSeed(seed: Long) = random.setSeed(seed)

  override def newInstance(): UniformGenerator = new UniformGenerator()
}

/**
 * Generates i.i.d. samples from the Standard Normal Distribution.
 */
class StandardNormalGenerator() extends DistributionGenerator {

  // XORShiftRandom for better performance. Thread safety isn't necessary here.
  private val random = new XORShiftRandom()

  /**
   * @return An i.i.d sample as a Double from the Normal distribution.
   */
  override def nextValue(): Double = {
      random.nextGaussian()
  }

  /** Set random seed. */
  override def setSeed(seed: Long) = random.setSeed(seed)

  override def newInstance(): StandardNormalGenerator = new StandardNormalGenerator()
}

/**
 * Generates i.i.d. samples from the Poisson distribution with the given mean.
 *
 * @param mean mean for the Poisson distribution.
 */
class PoissonGenerator(val mean: Double) extends DistributionGenerator {

  private var rng = new Poisson(mean, new DRand)

  /**
   * @return An i.i.d sample as a Double from the Poisson distribution.
   */
  override def nextValue(): Double = rng.nextDouble()

  /** Set random seed. */
  override def setSeed(seed: Long) {
    rng = new Poisson(mean, new DRand(seed.toInt))
  }

  override def newInstance(): PoissonGenerator = new PoissonGenerator(mean)
}
