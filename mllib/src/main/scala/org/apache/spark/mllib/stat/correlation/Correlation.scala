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

package org.apache.spark.mllib.stat.correlation

import org.apache.spark.mllib.linalg.{DenseVector, Matrix, Vector}
import org.apache.spark.rdd.RDD

/**
 * Trait for correlation algorithms.
 */
private[stat] trait Correlation {

  /**
   * Compute correlation for two datasets.
   */
  def computeCorrelation(x: RDD[Double], y: RDD[Double]): Double

  /**
   * Compute the correlation matrix S, for the input matrix, where S(i, j) is the correlation
   * between column i and j. S(i, j) can be NaN if the correlation is undefined for column i and j.
   */
  def computeCorrelationMatrix(X: RDD[Vector]): Matrix

  /**
   * Combine the two input RDD[Double]s into an RDD[Vector] and compute the correlation using the
   * correlation implementation for RDD[Vector]. Can be NaN if correlation is undefined for the
   * input vectors.
   */
  def computeCorrelationWithMatrixImpl(x: RDD[Double], y: RDD[Double]): Double = {
    val mat: RDD[Vector] = x.zip(y).mapPartitions({ iter =>
      iter.map { case (xi, yi) => new DenseVector(Array(xi, yi)) }
    })
    computeCorrelationMatrix(mat)(0, 1)
  }

}

/**
 * Delegates computation to the specific correlation object based on the input method name
 *
 * Currently supported correlations: pearson, spearman.
 * After new correlation algorithms are added, please update the documentation here and in
 * Statistics.scala for the correlation APIs.
 *
 * Maintains the default correlation type, pearson
 */
private[stat] object Correlations {

  // Note: after new types of correlations are implemented, please update this map
  val nameToObjectMap = Map(("pearson", PearsonCorrelation), ("spearman", SpearmansCorrelation))
  val defaultCorrName: String = "pearson"
  val defaultCorr: Correlation = nameToObjectMap(defaultCorrName)

  def corr(x: RDD[Double], y: RDD[Double], method: String = defaultCorrName): Double = {
    val correlation = getCorrelationFromName(method)
    correlation.computeCorrelation(x, y)
  }

  def corrMatrix(X: RDD[Vector], method: String = defaultCorrName): Matrix = {
    val correlation = getCorrelationFromName(method)
    correlation.computeCorrelationMatrix(X)
  }

  /**
   * Perform simple string processing to match the input correlation name with a known name
   *
   * private to mllib for ease of unit testing
   */
  private[mllib] def getCorrelationFromName(method: String): Correlation = {
    if (method.equals(defaultCorrName)) {
      defaultCorr
    } else {
      var correlation: Correlation = defaultCorr
      var matched = false
      nameToObjectMap.foreach { case (name, corr) =>
        if (!matched && method.equals(name)) {
          correlation = corr
          matched = true
        }
      }

      if (matched) {
        correlation
      } else {
        throw new IllegalArgumentException("Correlation name not recognized." +
          " Supported correlations: " + nameToObjectMap.keys.mkString(", "))
      }
    }
  }
}
