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

package org.apache.spark.mllib.stat

import org.apache.spark.mllib.linalg.{Matrix, Vector}
import org.apache.spark.mllib.stat.correlation.Correlations
import org.apache.spark.rdd.RDD

object Statistics {

  /**
   * Compute the Pearson correlation matrix for the input RDD of Vectors.
   */
  def corr(X: RDD[Vector]): Matrix = Correlations.corrMatrix(X)

  /**
   * Compute the correlation matrix for the input RDD of Vectors using the specified method.
   *
   * Methods currently supported: pearson (default), spearman
   */
  def corr(X: RDD[Vector], method: String): Matrix = Correlations.corrMatrix(X, method)

  /**
   * Compute the Pearson correlation for the input RDDs.
   */
  def corr(x: RDD[Double], y: RDD[Double]): Double = Correlations.corr(x, y)

  /**
   * Compute the correlation for the input RDDs using the specified method.
   *
   * Methods currently supported: pearson (default), spearman
   */
  def corr(x: RDD[Double], y: RDD[Double], method: String): Double = Correlations.corr(x, y, method)



}