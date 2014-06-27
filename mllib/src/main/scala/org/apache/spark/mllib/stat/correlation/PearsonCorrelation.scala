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

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Matrix, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

class PearsonCorrelation extends Correlation {

  //def computeCorrelationMatrix(x: RDD[Double], y: RDD[Double]): Matrix = {
  //
  //}

/*  def computeCorrelationMatrix(X: RDD[Vector]): Matrix = {
    val rowMatrix = new RowMatrix(X)
    val cov = rowMatrix.computeCovariance()
    //cov is a denseMatrix
  }*/
  def computeCorrelationMatrix(x: RDD[Double], y: RDD[Double]): Matrix = ???

  def computeCorrelationMatrix(X: RDD[Vector]): Matrix = {
    val rowMatrix = new RowMatrix(X)
    val cov = rowMatrix.computeCovariance()
    return cov
  }
}
/*

public RealMatrix covarianceToCorrelation(RealMatrix covarianceMatrix) {
        int nVars = covarianceMatrix.getColumnDimension();
        RealMatrix outMatrix = new BlockRealMatrix(nVars, nVars);
        for (int i = 0; i < nVars; i++) {
            double sigma = FastMath.sqrt(covarianceMatrix.getEntry(i, i));
            outMatrix.setEntry(i, i, 1d);
            for (int j = 0; j < i; j++) {
                double entry = covarianceMatrix.getEntry(i, j) /
                       (sigma * FastMath.sqrt(covarianceMatrix.getEntry(j, j)));
                outMatrix.setEntry(i, j, entry);
                outMatrix.setEntry(j, i, entry);
            }
        }
        return outMatrix;
    }
 */