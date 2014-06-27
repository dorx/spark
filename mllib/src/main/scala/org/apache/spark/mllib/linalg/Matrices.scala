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

package org.apache.spark.mllib.linalg

import breeze.linalg.{Matrix => BM, DenseMatrix => BDM}

/**
 * Trait for a local matrix.
 */
trait Matrix extends Serializable {

  /** Number of rows. */
  def numRows: Int

  /** Number of columns. */
  def numCols: Int

  /** Converts to a dense array in column major. */
  def toArray: Array[Double]

  /** Converts to a breeze matrix. */
  private[mllib] def toBreeze: BM[Double]

  /** Gets the (i, j)-th element. */
  private[mllib] def apply(i: Int, j: Int): Double = toBreeze(i, j)

  /** Updates the (i, j)-th element to value v. */
  private[mllib] def update(i: Int, j: Int, v: Double): Unit = {toBreeze(i, j) = v}

  override def toString: String = toBreeze.toString()
}

/**
 * Column-majored dense matrix.
 * The entry values are stored in a single array of doubles with columns listed in sequence.
 * For example, the following matrix
 * {{{
 *   1.0 2.0
 *   3.0 4.0
 *   5.0 6.0
 * }}}
 * is stored as `[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]`.
 *
 * @param numRows number of rows
 * @param numCols number of columns
 * @param values matrix entries in column major
 */
class DenseMatrix(val numRows: Int, val numCols: Int, val values: Array[Double]) extends Matrix {

  require(values.length == numRows * numCols)

  override def toArray: Array[Double] = values

  private[mllib] override def toBreeze: BM[Double] = new BDM[Double](numRows, numCols, values)
}

class SymmetricMatrix(val n: Int, val values: Array[Double])
  extends Matrix {

  require(values.size == (n + 1) * n / 2.0, "Wrong number of entries in array of values.")

  var colCache = 0
  var rowCache = 0
  var indexCache = 0

  /** Number of rows. */
  override def numRows: Int = n

  /** Number of columns. */
  override def numCols: Int = n

  override def apply(row: Int, col: Int): Double = {
    if (col < row) {
      return apply(col, row)
    }
    values(linearIndex(row, col))
  }

  override def update(row: Int, col: Int, v: Double): Unit = {
    if (col < row) {
      return update(col, row, v: Double)
    }
    values(linearIndex(row, col)) = v
  }

  // values should be upper triangular
  def linearIndex(row: Int, col: Int): Int = {
    if(row < - numRows || row >= numRows) throw new IndexOutOfBoundsException
    if(col < - numCols || col >= numCols) throw new IndexOutOfBoundsException
    val trueRow = if(row < 0) row + numRows else row
    val trueCol = if(col < 0) col + numCols else col
    if (trueRow == rowCache && trueCol == colCache) {
      return indexCache
    }

    val colSkips = (trueCol + 1) * trueCol / 2.0
    colCache = trueCol
    rowCache = trueRow
    indexCache = (colSkips + trueRow).toInt
    indexCache
  }

  /** Converts to a dense array in column major. */
  override def toArray: Array[Double] = Matrices.triuToFull(numCols, values).toArray

  /** Converts to a breeze matrix. */
  override private[mllib] def toBreeze: BM[Double] = new BDM[Double](numRows, numCols, toArray)
}

/**
 * Factory methods for [[org.apache.spark.mllib.linalg.Matrix]].
 */
object Matrices {

  /**
   * Creates a column-majored dense matrix.
   *
   * @param numRows number of rows
   * @param numCols number of columns
   * @param values matrix entries in column major
   */
  def dense(numRows: Int, numCols: Int, values: Array[Double]): Matrix = {
    new DenseMatrix(numRows, numCols, values)
  }

  /**
   * Creates a Matrix instance from a breeze matrix.
   * @param breeze a breeze matrix
   * @return a Matrix instance
   */
  private[mllib] def fromBreeze(breeze: BM[Double]): Matrix = {
    breeze match {
      case dm: BDM[Double] =>
        require(dm.majorStride == dm.rows,
          "Do not support stride size different from the number of rows.")
        new DenseMatrix(dm.rows, dm.cols, dm.data)
      case _ =>
        throw new UnsupportedOperationException(
          s"Do not support conversion from type ${breeze.getClass.getName}.")
    }
  }

  /**
   * Fills a full square matrix from its upper triangular part.
   */
  def triuToFull(n: Int, U: Array[Double]): Matrix = {
    val G = new BDM[Double](n, n)

    var row = 0
    var col = 0
    var idx = 0
    var value = 0.0
    while (col < n) {
      row = 0
      while (row < col) {
        value = U(idx)
        G(row, col) = value
        G(col, row) = value
        idx += 1
        row += 1
      }
      G(col, col) = U(idx)
      idx += 1
      col +=1
    }

    dense(n, n, G.data)
  }
}
