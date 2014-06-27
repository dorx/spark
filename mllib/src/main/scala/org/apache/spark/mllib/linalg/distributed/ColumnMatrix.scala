package org.apache.spark.mllib.linalg.distributed

import breeze.linalg.DenseMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.ColumnVector

class ColumnMatrix(
    val columns: RDD[ColumnVector],
    private var nRows: Long,
    private var nCols: Int) extends DistributedMatrix{

  def this(columns: RDD[ColumnVector]) = this(columns, 0L, 0)
  /** Gets or computes the number of rows. */
  override def numRows(): Long = ???

  /** Collects data and assembles a local dense breeze matrix (for test only). */
  override private[mllib] def toBreeze(): DenseMatrix[Double] = ???

  /** Gets or computes the number of columns. */
  override def numCols(): Long = ???

  //maybe column based computations here
}


object ColumnMatrix {
  def fromRowMatrix(rowMatrix: RowMatrix): ColumnMatrix = ???
}