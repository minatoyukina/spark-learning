package ml

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

object K_Means {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()
    val dataSet: DataFrame = spark.createDataFrame(Seq(
      (1, Vectors.dense(0.0, 0.0, 0.0)),
      (2, Vectors.dense(1.1, 4.5, 0.3)),
      (3, Vectors.dense(0.2, 0.2, 0.2)),
      (4, Vectors.dense(9.0, 9.0, 9.0)),
      (5, Vectors.dense(9.1, 0.8, 9.1)),
      (6, Vectors.dense(9.2, 1.2, 6.2)),
      (7, Vectors.dense(3.5, 6.2, 7.2)),
      (8, Vectors.dense(7.2, 9.2, 5.8)),
      (9, Vectors.dense(5.6, 4.4, 9.2)),
      (10, Vectors.dense(4.9, 8.6, 2.9))
    )).toDF("id", "features")

    val model = new KMeans().setK(3).setSeed(1L).fit(dataSet)
    val frame = model.transform(dataSet)
    frame.show(false)
  }

}
