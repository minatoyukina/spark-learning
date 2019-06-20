package ml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.sql.SparkSession

object ALS2 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()
    Logger.getRootLogger.setLevel(Level.WARN)
    val data = spark.sparkContext.textFile(this.getClass.getResource("/") + "test.data")
    val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toFloat)
    })
    val dataParts = ratings.randomSplit(Array(0.8, 0.2))
    val frame = spark.createDataFrame(dataParts(0)).toDF()
    val testDF = spark.createDataFrame(dataParts(1)).toDF()
    val asl = new ALS().setRank(10).setNumUserBlocks(10).setRegParam(0.01).setUserCol("user").setItemCol("item").setRatingCol("rating")
    val model = asl.fit(frame)

    val prediction = model.transform(testDF)
    prediction.show()

    val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
    val rmse = evaluator.evaluate(prediction)
    println(s"Root-mean-square error = $rmse")
  }
}
