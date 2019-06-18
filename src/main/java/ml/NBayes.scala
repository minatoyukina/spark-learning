package ml

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession

object NBayes {

  case class RawDataRecord(label: Double, features: Vector)

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()
    import spark.sqlContext.implicits._
    val trainDF = spark.sparkContext.textFile(this.getClass.getResource("/") + "mnist_train.csv").map {
      x =>
        val array = x.split(",")
        RawDataRecord(array(0).toDouble, Vectors.dense(array.takeRight(array.length - 1).map(_.toDouble)))
    }.toDF()

    val testDF = spark.sparkContext.textFile(this.getClass.getResource("/") + "mnist_test.csv").map {
      x =>
        val array = x.split(",")
        RawDataRecord(array(0).toDouble, Vectors.dense(array.takeRight(array.length - 1).map(_.toDouble)))
    }.toDF()

    val model = new NaiveBayes().fit(trainDF)
    val predictions = model.transform(testDF)
    //    predictions.show(false)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    predictions.select("label", "prediction", "probability").show(30, truncate = false)
    println("accuracy: " + accuracy)
  }
}
