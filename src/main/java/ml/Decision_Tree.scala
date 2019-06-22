package ml

import ml.NBayes.RawDataRecord
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.SparkSession

object Decision_Tree {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()
    import spark.sqlContext.implicits._
    val trainDF = spark.sparkContext.textFile(this.getClass.getResource("/") + "mnist_train.csv").map {
      x =>
        val array = x.split(",")
        Vectors.dense(array.takeRight(array.length - 1).map(_.toDouble))
        RawDataRecord(array(0).toDouble, Vectors.dense(array.takeRight(array.length - 1).map(_.toDouble)))
    }.toDF()

    val testDF = spark.sparkContext.textFile(this.getClass.getResource("/") + "mnist_test.csv").map {
      x =>
        val array = x.split(",")
        RawDataRecord(array(0).toDouble, Vectors.dense(array.takeRight(array.length - 1).map(_.toDouble)))
    }.toDF()
    val model = new DecisionTreeClassifier().fit(trainDF)
    val predictions = model.transform(testDF)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    predictions.select("label", "prediction", "probability").show(30, truncate = false)
    println("accuracy: " + accuracy)

    val regression = new DecisionTreeRegressor().fit(trainDF)
    val frame = regression.transform(testDF)
    frame.select("label", "prediction").show(false)
  }
}
