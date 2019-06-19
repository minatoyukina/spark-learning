package ml

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object LinearReg {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()
    val df = spark.read.format("libsvm").load(this.getClass.getResource("/") + "sample_linear_regression_data.txt")
    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val model = lr.fit(df)

    println(s"Coefficients: ${model.coefficients} Intercept： ${model.intercept}")
    val trainingSummary = model.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }
}
