package ml

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

object TF_IDF {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()

    val source = spark.createDataFrame(Seq(
      (0, "so spark like spark hadoop spark and spark like spark"),
      (1, "i wish i can like java so"),
      (2, "but i do not know how to so"),
      (3, "spark is good spark tool so")
    )).toDF("label", "sentence")
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(source)
    wordsData.show(false)
    val hashTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(1000)
    val featuredData = hashTF.transform(wordsData)
    featuredData.show(false)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featuredData)
    val result = idfModel.transform(featuredData)
    result.show(false)
    result.select("label", "features").show(false)

  }
}
