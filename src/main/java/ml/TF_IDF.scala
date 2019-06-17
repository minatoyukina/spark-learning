package ml

import org.ansj.recognition.impl.StopRecognition
import org.ansj.splitWord.analysis.ToAnalysis
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession

object TF_IDF {
  val spark: SparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()

  def main(args: Array[String]): Unit = {
    //    en()
    zh()
  }

  def en(): Unit = {
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

  def zh(): Unit = {
    import scala.collection.JavaConverters._

    val articles = spark.sparkContext.wholeTextFiles(this.getClass.getResource("/") + "news/*")
    val filename = articles.map(_._1.split("/").last)

    val stopWords = spark.sparkContext.textFile(this.getClass.getResource("/") + "zh-stopWords.txt").collect().toSeq.asJava
    val filter = new StopRecognition().insertStopWords(stopWords)
    filter.insertStopNatures("w", null, "null")
    val source = articles.map(file => {
      val str = ToAnalysis.parse(file._2).recognition(filter).toStringWithOutNature(" ")
      (file._1.split("/").last, str.split(" "))
    })
    val df = spark.createDataFrame(source).toDF("article", "words")

    //    val hashTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(100000)

    val model = new CountVectorizer().setInputCol("words").setOutputCol("rawFeatures").fit(df)

    val featuredData = model.transform(df)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featuredData)
    val result = idfModel.transform(featuredData)
    result.show(false)

    val wordMap = df.select("words").rdd.flatMap { row => {
      row.getAs[Seq[String]](0).map { w => (model.vocabulary.indexOf(w), w) }
    }
    }.collect().toMap

    val keyWords = result.select("features").rdd.map { x => {
      val v = x.getAs[SparseVector](0)
      v.indices.zip(v.values).sortWith((a, b) => {
        a._2 > b._2
      }).take(10).map(x => (wordMap(x._1), x._2))
    }
    }

    filename.zip(keyWords).collect().foreach(x => {
      println(x._1)
      x._2.foreach(x => println(x._1 + ": " + x._2 + " "))
      println("----" * 20)
    })
  }
}
