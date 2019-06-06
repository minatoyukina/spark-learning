package wc

import org.apache.spark.{SparkConf, SparkContext}

object SparkWC {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SparkWC").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val lines = sc.textFile(args(0))
//    val lines = sc.textFile("D:\\a.txt")
    val words = lines.filter(!_.equals("")).flatMap(_.split(" "))
    val paired = words.map((_, 1))
    val reduced = paired.reduceByKey(_ + _)
    val res = reduced.sortBy(_._2, ascending = false)
    res.saveAsTextFile(args(1))
//    println(res.collect().toBuffer)
    sc.stop()
  }
}
