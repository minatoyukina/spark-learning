package streaming

import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.streaming.kafka.KafkaUtils

object StreamingKafka {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName).setMaster("local[*]")
    val sc = new SparkContext(conf)

    //add spark-core_2.11-1.5.2.logging.jar to global library
    sc.setCheckpointDir("hdfs://db1:9000/sparkCP/")
    val ssc = new StreamingContext(sc, Seconds(5))

    val topics = Map("spark" -> 2)
    val lines = KafkaUtils.createStream(ssc, "db1:2181,db2:2181,db3:2181", "spark", topics).map(_._2)
    val ds1 = lines.flatMap(_.split("\\s+")).map((_, 1))
    val ds2 = ds1.updateStateByKey[Int]((x: Seq[Int], y: Option[Int]) => {
      Some(x.sum + y.getOrElse(0))
    })

    ds2.print()

    ssc.start()
    ssc.awaitTermination()
  }

}
