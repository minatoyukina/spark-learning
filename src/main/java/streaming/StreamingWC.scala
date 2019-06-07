package streaming

import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}

object StreamingWC {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName).setMaster("local[2]")
    val sc = new SparkContext(conf)

    sc.setCheckpointDir("hdfs://db1:9000/sparkCP/")
    val ssc = new StreamingContext(sc, Seconds(5))

    val ds = ssc.socketTextStream("192.168.178.130", 8888)
    //    val result = ds.flatMap(_.split(" ")).map((_, 1)).reduceByKey(_ + _)
    val result = ds.flatMap(_.split(" ")).map((_, 1)).updateStateByKey((it: Iterator[(String, Seq[Int], Option[Int])]) =>
      it.flatMap { case (x, y, z) => Some(y.sum + z.getOrElse(0)).map(m => (x, m)) }
      , new HashPartitioner(sc.defaultParallelism), rememberPartitioner = true)
    result.print()

    ssc.start()
    ssc.awaitTermination()
  }
}
