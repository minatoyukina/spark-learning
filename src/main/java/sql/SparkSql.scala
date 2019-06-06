package sql

import java.util.Properties

import org.apache.spark.sql.{SQLContext, SparkSession}
import org.junit.{Before, Test}
import org.apache.spark.sql.functions._

class SparkSql {
  var builder: SparkSession = _
  var sql: SQLContext = _

  @Before
  def init(): Unit = {
    builder = SparkSession.builder().appName("sparkSql").master("local").getOrCreate()
    sql = builder.sqlContext
  }

  @Test
  def testJson(): Unit = {
    val peopleInfo = sql.read.option("multiline", value = true).json(this.getClass.getResource("/") + "people.json")
    peopleInfo.createOrReplaceTempView("people")
    sql.sql("select * from people").show()
  }

  @Test
  def testInsert(): Unit = {
    val peopleInfo = sql.read.option("multiline", value = true).json(this.getClass.getResource("/") + "people.json")
    peopleInfo.createTempView("tempTable")

    val url = "jdbc:mysql://10.14.39.101:3306/diy?serverTimezone=GMT%2B8&useSSL=false"
    val table = "targetTable"
    val properties = new Properties()
    properties.setProperty("user", "root")
    properties.setProperty("password", "123456")
    val frame = sql.read.jdbc(url, table, properties)
    frame.createOrReplaceTempView(table)
    sql.sql(
      """
        |insert overwrite table targetTable
        |select id,name,age
        |from tempTable
      """.stripMargin)
  }

  @Test
  def testSchema(): Unit = {
    val peopleInfo = sql.read.option("multiline", value = true).json(this.getClass.getResource("/") + "people.json")
    println(peopleInfo.schema)
    println(peopleInfo.dtypes)
    println(peopleInfo.columns)
    println(peopleInfo.printSchema())
  }

  @Test
  def testJoin(): Unit = {
    val people = sql.read.option("multiline", value = true).json(this.getClass.getResource("/") + "people.json")
    val salary = sql.read.option("multiline", value = true).json(this.getClass.getResource("/") + "salary.json")
    people.join(salary, "id").show()
    people.join(salary, Seq("id", "name")).show()
    people.join(salary, Seq("id", "name"), "left_outer").show()
    people.join(salary, people("id") === salary("id") and people("name") === salary("name")).show()
  }

  @Test
  def testAgg(): Unit = {
    val salary = sql.read.option("multiline", value = true).json(this.getClass.getResource("/") + "salary.json")
    salary.groupBy("name").agg("salary" -> "max").sort(desc("max(salary)")).show
  }

  @Test
  def testJDBC(): Unit = {
    val url = "jdbc:mysql://10.14.39.101:3306/diy?serverTimezone=GMT%2B8&useSSL=false"
    val table = "tbl_user"
    val properties = new Properties()
    properties.setProperty("user", "root")
    properties.setProperty("password", "123456")
    val frame = sql.read.jdbc(url, table, properties)
    frame.createOrReplaceTempView(table)
    sql.sql("select * from tbl_user").show()
  }

}
