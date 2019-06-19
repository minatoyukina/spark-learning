package util

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import scala.io.Source

object MnistToImg {


  def main(args: Array[String]): Unit = {
    val source = Source.fromInputStream(this.getClass.getResourceAsStream("/mnist_train.csv"))
    val strings = source.getLines().next().split(",")
    val ints = strings.takeRight(strings.length - 1).map(_.toInt)
    val bufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB)
    var count = 0
    (0 to 27).foreach(x => (0 to 27).foreach(y => {
      bufferedImage.setRGB(y, x, ints(count))
      count += 1
    }))

    ImageIO.write(bufferedImage, "png", new File("abc.png"))
  }

}



