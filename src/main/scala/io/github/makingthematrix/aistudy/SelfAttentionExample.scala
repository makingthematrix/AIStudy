package io.github.makingthematrix.aistudy

import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.StdArrays
import org.tensorflow.op.core.Constant
import org.tensorflow.types.TFloat32

import scala.util.Random

object SelfAttentionExample {
	def main(args: Array[String]): Unit = {
		// Define a simple example with a sentence
		val sentence = "Alice has a cat"
		val words = sentence.split(" ")

		// Assume we have a simple embedding layer
		// For demonstration, we'll use a small embedding size
		val vocabSize: Int = 100
		val embeddingDim: Int = 8
		
		// Randomly initialize the embedding matrix// Randomly initialize the embedding matrix
		val embeddingsArray = new Array[Array[Float]](vocabSize)
		var i = 0
		while (i < vocabSize) {
			embeddingsArray(i) = new Array[Float](embeddingDim)
			var j = 0
			while (j < embeddingDim) {
				embeddingsArray(i)(j) = Math.random.toFloat
				j += 1
			}
			i += 1
		}
		
/*		val embeddings: Tensor = Tensor.of(classOf[TFloat32], Shape.of(vocabSize, embeddingDim), StdArrays.ndCopyOf(embeddingsArray))
			// TFloat32.tensorOf(Shape.of(vocabSize, embeddingDim), embeddingsArray)


		// Print the vectors
		println(s"Query Vector:\n$queryVector")
		println(s"Key Vector:\n$keyVector")
		println(s"Value Vector:\n$valueVector")*/
	}
}


