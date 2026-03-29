package com.example.executorchedgeaiinference

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

import org.pytorch.executorch.Tensor
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module

import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var module: Module
    private lateinit var resultText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultText = findViewById(R.id.resultText)

        module = Module.load(assetFilePath("smollm_135m_fixed.pte"))

        val button = findViewById<Button>(R.id.runButton)
        button.setOnClickListener {
            runInference()
        }
    }

    private fun runInference() {
        // SmolLM expects Long (int64) tokens.
        // Using a sample sequence length of 64 for demonstration.
        val sequenceLength = 64L
        val inputData = LongArray(sequenceLength.toInt()) { 0L }
        val maskData = LongArray(sequenceLength.toInt()) { 1L }

        val inputTensor = Tensor.fromBlob(
            inputData,
            longArrayOf(1, sequenceLength)
        )

        val maskTensor = Tensor.fromBlob(
            maskData,
            longArrayOf(1, sequenceLength)
        )

        try {
            // ✅ Use EValue instead of IValue
            val outputs = module.forward(EValue.from(inputTensor), EValue.from(maskTensor))
            val outputTensor = outputs[0].toTensor()
            val scores = outputTensor.dataAsFloatArray

            val maxIdx = scores.indices.maxByOrNull { scores[it] } ?: -1
            resultText.text = "Predicted class index: $maxIdx"
        } catch (e: Exception) {
            e.printStackTrace()
            resultText.text = "Error: ${e.message}"
        }
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        return file.absolutePath
    }
}
