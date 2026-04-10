package com.example.phi3chat


import android.annotation.SuppressLint
import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.facebook.soloader.SoLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.LlamaCallback
import org.pytorch.executorch.LlamaModule
import java.io.File

class MainActivity : AppCompatActivity() {

    // ── Constants ─────────────────────────────────────────────────────────────
    companion object {
        private const val TAG             = "EdgeApp_Phi3"
        private const val MAX_NEW_TOKENS  = 256
        private const val TEMPERATURE     = 0.0f // Set to 0.0 for deterministic output (debugging)
        private const val SYSTEM_PROMPT   =
            "You are a helpful AI assistant."

        private const val HF_BASE_URL     =
            "https://huggingface.co/akhilsalogra/phi3-mobile-model/resolve/main"
        private const val MODEL_FILENAME  = "phi3_mini_8da4w.pte"
        private const val TOKENIZER_FILE  = "tokenizer.bin"
        private const val MODEL_URL       = "$HF_BASE_URL/$MODEL_FILENAME"
        private const val TOKENIZER_URL   = "$HF_BASE_URL/tokenizer.bin"

        private const val HF_TOKEN = "";


    }

    // ── UI ────────────────────────────────────────────────────────────────────
    private lateinit var recyclerView   : RecyclerView
    private lateinit var messageInput   : EditText
    private lateinit var sendButton     : ImageButton
    private lateinit var clearButton    : ImageButton
    private lateinit var statusText     : TextView
    private lateinit var progressBar    : ProgressBar
    private lateinit var progressText   : TextView
    private lateinit var chatAdapter    : ChatAdapter

    // ── Model ─────────────────────────────────────────────────────────────────
    private var llamaModule    : LlamaModule? = null
    private var isModelLoaded  = false
    private var isGenerating   = false

    // ── Download tracking ─────────────────────────────────────────────────────
    private var modelDownloadId     : Long = -1
    private var tokenizerDownloadId : Long = -1
    private var downloadManager     : DownloadManager? = null
    private var downloadReceiver    : BroadcastReceiver? = null

    // ── Conversation history ──────────────────────────────────────────────────
    private val conversationHistory = StringBuilder()

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        SoLoader.init(this, false)
        setContentView(R.layout.activity_main)
        setupUI()
        initModelFiles()
    }

    override fun onDestroy() {
        super.onDestroy()
        llamaModule?.stop()
        llamaModule = null
        downloadReceiver?.let { unregisterReceiver(it) }
    }

    // ── UI Setup ──────────────────────────────────────────────────────────────
    private fun setupUI() {
        recyclerView = findViewById(R.id.chat_recycler)
        messageInput = findViewById(R.id.message_input)
        sendButton   = findViewById(R.id.send_button)
        clearButton  = findViewById(R.id.clear_button)
        statusText   = findViewById(R.id.status_text)
        progressBar  = findViewById(R.id.download_progress)
        progressText = findViewById(R.id.progress_text)


        chatAdapter = ChatAdapter()
        recyclerView.apply {
            layoutManager = LinearLayoutManager(this@MainActivity).apply {
                stackFromEnd = true
            }
            adapter = chatAdapter
        }

        setInputEnabled(false)

        sendButton.setOnClickListener { sendMessage() }

        clearButton.setOnClickListener {
            conversationHistory.clear()
            chatAdapter.clearMessages()
            chatAdapter.addMessage(
                ChatMessage("Chat cleared. Ask me anything!", isUser = false)
            )
        }

        messageInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) { sendMessage(); true }
            else false
        }
    }

    private fun setInputEnabled(enabled: Boolean) {
        messageInput.isEnabled = enabled
        sendButton.isEnabled   = enabled
        messageInput.hint      = if (enabled) "Ask me anything..." else "Loading model..."
    }

    private fun initModelFiles() {
        val modelFile     = getModelFile(MODEL_FILENAME)
        val tokenizerFile = getModelFile(TOKENIZER_FILE)

        Log.d(TAG, "Checking model files: ${modelFile.absolutePath}")

        if (modelFile.exists() && modelFile.length() > 0 &&
            tokenizerFile.exists() && tokenizerFile.length() > 0) {
            Log.d(TAG, "Files found. Model size: ${modelFile.length()}, Tokenizer size: ${tokenizerFile.length()}")
            statusText.text = "✅ Model files found"
            loadModelAsync(modelFile.absolutePath, tokenizerFile.absolutePath)
        } else {
            Log.d(TAG, "Files NOT found or incomplete. Starting download.")
            chatAdapter.addMessage(
                ChatMessage("📥 Model files missing. Downloading (~2.2GB)…", isUser = false)
            )
            startDownloads()
        }
    }

    @SuppressLint("UnspecifiedRegisterReceiverFlag")
    private fun startDownloads() {
        downloadManager = getSystemService(DOWNLOAD_SERVICE) as DownloadManager

        downloadReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                val completedId = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1)
                onDownloadComplete(completedId)
            }
        }

        val filter = IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(downloadReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(downloadReceiver, filter)
        }

        tokenizerDownloadId = enqueueDownload(TOKENIZER_URL, TOKENIZER_FILE, "Phi-3 Tokenizer")
        modelDownloadId = enqueueDownload(MODEL_URL, MODEL_FILENAME, "Phi-3 Model")

        pollDownloadProgress()
    }

    private fun enqueueDownload(url: String, fileName: String, title: String): Long {
        val destFile = getModelFile(fileName)
        val request = DownloadManager.Request(Uri.parse(url)).apply {
            setTitle(title)
            setDestinationUri(Uri.fromFile(destFile))
            setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            setAllowedNetworkTypes(DownloadManager.Request.NETWORK_WIFI or DownloadManager.Request.NETWORK_MOBILE)
            if (HF_TOKEN.isNotEmpty()) {
                addRequestHeader("Authorization", "Bearer $HF_TOKEN")
            }
        }
        return downloadManager!!.enqueue(request)
    }

    private fun pollDownloadProgress() {
        lifecycleScope.launch(Dispatchers.IO) {
            while (!isModelLoaded) {
                val mP = queryDownloadProgress(modelDownloadId)
                val tP = queryDownloadProgress(tokenizerDownloadId)
                withContext(Dispatchers.Main) {
                    val overall = (mP * 0.99f + tP * 0.01f).toInt()
                    progressBar.progress = overall
                    progressText.text = "Model: $mP% | Tokenizer: $tP%"
                    statusText.text = "📥 Downloading… $overall%"
                }
                if (mP >= 100 && tP >= 100) break
                delay(1000)
            }
        }
    }

    private fun queryDownloadProgress(id: Long): Int {
        if (id == -1L) return 0
        val cursor = downloadManager?.query(DownloadManager.Query().setFilterById(id)) ?: return 0
        return cursor.use {
            if (!it.moveToFirst()) return 0
            val status = it.getInt(it.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
            if (status == DownloadManager.STATUS_SUCCESSFUL) return 100
            val downloaded = it.getLong(it.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
            val total = it.getLong(it.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
            if (total <= 0) 0 else ((downloaded * 100L) / total).toInt()
        }
    }

    private fun onDownloadComplete(id: Long) {
        val modelFile = getModelFile(MODEL_FILENAME)
        val tokenizerFile = getModelFile(TOKENIZER_FILE)
        if (modelFile.exists() && modelFile.length() > 1000000000 && // Roughly check for > 1GB
            tokenizerFile.exists() && tokenizerFile.length() > 500000) {
            runOnUiThread {
                chatAdapter.addMessage(ChatMessage("✅ Download complete! Loading model…", isUser = false))
                loadModelAsync(modelFile.absolutePath, tokenizerFile.absolutePath)
            }
        }
    }

    private fun loadModelAsync(modelPath: String, tokenizerPath: String) {
        statusText.text = "⏳ Loading Phi-3 Mini…"
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "Initializing LlamaModule...")
                llamaModule = LlamaModule(modelPath, tokenizerPath, TEMPERATURE)


                Log.d(TAG, "Running warm-up generation...")
                llamaModule?.generate("Hi", 5, object : LlamaCallback {
                    override fun onResult(token: String?) {
                        Log.d(TAG, "Warm-up token: '$token'")
                    }
                    override fun onStats(tps: Float) {
                        Log.d(TAG, "Warm-up tps: $tps")
                    }
                })

                withContext(Dispatchers.Main) {
                    isModelLoaded = true
                    statusText.text = "✅ Phi-3 Mini ready"
                    setInputEnabled(true)
                    chatAdapter.addMessage(ChatMessage("Model loaded. Ask me anything!", isUser = false))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading model", e)
                withContext(Dispatchers.Main) {
                    statusText.text = "❌ Load failed"
                    chatAdapter.addMessage(ChatMessage("⚠️ Error: ${e.message}", isUser = false))
                }
            }
        }
    }

    private fun sendMessage() {
        val userText = messageInput.text.toString().trim()
        if (userText.isEmpty() || !isModelLoaded || isGenerating) return

        messageInput.text.clear()
        hideKeyboard()

        chatAdapter.addMessage(ChatMessage(userText, isUser = true))
        scrollToBottom()

        chatAdapter.addMessage(ChatMessage("", isUser = false, isStreaming = true))

        val prompt = buildPhi3Prompt(userText)
        Log.d(TAG, "Starting generation with prompt:\n$prompt")

        val generatedTokens = StringBuilder()
        isGenerating = true
        sendButton.isEnabled = false
        statusText.text = "⚡ Generating…"

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                llamaModule?.generate(prompt, MAX_NEW_TOKENS, object : LlamaCallback {
                    override fun onResult(token: String?) {
                        Log.d(TAG, "Received token: '$token'")
                        token?.let {
                            generatedTokens.append(it)
                            lifecycleScope.launch(Dispatchers.Main) {
                                chatAdapter.updateLastMessage(generatedTokens.toString(), true)
                                scrollToBottom()
                            }
                        }
                    }

                    override fun onStats(tps: Float) {
                        Log.d(TAG, "Generation stats: $tps tok/s")
                        lifecycleScope.launch(Dispatchers.Main) {
                            statusText.text = "⚡ %.1f tok/s".format(tps)
                        }
                    }
                })

                withContext(Dispatchers.Main) {
                    val final = generatedTokens.toString()
                    Log.d(TAG, "Generation complete. Final text: '$final'")
                    chatAdapter.updateLastMessage(final, false)
                    conversationHistory.append("<|user|>\n$userText<|end|>\n<|assistant|>\n$final<|end|>\n")
                    isGenerating = false
                    sendButton.isEnabled = true
                    statusText.text = "✅ Phi-3 Mini ready"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Generation error", e)
                withContext(Dispatchers.Main) {
                    chatAdapter.updateLastMessage("⚠️ Error: ${e.message}", false)
                    isGenerating = false
                    sendButton.isEnabled = true
                }
            }
        }
    }

    private fun buildPhi3Prompt(userMessage: String): String =
        "<|system|>\n$SYSTEM_PROMPT<|end|>\n" +
                conversationHistory.toString() +
                "<|user|>\n$userMessage<|end|>\n" +
                "<|assistant|>\n"

    private fun getModelFile(fileName: String): File {
        val dir = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS) ?: filesDir
        return File(dir, fileName)
    }

    private fun scrollToBottom() {
        if (chatAdapter.itemCount > 0) recyclerView.scrollToPosition(chatAdapter.itemCount - 1)
    }

    private fun hideKeyboard() {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(currentFocus?.windowToken, 0)
    }
}
