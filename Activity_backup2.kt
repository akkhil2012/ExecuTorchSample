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
        private const val TOKENIZER_FILE  = "tokenizer.model"
        private const val MODEL_URL       = "$HF_BASE_URL/$MODEL_FILENAME"
        private const val TOKENIZER_URL   = "$HF_BASE_URL/$TOKENIZER_FILE"

        


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

    private fun downloadTokenizerWithOkHttp(onComplete: (Boolean) -> Unit) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val client = okhttp3.OkHttpClient.Builder()
                    .connectTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
                    .readTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
                    .build()

                val request = okhttp3.Request.Builder()
                    .url(TOKENIZER_URL)
                    .build()

                Log.d(TAG, "Downloading tokenizer via OkHttp from $TOKENIZER_URL")
                val response = client.newCall(request).execute()
                Log.d(TAG, "Response code: ${response.code}")
                Log.d(TAG, "Content-Length header: ${response.header("Content-Length")}")
                Log.d(TAG, "Content-Type: ${response.header("Content-Type")}")

                if (!response.isSuccessful) {
                    Log.e(TAG, "Tokenizer download failed: ${response.code}")
                    withContext(Dispatchers.Main) { onComplete(false) }
                    return@launch
                }

                val destFile = getModelFile(TOKENIZER_FILE)
                response.body?.byteStream()?.use { input ->
                    destFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }

                Log.d(TAG, "Tokenizer downloaded: ${destFile.length()} bytes")
                withContext(Dispatchers.Main) {
                    onComplete(destFile.length() > 480_000L)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Tokenizer download error", e)
                withContext(Dispatchers.Main) { onComplete(false) }
            }
        }
    }


    private fun isValidTokenizerBin(file: File): Boolean {
        if (!file.exists()) {
            Log.w(TAG, "Tokenizer file does not exist: ${file.absolutePath}")
            return false
        }

        // Size check — tokenizer.json is ~433KB, real tokenizer.bin is 500KB+
        if (file.length() < 480_000L) {
            Log.w(TAG, "❌ Tokenizer too small: ${file.length()} bytes — likely tokenizer.json renamed to tokenizer.bin")
            return false
        }

        // Magic bytes check — SentencePiece binary starts with 0x0A (protobuf field tag)
        // tokenizer.json starts with '{' (0x7B) — completely different
        try {
            val header = ByteArray(4)
            file.inputStream().use { it.read(header) }

            val firstByte = header[0].toInt() and 0xFF
            Log.d(TAG, "Tokenizer first byte: 0x${firstByte.toString(16).uppercase()} | size: ${file.length()} bytes")

            // JSON files start with '{' = 0x7B
            if (firstByte == 0x7B) {
                Log.e(TAG, "❌ TOKENIZER IS JSON — first byte is '{' (0x7B). This is tokenizer.json not tokenizer.bin!")
                return false
            }

            // SentencePiece protobuf starts with 0x0A
            if (firstByte != 0x0A) {
                Log.w(TAG, "⚠️ Unexpected first byte 0x${firstByte.toString(16).uppercase()} — may not be valid SentencePiece binary")
                // Still allow it — some valid tokenizer.bin variants start differently
            }

            Log.d(TAG, "✅ Tokenizer looks valid: size=${file.length()}, firstByte=0x${firstByte.toString(16).uppercase()}")
            return true

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to read tokenizer header: ${e.message}")
            return false
        }
    }


    private fun initModelFiles() {
        val modelFile     = getModelFile(MODEL_FILENAME)
        val tokenizerFile = getModelFile(TOKENIZER_FILE)

        Log.d(TAG, "=== FILE CHECK ===")
        Log.d(TAG, "Model path: ${modelFile.absolutePath}")
        Log.d(TAG, "Model exists: ${modelFile.exists()}, size: ${modelFile.length()} bytes")
        Log.d(TAG, "Tokenizer path: ${tokenizerFile.absolutePath}")
        Log.d(TAG, "Tokenizer exists: ${tokenizerFile.exists()}, size: ${tokenizerFile.length()} bytes")

        if (tokenizerFile.exists() && !isValidTokenizerBin(tokenizerFile)) {
            Log.e(TAG, "❌ Wrong tokenizer detected — deleting and re-downloading")
            tokenizerFile.delete()
            chatAdapter.addMessage(
                ChatMessage(
                    "❌ Wrong tokenizer file detected (tokenizer.json renamed as tokenizer.bin). " +
                            "Deleting and downloading correct SentencePiece tokenizer.bin…",
                    isUser = false
                )
            )
        }


        val modelOk     = modelFile.exists() && modelFile.length() > 1_000_000_000L
        val tokenizerOk = tokenizerFile.exists() && tokenizerFile.length() > 400_000L

        Log.d(TAG, "modelOk=$modelOk, tokenizerOk=$tokenizerOk")

        when {
            // ── Both files ready → load immediately, skip all downloads ──────────
            modelOk && tokenizerOk -> {
                Log.d(TAG, "✅ Both files already present — skipping download")
                statusText.text = "✅ Model files found"
                chatAdapter.addMessage(
                    ChatMessage("✅ Model already downloaded. Loading…", isUser = false)
                )
                loadModelAsync(modelFile.absolutePath, tokenizerFile.absolutePath)
            }

            // ── Model present but tokenizer missing/corrupt → only download tokenizer
            modelOk && !tokenizerOk -> {
                Log.w(TAG, "⚠️ Model OK but tokenizer missing/corrupt (${tokenizerFile.length()} bytes) — downloading tokenizer only")
                if (tokenizerFile.exists()) tokenizerFile.delete()
                chatAdapter.addMessage(
                    ChatMessage("📥 Model found. Downloading tokenizer only…", isUser = false)
                )
                downloadTokenizerOnly(modelFile)
            }

            // ── Model missing → download everything ──────────────────────────────
            else -> {
                Log.d(TAG, "📥 Model missing — starting full download")
                if (!tokenizerOk && tokenizerFile.exists()) tokenizerFile.delete()
                chatAdapter.addMessage(
                    ChatMessage("📥 Downloading model (~2.2 GB) and tokenizer…", isUser = false)
                )
                startDownloads()
            }
        }
    }


    private fun downloadTokenizerOnly(modelFile: File) {
        statusText.text = "📥 Downloading tokenizer…"

        Log.d(TAG, "Starting tokenizer download from: $TOKENIZER_URL")
        chatAdapter.addMessage(
            ChatMessage("📥 Downloading tokenizer from HuggingFace…", isUser = false)
        )

        downloadTokenizerWithOkHttp { success ->
            if (success) {
                Log.d(TAG, "✅ Tokenizer downloaded and validated successfully")
                chatAdapter.addMessage(
                    ChatMessage("✅ Tokenizer ready. Loading model…", isUser = false)
                )
                loadModelAsync(
                    modelFile.absolutePath,
                    getModelFile(TOKENIZER_FILE).absolutePath
                )
            } else {
                Log.e(TAG, "❌ Tokenizer download failed or file is invalid")
                statusText.text = "❌ Tokenizer failed"

                // Check if it failed due to wrong file content vs network error
                val tokenizerFile = getModelFile(TOKENIZER_FILE)
                val failReason = when {
                    !tokenizerFile.exists() ->
                        "Network error — file not downloaded.\nCheck your internet connection."
                    tokenizerFile.length() < 480_000L ->
                        "Wrong file on server (${tokenizerFile.length()} bytes).\n" +
                                "Your HuggingFace repo has tokenizer.json uploaded as tokenizer.bin.\n" +
                                "Upload the real SentencePiece tokenizer.bin (~500KB+)."
                    else ->
                        "File downloaded but invalid format.\n" +
                                "First byte is '{' — this is a JSON file, not SentencePiece binary."
                }

                Log.e(TAG, "Fail reason: $failReason")

                chatAdapter.addMessage(
                    ChatMessage(
                        "❌ Tokenizer download failed.\n\n$failReason\n\n" +
                                "Tap 🔄 Clear button to retry after fixing the issue.",
                        isUser = false
                    )
                )

                // Re-enable clear button so user can retry
                sendButton.isEnabled = false
                clearButton.isEnabled = true
                statusText.text = "❌ Fix tokenizer then retry"
            }
        }
    }


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
            @Suppress("UnspecifiedRegisterReceiverFlag")
            registerReceiver(downloadReceiver, filter)
        }

        // Download tokenizer first via OkHttp
        chatAdapter.addMessage(ChatMessage("📥 Downloading tokenizer…", isUser = false))
        downloadTokenizerWithOkHttp { success ->
            Log.d(TAG, "Tokenizer OkHttp download success=$success")
            chatAdapter.addMessage(
                ChatMessage(
                    if (success) "✅ Tokenizer ready" else "❌ Tokenizer download failed",
                    isUser = false
                )
            )
        }

        // Download model via DownloadManager
        modelDownloadId = enqueueDownload(MODEL_URL, MODEL_FILENAME, "Phi-3 Model")
        pollDownloadProgress()
    }


    private fun enqueueDownload(url: String, fileName: String, title: String): Long {
        val destFile = getModelFile(fileName)

        // Delete any existing partial/corrupt file first
        if (destFile.exists()) {
            Log.d(TAG, "Deleting existing file: ${destFile.name} (${destFile.length()} bytes)")
            destFile.delete()
        }

        val request = DownloadManager.Request(Uri.parse(url)).apply {
            setTitle(title)
            setDescription("Downloading $fileName")
            setDestinationUri(Uri.fromFile(destFile))
            setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            setAllowedNetworkTypes(
                DownloadManager.Request.NETWORK_WIFI or
                        DownloadManager.Request.NETWORK_MOBILE
            )
            // Allow roaming
            setAllowedOverRoaming(true)
            if (HF_TOKEN.isNotEmpty()) {
                addRequestHeader("Authorization", "Bearer $HF_TOKEN")
            }
        }
        val id = downloadManager!!.enqueue(request)
        Log.d(TAG, "Enqueued download: $fileName, id=$id, dest=${destFile.absolutePath}")
        return id
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
        val modelFile     = getModelFile(MODEL_FILENAME)
        val tokenizerFile = getModelFile(TOKENIZER_FILE)

        Log.d(TAG, "Download complete broadcast: id=$id, modelDownloadId=$modelDownloadId")
        Log.d(TAG, "Model size: ${modelFile.length()}, Tokenizer size: ${tokenizerFile.length()}")

        if (id != modelDownloadId) {
            Log.d(TAG, "Ignoring unrelated download id=$id")
            return
        }

        val modelOk     = modelFile.exists() && modelFile.length() > 1_000_000_000L
        val tokenizerOk = tokenizerFile.exists() && tokenizerFile.length() > 400_000L

        when {
            modelOk && tokenizerOk -> {
                Log.d(TAG, "✅ Both files ready after download — loading model")
                runOnUiThread {
                    chatAdapter.addMessage(
                        ChatMessage("✅ Download complete! Loading model…", isUser = false)
                    )
                    loadModelAsync(modelFile.absolutePath, tokenizerFile.absolutePath)
                }
            }

            modelOk && !tokenizerOk -> {
                // Model downloaded but tokenizer OkHttp may still be in progress
                Log.w(TAG, "⚠️ Model downloaded but tokenizer not ready yet (${tokenizerFile.length()} bytes) — waiting")
                runOnUiThread {
                    chatAdapter.addMessage(
                        ChatMessage("⏳ Model downloaded. Waiting for tokenizer…", isUser = false)
                    )
                }
                // Poll until tokenizer arrives (OkHttp may still be writing it)
                waitForTokenizerThenLoad(modelFile)
            }

            else -> {
                Log.e(TAG, "❌ Model download incomplete: size=${modelFile.length()}")
                runOnUiThread {
                    chatAdapter.addMessage(
                        ChatMessage("❌ Model download incomplete. Please restart.", isUser = false)
                    )
                }
            }
        }
    }

    private fun waitForTokenizerThenLoad(modelFile: File) {
        lifecycleScope.launch(Dispatchers.IO) {
            val tokenizerFile = getModelFile(TOKENIZER_FILE)
            var waited = 0
            val maxWaitSeconds = 60

            while (waited < maxWaitSeconds) {
                if (tokenizerFile.exists() && tokenizerFile.length() > 400_000L) {
                    Log.d(TAG, "✅ Tokenizer ready after ${waited}s wait")
                    withContext(Dispatchers.Main) {
                        chatAdapter.addMessage(
                            ChatMessage("✅ All files ready. Loading model…", isUser = false)
                        )
                        loadModelAsync(modelFile.absolutePath, tokenizerFile.absolutePath)
                    }
                    return@launch
                }
                delay(1000)
                waited++
                Log.d(TAG, "Waiting for tokenizer… ${waited}s (size=${tokenizerFile.length()})")
            }

            // Timed out — try downloading tokenizer again
            Log.e(TAG, "❌ Tokenizer not ready after ${maxWaitSeconds}s — re-downloading")
            withContext(Dispatchers.Main) {
                chatAdapter.addMessage(
                    ChatMessage("⏳ Re-trying tokenizer download…", isUser = false)
                )
                downloadTokenizerOnly(modelFile)
            }
        }
    }


    private fun loadModelAsync(modelPath: String, tokenizerPath: String) {
        statusText.text = "⏳ Loading Phi-3 Mini…"
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "Initializing LlamaModule...")
                llamaModule = LlamaModule(3,modelPath, tokenizerPath, TEMPERATURE)

                // ❌ DELETE THIS ENTIRE BLOCK — this is breaking your model
                // llamaModule?.generate("Hi", 5, object : LlamaCallback {
                //     override fun onResult(token: String?) { ... }
                //     override fun onStats(tps: Float) { ... }
                // })

                withContext(Dispatchers.Main) {
                    isModelLoaded = true
                    conversationHistory.clear()
                    statusText.text = "✅ Phi-3 Mini ready"
                    setInputEnabled(true)
                    chatAdapter.addMessage(
                        ChatMessage("Model loaded! Ask me anything.", isUser = false)
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Load error", e)
                withContext(Dispatchers.Main) {
                    statusText.text = "❌ Load failed: ${e.message}"
                    chatAdapter.addMessage(ChatMessage("⚠️ ${e.message}", isUser = false))
                }
            }
        }
    }
    /*private fun loadModelAsync(modelPath: String, tokenizerPath: String) {
        statusText.text = "⏳ Loading Phi-3 Mini…"
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "Initializing LlamaModule...")
                llamaModule = LlamaModule(modelPath, tokenizerPath, TEMPERATURE)


                Log.d(TAG, "Running warm-up generation...")
                llamaModule?.generate("Hi", 5, object : LlamaCallback {
                    override fun onResult(token: String?) {
                        Log.d(TAG, "Warm-up token: '$token'")
                       // Log.d(TAG, "RAW TOKEN: '${token}' | accumulated: ${token.length} chars")
                    }
                    override fun onStats(tps: Float) {
                        Log.d(TAG, "Warm-up tps: $tps")
                       // Log.d(TAG, "RAW TOKEN: '${token}' | accumulated: ${generatedTokens.length} chars")
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
    }*/

    private fun sendMessage() {
        val userText = messageInput.text.toString().trim()
        if (userText.isEmpty() || !isModelLoaded || isGenerating) return

        messageInput.text.clear()
        hideKeyboard()

        chatAdapter.addMessage(ChatMessage(userText, isUser = true))
        scrollToBottom()

        chatAdapter.addMessage(ChatMessage("", isUser = false, isStreaming = true))

        val prompt = buildPhi3Prompt(userText)
        Log.d(TAG, "════════════════════════════════════")
        Log.d(TAG, "USER INPUT: '$userText'")
        Log.d(TAG, "FULL PROMPT SENT TO MODEL:")
        Log.d(TAG, prompt)
        Log.d(TAG, "════════════════════════════════════")

        Log.d(TAG, "Starting generation with prompt:\n$prompt")

        val generatedTokens = StringBuilder()
        isGenerating = true
        sendButton.isEnabled = false
        statusText.text = "⚡ Generating…"

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                var tokenCount = 0
                var nullCount = 0
                var emptyCount = 0
                llamaModule?.generate(prompt, MAX_NEW_TOKENS, object : LlamaCallback {
                    override fun onResult(token: String?) {
                        tokenCount++
                        when {
                            token == null -> {
                                nullCount++
                                Log.w(TAG, "⚠️ TOKEN #$tokenCount IS NULL")
                            }
                            token.isEmpty() -> {
                                emptyCount++
                                Log.w(TAG, "⚠️ TOKEN #$tokenCount IS EMPTY STRING")
                            }
                            else -> {
                                Log.d(TAG, "✅ TOKEN #$tokenCount: '$token'")
                                generatedTokens.append(token)
                                lifecycleScope.launch(Dispatchers.Main) {
                                    chatAdapter.updateLastMessage(generatedTokens.toString(), true)
                                    scrollToBottom()
                                }
                            }
                        }
                    }

                    override fun onStats(tps: Float) {
                        Log.d(TAG, "STATS: tps=$tps | tokenCount=$tokenCount | nullCount=$nullCount | emptyCount=$emptyCount")
                    }
                })

                /*llamaModule?.generate(prompt, MAX_NEW_TOKENS, object : LlamaCallback {
                    override fun onResult(token: String?) {
                        Log.d(TAG, "RAW TOKEN: '${token}' | accumulated: ${generatedTokens.length} chars")

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
                })*/

                withContext(Dispatchers.Main) {
                    val final = generatedTokens.toString()
                    Log.d(TAG, "════════════════════════════════════")
                    Log.d(TAG, "GENERATION DONE")
                    Log.d(TAG, "tokenCount  = $tokenCount")
                    Log.d(TAG, "nullCount   = $nullCount")
                    Log.d(TAG, "emptyCount  = $emptyCount")
                    Log.d(TAG, "final length= ${final.length}")
                    Log.d(TAG, "final text  = '$final'")
                    Log.d(TAG, "Tokenizer size: ${getModelFile(TOKENIZER_FILE).length()} bytes")

                    Log.d(TAG, "════════════════════════════════════")

                    //val final = "talk to tarla dalal"
                    Log.d(TAG, "Generation complete. Final text------>: '$final'")
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
