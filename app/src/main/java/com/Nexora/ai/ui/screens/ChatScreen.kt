package com.Nexora.ai.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Slider
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.Nexora.ai.ml.Inference
import com.Nexora.ai.train.TrainingController
import com.Nexora.ai.ui.theme.NexoraBg
import com.Nexora.ai.ui.theme.NexoraMuted
import com.Nexora.ai.ui.theme.NexoraOnSurface
import com.Nexora.ai.ui.theme.NexoraPrimary
import com.Nexora.ai.ui.theme.NexoraSurface
import com.Nexora.ai.ui.theme.NexoraSurfaceVar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private class Msg(val fromUser: Boolean, text: String) {
    var text by mutableStateOf(text)
}

@Composable
fun ChatScreen() {
    val state by TrainingController.state.collectAsState()
    val model = TrainingController.model
    val tokenizer = TrainingController.tokenizer
    val messages = remember { mutableStateListOf<Msg>() }
    var input by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var showSettings by remember { mutableStateOf(false) }
    var temperature by remember { mutableFloatStateOf(0.8f) }
    var maxTokens by remember { mutableIntStateOf(160) }
    var topK by remember { mutableIntStateOf(40) }
    val quickPrompts = listOf("Jelaskan cara kerja model bahasa", "Bantu periksa kode Kotlin", "Buat rencana training model")
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()

    // Keep the latest generated prefix visible while the model is producing tokens.
    LaunchedEffect(messages.size, messages.lastOrNull()?.text) {
        if (messages.isNotEmpty()) listState.scrollToItem(messages.lastIndex)
    }

    Column(
        Modifier
            .fillMaxSize()
            .background(NexoraBg)
            .padding(horizontal = 16.dp)
            .imePadding()
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 12.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Surface(
                modifier = Modifier.size(42.dp),
                shape = RoundedCornerShape(15.dp),
                color = NexoraPrimary.copy(alpha = 0.13f)
            ) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Icon(Icons.Filled.AutoAwesome, contentDescription = null, tint = NexoraPrimary, modifier = Modifier.size(21.dp))
                }
            }
            Column(Modifier.weight(1f).padding(start = 11.dp)) {
                Text("Nexora", color = NexoraOnSurface, fontSize = 20.sp, fontWeight = FontWeight.Bold)
                Text("Model bahasa lokal", color = NexoraMuted, fontSize = 12.sp)
            }
            Surface(
                shape = RoundedCornerShape(50.dp),
                color = if (model != null && tokenizer != null) NexoraPrimary.copy(alpha = 0.12f) else NexoraSurfaceVar
            ) {
                Text(
                    if (model != null && tokenizer != null) "Model aktif" else "Belum dilatih",
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp),
                    color = if (model != null && tokenizer != null) NexoraPrimary else NexoraMuted,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Medium
                )
            }
            IconButton(onClick = { showSettings = !showSettings }, enabled = modelReady && !busy) {
                Icon(Icons.Filled.Tune, contentDescription = "Pengaturan respons", tint = NexoraPrimary)
            }
            IconButton(onClick = { if (!busy) messages.clear() }, enabled = messages.isNotEmpty() && !busy) {
                Icon(Icons.Filled.DeleteSweep, contentDescription = "Bersihkan percakapan", tint = NexoraMuted)
            }
        }

        if (showSettings && modelReady) {
            Card(
                modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
                colors = CardDefaults.cardColors(containerColor = NexoraSurface),
                shape = RoundedCornerShape(18.dp)
            ) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Temperature", color = NexoraOnSurface, fontSize = 12.sp)
                        Text("%.2f".format(temperature), color = NexoraPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                    Slider(value = temperature, onValueChange = { temperature = it }, valueRange = 0.2f..1.5f)
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Token maksimum", color = NexoraOnSurface, fontSize = 12.sp)
                        Text(maxTokens.toString(), color = NexoraPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                    Slider(value = maxTokens.toFloat(), onValueChange = { maxTokens = it.toInt() }, valueRange = 20f..256f, steps = 22)
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Top-K sampling", color = NexoraOnSurface, fontSize = 12.sp)
                        Text(topK.toString() + " kandidat", color = NexoraPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                    Slider(value = topK.toFloat(), onValueChange = { topK = it.toInt().coerceIn(1, 100) }, valueRange = 1f..100f, steps = 98)
                    Text("Pengaturan ini mengubah cara token dipilih, bukan bobot model yang sudah dilatih.", color = NexoraMuted, fontSize = 11.sp)
                }
            }
        }

        if (messages.isEmpty()) {
            Column(
                modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 4.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Surface(shape = CircleShape, color = NexoraSurfaceVar, modifier = Modifier.size(70.dp)) {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Icon(Icons.Filled.AutoAwesome, contentDescription = null, tint = NexoraPrimary, modifier = Modifier.size(30.dp))
                    }
                }
                Text(
                    "Apa yang ingin kamu pelajari?",
                    modifier = Modifier.padding(top = 18.dp),
                    color = NexoraOnSurface,
                    fontSize = 21.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    if (model == null || tokenizer == null)
                        "Impor dataset dan latih model terlebih dahulu. Chat memakai bobot model yang tersimpan di perangkat."
                    else "Tanyakan sesuatu ke model lokal yang sudah kamu latih.",
                    modifier = Modifier.padding(top = 8.dp, bottom = 22.dp),
                    color = NexoraMuted,
                    fontSize = 13.sp,
                    lineHeight = 19.sp
                )
                listOf(
                    "Jelaskan cara kerja model bahasa",
                    "Bantu periksa kode Kotlin",
                    "Buat rencana training model"
                ).forEach { prompt ->
                    Surface(
                        modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp).clickable { input = prompt },
                        shape = RoundedCornerShape(16.dp),
                        color = NexoraSurface
                    ) {
                        Text(
                            prompt,
                            modifier = Modifier.padding(horizontal = 14.dp, vertical = 13.dp),
                            color = NexoraOnSurface,
                            fontSize = 13.sp
                        )
                    }
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                state = listState,
                contentPadding = PaddingValues(vertical = 12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(messages) { message ->
                    Box(
                        modifier = Modifier.fillMaxWidth(),
                        contentAlignment = if (message.fromUser) Alignment.CenterEnd else Alignment.CenterStart
                    ) {
                        Surface(
                            modifier = Modifier.widthIn(max = 320.dp),
                            shape = RoundedCornerShape(
                                topStart = 20.dp,
                                topEnd = 20.dp,
                                bottomStart = if (message.fromUser) 20.dp else 6.dp,
                                bottomEnd = if (message.fromUser) 6.dp else 20.dp
                            ),
                            color = if (message.fromUser) NexoraPrimary.copy(alpha = 0.16f) else NexoraSurfaceVar
                        ) {
                            Text(
                                message.text + if (!message.fromUser && busy && message === messages.lastOrNull()) " ▍" else "",
                                modifier = Modifier.padding(horizontal = 15.dp, vertical = 11.dp),
                                color = NexoraOnSurface,
                                fontSize = 14.sp,
                                lineHeight = 21.sp
                            )
                        }
                    }
                }
            }
        }

        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            contentPadding = PaddingValues(vertical = 4.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            items(quickPrompts, key = { it }) { prompt ->
                FilterChip(
                    selected = false,
                    onClick = { if (!busy) input = prompt },
                    label = { Text(prompt, fontSize = 11.sp) },
                    colors = FilterChipDefaults.filterChipColors(containerColor = NexoraSurface)
                )
            }
        }
        if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp, bottom = 10.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                modifier = Modifier.weight(1f),
                enabled = !busy,
                placeholder = { Text("Ketik pesan…", color = NexoraMuted) },
                shape = RoundedCornerShape(24.dp),
                minLines = 1,
                maxLines = 4
            )
            Spacer(Modifier.size(8.dp))
            val canSend = !busy && model != null && tokenizer != null && input.isNotBlank()
            Surface(
                modifier = Modifier.size(50.dp),
                shape = CircleShape,
                color = if (canSend) NexoraPrimary else NexoraSurfaceVar
            ) {
                IconButton(
                    enabled = canSend,
                    onClick = {
                        val prompt = input.trim()
                        if (prompt.isEmpty() || model == null || tokenizer == null) return@IconButton
                        messages += Msg(true, prompt)
                        input = ""
                        busy = true
                        val assistant = Msg(false, "")
                        messages += assistant
                        scope.launch {
                            try {
                                val answer = withContext(Dispatchers.Default) {
                                    Inference.generate(
                                        model,
                                        tokenizer,
                                        prompt,
                                        Inference.GenConfig(maxNewTokens = maxTokens, temperature = temperature, topK = topK)
                                    ) { prefix ->
                                        // The inference engine sends the decoded prefix for UTF-8 safety.
                                        assistant.text = prefix
                                    }
                                }
                                assistant.text = answer.ifBlank { "Model tidak menghasilkan token untuk prompt ini." }
                            } catch (e: Exception) {
                                assistant.text = "Gagal menghasilkan jawaban: " + (e.message ?: e::class.java.simpleName)
                            } finally {
                                busy = false
                            }
                        }
                    },
                    modifier = Modifier.fillMaxSize()
                ) {
                    Icon(
                        Icons.AutoMirrored.Filled.Send,
                        contentDescription = "Kirim pesan",
                        tint = if (canSend) Color.Black else NexoraMuted
                    )
                }
            }
        }

        if (model == null || tokenizer == null) {
            Text(
                "Mode chat aktif setelah model selesai dilatih atau checkpoint dimuat.",
                modifier = Modifier.padding(bottom = 8.dp),
                color = NexoraMuted,
                fontSize = 11.sp
            )
        } else if (busy) {
            Text(
                "Model sedang menghasilkan jawaban…",
                modifier = Modifier.padding(bottom = 8.dp),
                color = NexoraMuted,
                fontSize = 11.sp
            )
        } else {
            Text(
                "Berjalan lokal di perangkat • tanpa API cloud",
                modifier = Modifier.padding(bottom = 8.dp),
                color = NexoraMuted,
                fontSize = 11.sp
            )
        }
    }
}
