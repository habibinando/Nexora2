package com.Nexora.ai.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.Nexora.ai.ml.Inference
import com.Nexora.ai.train.TrainingController
import com.Nexora.ai.ui.components.GradientLabel
import com.Nexora.ai.ui.theme.NexoraMuted
import com.Nexora.ai.ui.theme.NexoraPrimary
import com.Nexora.ai.ui.theme.NexoraSurfaceVar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private class Msg(val fromUser: Boolean, text: String) {
    var text by mutableStateOf(text)
}

@Composable
fun ChatScreen() {
    val s by TrainingController.state.collectAsState()
    val model = TrainingController.model
    val tokenizer = TrainingController.tokenizer
    val messages = remember { mutableStateListOf<Msg>() }
    var input by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()

    Column(Modifier.fillMaxSize().padding(16.dp)) {
        GradientLabel("Chat dengan model Anda")
        Text(
            if (model == null || tokenizer == null)
                "Belum ada model terlatih. Impor dataset lalu latih model di tab Train — chat ini memakai bobot hasil latihan nyata, bukan mock."
            else "Model aktif: ${s.totalParams / 1_000_000}M parameter · step ${s.step}. Generasi token-per-token secara real-time.",
            color = NexoraMuted, fontSize = 12.sp, modifier = Modifier.padding(bottom = 8.dp)
        )

        LazyColumn(Modifier.weight(1f), state = listState, verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(messages) { m ->
                Box(Modifier.fillMaxWidth(), contentAlignment = if (m.fromUser) Alignment.CenterEnd else Alignment.CenterStart) {
                    Text(
                        m.text + if (!m.fromUser && busy && m === messages.lastOrNull()) "▍" else "",
                        modifier = Modifier
                            .clip(RoundedCornerShape(16.dp))
                            .background(if (m.fromUser) NexoraPrimary else NexoraSurfaceVar)
                            .padding(12.dp),
                        color = if (m.fromUser) Color.Black else Color(0xFFE6EDF3),
                        fontSize = 14.sp
                    )
                }
            }
        }

        Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = input, onValueChange = { input = it },
                modifier = Modifier.weight(1f), placeholder = { Text("Ketik pesan…", color = NexoraMuted) },
                enabled = !busy && model != null
            )
            Button(
                onClick = {
                    val prompt = input.trim(); if (prompt.isEmpty() || model == null || tokenizer == null) return@Button
                    messages += Msg(true, prompt); input = ""; busy = true
                    val assistant = Msg(false, ""); messages += assistant
                    scope.launch {
                        withContext(Dispatchers.Default) {
                            Inference.generate(model, tokenizer, prompt, Inference.GenConfig(maxNewTokens = 80)) { tok ->
                                assistant.text += tok
                            }
                        }
                        busy = false
                    }
                },
                enabled = !busy && model != null,
                colors = ButtonDefaults.buttonColors(containerColor = NexoraPrimary, contentColor = Color.Black)
            ) { Text("Kirim", fontWeight = FontWeight.Bold) }
        }
    }
}
