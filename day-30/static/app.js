"use strict";
const $ = (id) => document.getElementById(id);
let selected = null, currentJob = null, pollTimer = null, generation = 0;

async function api(path, options = {}) {
  const response = await fetch(path, {credentials: "same-origin", ...options,
    headers: {"Content-Type": "application/json", ...options.headers}});
  let data;
  try { data = await response.json(); }
  catch (_) {
    throw new Error(response.status === 429 ? "Слишком много запросов. Подождите немного."
      : "Сервер временно недоступен. Попробуйте ещё раз.");
  }
  if (!response.ok) {
    if (response.status === 401 && path !== "/api/login") showLogin();
    const detail = data.detail;
    const error = new Error(typeof detail === "string" ? detail : detail?.message || "Ошибка запроса.");
    error.detail = detail;
    throw error;
  }
  return data;
}

function showLogin() {
  generation++;
  clearTimeout(pollTimer);
  selected = currentJob = null;
  $("chat-screen").hidden = true;
  $("login-screen").hidden = false;
  $("password").value = "";
  $("messages").replaceChildren();
  $("chat-list").replaceChildren();
}

function message(role, content, pending = false) {
  const article = document.createElement("article");
  article.className = `message ${role}${pending ? " pending" : ""}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "ВЫ" : "QWEN";
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = content;
  article.append(label, body);
  $("messages").append(article);
}

function welcome() {
  const block = document.createElement("div");
  block.className = "welcome";
  const badge = document.createElement("span");
  badge.className = "badge"; badge.textContent = "QWEN 3 · 1.7B";
  const title = document.createElement("h3"); title.textContent = "Начнём с вопроса.";
  const text = document.createElement("p");
  text.textContent = "Попросите объяснить понятие, придумать идею или помочь с текстом. Запросы всех пользователей обрабатываются по очереди — её положение будет видно здесь.";
  block.append(badge, title, text); $("messages").append(block);
}

function busy(job) {
  currentJob = job;
  $("send").disabled = !!job;
  $("delete-chat").disabled = !!job;
  $("job-status").hidden = !job;
  if (job) $("job-status").textContent = job.status === "running"
    ? "Модель готовит ответ…" : `Ожидает в очереди · место ${job.position || 1}`;
}

async function listChats() {
  const chats = await api("/api/chats");
  $("chat-list").replaceChildren();
  for (const chat of chats) {
    const button = document.createElement("button");
    button.className = "chat-link" + (chat.id === selected ? " selected" : "");
    button.textContent = chat.title; button.title = chat.title;
    button.onclick = () => openChat(chat.id).catch(showError);
    $("chat-list").append(button);
  }
  return chats;
}

function showError(error) { $("chat-error").textContent = error.message; }

async function openChat(id) {
  const version = ++generation;
  clearTimeout(pollTimer);
  const chat = await api(`/api/chats/${id}`);
  if (version !== generation) return;
  selected = id;
  $("chat-error").textContent = "";
  $("message").value = "";
  $("chat-title").textContent = chat.title;
  $("delete-chat").hidden = false;
  $("messages").replaceChildren();
  if (!chat.messages.length && !chat.pending) welcome();
  for (const turn of chat.messages) message(turn.role, turn.content);
  busy(chat.pending);
  if (chat.pending) {
    message("user", chat.pending.text, true);
    $("context-meter").textContent = `Последний запрос: ${chat.pending.input_tokens} / 1760 токенов`;
    poll(chat.pending.id, version);
  } else $("context-meter").textContent = "Контекст проверяется при отправке · максимум 1760 токенов";
  await listChats();
  $("messages").scrollTop = $("messages").scrollHeight;
}

async function poll(id, version) {
  try {
    const job = await api(`/api/jobs/${id}`);
    if (version !== generation) return;
    if (job.status === "done" || job.status === "error") {
      await openChat(job.chat_id);
      $("context-meter").textContent = `Последний запрос: ${job.input_tokens} / 1760 токенов`;
      if (job.status === "error") showError(new Error(job.error));
      if (job.result?.truncated) $("job-status").textContent = "Ответ достиг лимита 256 токенов. Можно попросить продолжить.";
      if (job.result?.truncated) $("job-status").hidden = false;
      return;
    }
    busy(job);
    pollTimer = setTimeout(() => poll(id, version), 900);
  } catch (error) {
    if (version !== generation) return;
    showError(error);
    // An uncertain network result is not permission to send the message again.
    pollTimer = setTimeout(() => poll(id, version), 3000);
  }
}

async function enter() {
  const me = await api("/api/me");
  $("account-name").textContent = me.username;
  $("login-screen").hidden = true; $("chat-screen").hidden = false;
  const chats = await listChats();
  if (chats.length) await openChat(chats[0].id);
  else {
    $("messages").replaceChildren(); welcome(); busy(null);
    $("chat-title").textContent = "Новый диалог"; $("delete-chat").hidden = true;
  }
}

$("login-form").onsubmit = async (event) => {
  event.preventDefault(); $("login-button").disabled = true; $("login-error").textContent = "";
  try {
    await api("/api/login", {method: "POST", body: JSON.stringify({
      username: $("username").value, password: $("password").value})});
    $("password").value = ""; await enter();
  } catch (error) { $("login-error").textContent = error.message; }
  finally { $("login-button").disabled = false; }
};

$("new-chat").onclick = async () => {
  try { const chat = await api("/api/chats", {method: "POST", body: "{}"}); await openChat(chat.id); }
  catch (error) { showError(error); }
};

$("delete-chat").onclick = async () => {
  if (!selected || !confirm("Удалить этот диалог и его сообщения?")) return;
  try { await api(`/api/chats/${selected}`, {method: "DELETE"}); selected = null; await enter(); }
  catch (error) { showError(error); }
};

$("logout").onclick = async () => {
  try { await api("/api/logout", {method: "POST", body: "{}"}); showLogin(); }
  catch (error) { showError(error); }
};

$("message-form").onsubmit = async (event) => {
  event.preventDefault(); if (currentJob) return;
  const text = $("message").value.trim(); if (!text) return;
  $("send").disabled = true; $("chat-error").textContent = "";
  try {
    if (!selected) selected = (await api("/api/chats", {method: "POST", body: "{}"})).id;
    await api(`/api/chats/${selected}/messages`, {method: "POST", body: JSON.stringify({text})});
    await openChat(selected);
  } catch (error) {
    showError(error);
    if (error.detail?.code === "context_limit") $("context-meter").textContent =
      `Не помещается: ${error.detail.input_tokens} / ${error.detail.input_limit} токенов`;
    // Restore server state after a lost submission response to avoid a duplicate.
    if (selected) {
      try { const chat = await api(`/api/chats/${selected}`);
        if (chat.pending) await openChat(selected);
      } catch (_) { /* Keep the visible original error. */ }
    }
  } finally { $("send").disabled = !!currentJob; }
};

$("message").onkeydown = (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault(); if (!$("send").disabled) $("message-form").requestSubmit();
  }
};
enter().catch(() => showLogin());
