const samples = {
  refund:
    "您好，我在 2026/05/29 購買了 Orbit X1 降噪耳機，訂單編號 OD-583920。商品昨天到貨後左耳完全沒有聲音，我已經嘗試重置還是無法使用。這是送給主管的生日禮物，現在真的很尷尬。請協助我辦理退貨退款，並告知多久會退回信用卡。",
  delivery:
    "客服您好，我的訂單 OD-104482 原本預計 2026/06/02 送達，但物流狀態已經三天停在桃園轉運中心。我下週一要出差，需要在 2026/06/07 前收到 NexPad Pro 平板。麻煩協助確認配送進度，謝謝。",
  billing:
    "你好，我是 Premium 方案用戶。我的帳號 email 是 mina.chen@example.com，6 月信用卡帳單被扣了兩次 NT$1,280，但後台只顯示一筆發票 INV-88219。請幫我確認是否重複扣款，並退款多收的金額。"
};

const urgentWords = ["主管", "下週一", "出差", "生日", "三天", "重複扣款", "無法使用", "前收到"];

const input = document.querySelector("#messageInput");
const runButton = document.querySelector("#runPipeline");
const copyButton = document.querySelector("#copyReply");
const copyState = document.querySelector("#copyState");
const segmentButtons = [...document.querySelectorAll(".segment")];
const stepButtons = [...document.querySelectorAll(".pipeline-step")];

function setActiveStep(stepName) {
  stepButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.step === stepName);
  });
}

function animatePipelineSteps() {
  const steps = ["intent", "sentiment", "entities", "summary", "reply"];
  steps.forEach((step, index) => {
    window.setTimeout(() => setActiveStep(step), index * 140);
  });
}

function decidePriority(text, sentimentLabel, intentLabel) {
  const urgentHits = urgentWords.filter((word) => text.includes(word));

  if (sentimentLabel === "負向" && urgentHits.length >= 1) {
    return {
      label: "P1",
      reason: `客戶情緒負向，且包含「${urgentHits[0]}」等時效或壓力線索，建議優先處理。`
    };
  }

  if (intentLabel === "退貨退款" || intentLabel === "帳務付款") {
    return {
      label: "P2",
      reason: "案件涉及退款、扣款或金流問題，建議在當日內回覆。"
    };
  }

  if (sentimentLabel === "負向") {
    return {
      label: "P2",
      reason: "客戶情緒偏負向，建議盡快回覆並補齊處理資訊。"
    };
  }

  return {
    label: "P3",
    reason: "目前風險較低，可依一般客服流程處理。"
  };
}

function renderEntities(orderInfo, entities) {
  const list = document.querySelector("#entityList");
  list.innerHTML = "";

  const displayItems = [];

  if (orderInfo?.product) {
    displayItems.push({
      type: "商品",
      value: orderInfo.product
    });
  }

  if (orderInfo?.orders?.length) {
    displayItems.push({
      type: "訂單",
      value: orderInfo.orders.join("、")
    });
  }

  if (orderInfo?.dates?.length) {
    displayItems.push({
      type: "日期",
      value: orderInfo.dates.join("、")
    });
  }

  if (orderInfo?.emails?.length) {
    displayItems.push({
      type: "Email",
      value: orderInfo.emails.join("、")
    });
  }

  if (orderInfo?.invoices?.length) {
    displayItems.push({
      type: "發票",
      value: orderInfo.invoices.join("、")
    });
  }

  if (orderInfo?.amounts?.length) {
    displayItems.push({
      type: "金額",
      value: orderInfo.amounts.join("、")
    });
  }

  if (Array.isArray(entities)) {
    entities.slice(0, 8).forEach((entity) => {
      displayItems.push({
        type: entity.type || entity.entity || "ENTITY",
        value: entity.text || entity.word || "-"
      });
    });
  }

  if (displayItems.length === 0) {
    const item = document.createElement("span");
    item.className = "entity";
    item.innerHTML = `<b>狀態</b><span>未偵測到明確實體</span>`;
    list.appendChild(item);
    return;
  }

  displayItems.forEach((entity) => {
    const item = document.createElement("span");
    item.className = "entity";
    item.innerHTML = `<b>${escapeHtml(entity.type)}</b><span>${escapeHtml(entity.value)}</span>`;
    list.appendChild(item);
  });
}

function renderResult(result, processingTime, originalText) {
  const intent = result.intent || {};
  const sentiment = result.sentiment || {};
  const orderInfo = result.order_info || {};
  const entities = result.entities || [];

  const intentLabel = intent.label || "一般客服";
  const intentScore = Number(intent.score || 0);
  const sentimentLabel = sentiment.label || "中性";
  const sentimentScore = Number(sentiment.score || 0);

  document.querySelector("#processingTime").textContent = `${processingTime}s`;

  document.querySelector("#intentLabel").textContent = intentLabel;
  document.querySelector("#intentReason").textContent =
    `後端 Transformers Pipeline 判斷此信件最接近「${intentLabel}」，信心分數為 ${intentScore.toFixed(4)}。`;
  document.querySelector("#intentBar").style.width = `${Math.round(intentScore * 100)}%`;

  document.querySelector("#sentimentLabel").textContent = sentimentLabel;
  document.querySelector("#sentimentReason").textContent =
    `情緒模型判斷客戶情緒為「${sentimentLabel}」，信心分數為 ${sentimentScore.toFixed(4)}。`;

  const priority = decidePriority(originalText, sentimentLabel, intentLabel);
  document.querySelector("#priorityLabel").textContent = priority.label;
  document.querySelector("#priorityReason").textContent = priority.reason;

  renderEntities(orderInfo, entities);

  document.querySelector("#summaryText").textContent = result.summary || "尚無摘要。";
  document.querySelector("#replyText").textContent = result.reply || "尚無回覆草稿。";
}

async function analyze() {
  const text = input.value.trim();

  if (!text) {
    alert("請先輸入客服信件內容。");
    return;
  }

  runButton.classList.add("is-running");
  runButton.disabled = true;
  copyState.textContent = "";
  runButton.innerHTML = "分析中...";

  animatePipelineSteps();

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: text
      })
    });

    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "後端分析失敗");
    }

    renderResult(data.result, data.processing_time_seconds, text);
  } catch (error) {
    console.error(error);
    alert(`分析失敗：${error.message}`);
  } finally {
    runButton.disabled = false;
    runButton.classList.remove("is-running");
    runButton.innerHTML = `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
      執行分析
    `;
  }
}

segmentButtons.forEach((button) => {
  button.addEventListener("click", () => {
    segmentButtons.forEach((item) => {
      item.classList.toggle("is-selected", item === button);
      item.setAttribute("aria-selected", item === button ? "true" : "false");
    });

    input.value = samples[button.dataset.sample];
  });
});

runButton.addEventListener("click", analyze);

copyButton.addEventListener("click", async () => {
  const text = document.querySelector("#replyText").textContent;

  if (!text.trim()) {
    copyState.textContent = "尚無回覆可複製";
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    copyState.textContent = "已複製";
    window.setTimeout(() => {
      copyState.textContent = "";
    }, 1400);
  } catch {
    copyState.textContent = "複製失敗";
  }
});

stepButtons.forEach((button) => {
  button.addEventListener("click", () => setActiveStep(button.dataset.step));
});

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

input.value = samples.refund;