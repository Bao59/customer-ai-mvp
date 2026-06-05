from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_LABELS = [
    "退貨退款",
    "物流配送",
    "帳務付款",
    "產品技術問題",
    "一般客服詢問",
]


@dataclass
class PipelineResult:
    intent: dict[str, Any]
    sentiment: dict[str, Any]
    entities: list[dict[str, Any]]
    order_info: dict[str, Any]
    summary: str
    reply: str
    suggested_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CustomerSupportPipeline:
    """
    AI 客服信件分析工作流。

    使用 Transformers Pipeline 串接：
    1. zero-shot-classification：意圖分類
    2. sentiment-analysis：情緒分析
    3. token-classification：NER 實體抽取
    4. summarization：摘要模型，失敗時使用規則式摘要
    5. text-generation：預設關閉，正式展示以模板回覆為主
    """

    def __init__(self) -> None:
        from transformers import pipeline

        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model=os.getenv("MODEL_ZERO_SHOT", "joeddav/xlm-roberta-large-xnli"),
        )

        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=os.getenv(
                "MODEL_SENTIMENT",
                "uer/roberta-base-finetuned-jd-binary-chinese",
            ),
        )

        self.entity_extractor = pipeline(
            "token-classification",
            model=os.getenv("MODEL_NER", "ckiplab/bert-base-chinese-ner"),
            aggregation_strategy="simple",
        )

        self.summarizer = None
        try:
            self.summarizer = pipeline(
                "summarization",
                model=os.getenv("MODEL_SUMMARY", "csebuetnlp/mT5_multilingual_XLSum"),
            )
        except Exception as exc:
            print(f"[WARN] Summarizer load failed. Use rule-based summary. Error: {exc}")

        self.generator = None
        if os.getenv("ENABLE_GENERATION", "0") == "1":
            try:
                self.generator = pipeline(
                    "text-generation",
                    model=os.getenv(
                        "MODEL_GENERATION",
                        "uer/gpt2-chinese-cluecorpussmall",
                    ),
                )
            except Exception as exc:
                print(f"[WARN] Generator load failed. Use template reply. Error: {exc}")

    def run(self, message: str) -> PipelineResult:
        if not message or not message.strip():
            raise ValueError("message cannot be empty")

        safe_message = self._truncate(message.strip())

        intent_raw = self.intent_classifier(
            safe_message,
            candidate_labels=DEFAULT_LABELS,
            multi_label=False,
        )

        intent = {
            "label": intent_raw["labels"][0],
            "score": round(float(intent_raw["scores"][0]), 4),
            "ranking": [
                {
                    "label": label,
                    "score": round(float(score), 4),
                }
                for label, score in zip(intent_raw["labels"], intent_raw["scores"])
            ],
        }

        sentiment_raw = self.sentiment_analyzer(safe_message)[0]
        sentiment = self._format_sentiment(sentiment_raw)

        raw_entities = self.entity_extractor(safe_message)
        entities = self._clean_entities(raw_entities)

        order_info = self._extract_order_info(safe_message)

        summary = self._summarize(
            message=safe_message,
            order_info=order_info,
        )

        suggested_action = self._get_suggested_action(intent["label"])

        reply = self._generate_reply(
            intent=intent["label"],
            sentiment=sentiment["label"],
            summary=summary,
            order_info=order_info,
            suggested_action=suggested_action,
            message=safe_message,
        )

        return PipelineResult(
            intent=intent,
            sentiment=sentiment,
            entities=entities,
            order_info=order_info,
            summary=summary,
            reply=reply,
            suggested_action=suggested_action,
        )

    def _truncate(self, text: str, max_chars: int = 700) -> str:
        return text[:max_chars]

    def _format_sentiment(self, sentiment_result: dict[str, Any]) -> dict[str, Any]:
        raw_label = str(sentiment_result.get("label", ""))
        score = round(float(sentiment_result.get("score", 0)), 4)

        label_lower = raw_label.lower()

        if "negative" in label_lower or raw_label.upper() in ["NEGATIVE", "LABEL_0"]:
            label_zh = "負向"
        elif "positive" in label_lower or raw_label.upper() in ["POSITIVE", "LABEL_1"]:
            label_zh = "正向"
        else:
            label_zh = "中性"

        return {
            "label": label_zh,
            "score": score,
            "raw_label": raw_label,
        }

    def _clean_entities(self, raw_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []

        for ent in raw_entities:
            word = str(ent.get("word", "")).replace(" ", "")
            entity_type = str(ent.get("entity_group", ""))
            score = round(float(ent.get("score", 0)), 4)

            if not word:
                continue

            entities.append(
                {
                    "type": entity_type,
                    "text": word,
                    "score": score,
                }
            )

        return entities

    def _extract_order_info(self, message: str) -> dict[str, Any]:
        date_pattern = r"\d{4}[/-]\d{2}[/-]\d{2}"
        order_pattern = r"[A-Z]{2}-?\d{6}"
        email_pattern = r"[\w.-]+@[\w.-]+\.\w+"
        invoice_pattern = r"INV-\d+"
        amount_pattern = r"NT\$\d[\d,]*"

        dates = re.findall(date_pattern, message)
        orders = re.findall(order_pattern, message)
        emails = re.findall(email_pattern, message)
        invoices = re.findall(invoice_pattern, message)
        amounts = re.findall(amount_pattern, message)

        product = None

        product_patterns = [
            r"購買了\s*(.*?)(?:，|,|。|\n)",
            r"購買\s*(.*?)(?:，|,|。|\n)",
            r"收到\s*(.*?)(?:，|,|。|\n)",
            r"商品(?:名稱)?[:：]\s*(.*?)(?:，|,|。|\n)",
        ]

        for pattern in product_patterns:
            match = re.search(pattern, message)
            if match:
                product = match.group(1).strip()
                break

        if product is None:
            known_products = ["Orbit X1 降噪耳機", "NexPad Pro 平板", "Premium 方案"]
            for item in known_products:
                if item in message:
                    product = item
                    break

        return {
            "dates": dates,
            "orders": orders,
            "emails": emails,
            "invoices": invoices,
            "amounts": amounts,
            "product": product,
        }

    def _summarize(self, message: str, order_info: dict[str, Any]) -> str:
        """
        正式網站版以規則式摘要為主，避免小模型摘要偏題。
        摘要模型保留載入，但不直接覆蓋穩定摘要。
        """
        return self._rule_based_summary(message, order_info)

    def _rule_based_summary(self, message: str, order_info: dict[str, Any]) -> str:
        product = order_info.get("product") or "商品"

        orders = order_info.get("orders", [])
        invoices = order_info.get("invoices", [])
        order_text = orders[0] if orders else invoices[0] if invoices else "未提供編號"

        issue_parts: list[str] = []

        if "左耳" in message and ("沒有聲音" in message or "無聲" in message):
            issue_parts.append("到貨後左耳沒有聲音")

        if "右耳" in message and ("沒有聲音" in message or "無聲" in message):
            issue_parts.append("到貨後右耳沒有聲音")

        if "重置" in message:
            issue_parts.append("已嘗試重置但問題仍未解決")

        if "退貨" in message or "退款" in message:
            issue_parts.append("希望辦理退貨退款")

        if "信用卡" in message and "退款" in message:
            issue_parts.append("想了解信用卡退款時間")

        if "物流" in message or "配送" in message or "送達" in message or "轉運" in message:
            issue_parts.append("想確認配送進度與預計到貨時間")

        if "扣款" in message or "帳單" in message or "發票" in message or "多收" in message:
            issue_parts.append("想確認帳務付款或重複扣款問題")

        if "出差" in message or "生日" in message or "主管" in message:
            issue_parts.append("案件帶有時間壓力或情境壓力")

        if not issue_parts:
            issue_parts.append("提出客服協助需求")

        issue_text = "，".join(dict.fromkeys(issue_parts))

        return f"客戶詢問與 {product} 相關的售後問題，案件編號為 {order_text}。主要重點為：{issue_text}。"

    def _get_suggested_action(self, intent: str) -> str:
        if intent == "退貨退款":
            return "建議客服確認訂單狀態、商品故障證明與退貨退款流程，並告知信用卡退款需依銀行作業時間。"

        if intent == "物流配送":
            return "建議客服確認物流節點、配送狀態與預計到貨時間，若有時效壓力應優先追蹤。"

        if intent == "帳務付款":
            return "建議客服核對付款紀錄、發票資料與銀行扣款狀態，若確認重複扣款應協助退款。"

        if intent == "產品技術問題":
            return "建議客服請客戶提供產品型號、故障描述、照片或影片，必要時轉交技術支援判斷。"

        return "建議客服先確認客戶需求，補齊必要資訊後再進行後續處理。"

    def _generate_reply(
        self,
        intent: str,
        sentiment: str,
        summary: str,
        order_info: dict[str, Any],
        suggested_action: str,
        message: str,
    ) -> str:
        if self.generator is not None:
            generated = self._generate_reply_by_model(
                intent=intent,
                sentiment=sentiment,
                summary=summary,
                message=message,
            )

            if generated:
                return generated

        return self._generate_reply_by_template(
            intent=intent,
            sentiment=sentiment,
            summary=summary,
            order_info=order_info,
            suggested_action=suggested_action,
        )

    def _generate_reply_by_template(
        self,
        intent: str,
        sentiment: str,
        summary: str,
        order_info: dict[str, Any],
        suggested_action: str,
    ) -> str:
        orders = order_info.get("orders", [])
        invoices = order_info.get("invoices", [])
        product = order_info.get("product") or "商品"

        reference = orders[0] if orders else invoices[0] if invoices else "您提供的資訊"

        if sentiment == "負向":
            opening = "很抱歉讓您遇到這次不便，我們會優先協助確認並處理。"
        else:
            opening = "感謝您的來信，我們已收到您的需求。"

        return f"""您好：

{opening}

根據您的描述，您的問題主要屬於「{intent}」。
目前整理重點如下：
{summary}

相關資訊：
- 參考編號：{reference}
- 商品 / 服務：{product}

處理建議：
{suggested_action}

為了加速處理，建議您提供完整訂單資訊、相關照片或付款紀錄截圖。我們預計於 1 個工作天內回覆處理進度。

謝謝您的耐心等候。"""

    def _generate_reply_by_model(
        self,
        intent: str,
        sentiment: str,
        summary: str,
        message: str,
    ) -> str | None:
        prompt = (
            "請以客服語氣回覆以下客戶，需包含道歉、處理方式、預計回覆時間。\n"
            f"客服信件分類：{intent}\n"
            f"客戶情緒：{sentiment}\n"
            f"客戶訊息：{message}\n"
            f"摘要：{summary}\n"
            "客服回覆："
        )

        try:
            output = self.generator(
                prompt,
                max_new_tokens=160,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.generator.tokenizer.eos_token_id,
            )[0]["generated_text"]

            reply = output.split("客服回覆：")[-1].strip()

            if len(reply) < 20:
                return None

            return reply

        except Exception:
            return None


_pipeline_instance: CustomerSupportPipeline | None = None


def get_pipeline() -> CustomerSupportPipeline:
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = CustomerSupportPipeline()

    return _pipeline_instance


if __name__ == "__main__":
    demo_message = (
        "您好，我在 2026/05/29 購買了 Orbit X1 降噪耳機，訂單編號 OD-583920。"
        "商品昨天到貨後左耳完全沒有聲音，我已經嘗試重置還是無法使用。"
        "這是送給主管的生日禮物，現在真的很尷尬。"
        "請協助我辦理退貨退款，並告知多久會退回信用卡。"
    )

    workflow = get_pipeline()
    result = workflow.run(demo_message)
    print(result.to_dict())