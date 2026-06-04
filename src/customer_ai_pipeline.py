from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from transformers import pipeline


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CustomerSupportPipeline:
    def __init__(self) -> None:
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
        )

        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="uer/roberta-base-finetuned-jd-binary-chinese",
        )

        self.entity_extractor = pipeline(
            "token-classification",
            model="ckiplab/bert-base-chinese-ner",
            aggregation_strategy="simple",
        )

        self.summarizer = pipeline(
            "summarization",
            model="csebuetnlp/mT5_multilingual_XLSum",
        )

    def run(self, message: str) -> PipelineResult:
        safe_message = self._truncate(message)

        intent_raw = self.intent_classifier(
            safe_message,
            candidate_labels=DEFAULT_LABELS,
            multi_label=False,
        )

        intent = {
            "label": intent_raw["labels"][0],
            "score": float(intent_raw["scores"][0]),
            "ranking": [
                {"label": label, "score": float(score)}
                for label, score in zip(intent_raw["labels"], intent_raw["scores"])
            ],
        }

        sentiment_raw = self.sentiment_analyzer(safe_message)[0]
        sentiment = self._format_sentiment(sentiment_raw)

        raw_entities = self.entity_extractor(safe_message)
        entities = self._clean_entities(raw_entities)

        order_info = self._extract_order_info(message)

        summary = self._summarize(safe_message)

        reply = self._generate_reply(
            intent=intent["label"],
            sentiment=sentiment["label"],
            summary=summary,
            order_info=order_info,
        )

        return PipelineResult(
            intent=intent,
            sentiment=sentiment,
            entities=entities,
            order_info=order_info,
            summary=summary,
            reply=reply,
        )

    def _truncate(self, text: str, max_chars: int = 500) -> str:
        return text[:max_chars]

    def _format_sentiment(self, sentiment_result: dict[str, Any]) -> dict[str, Any]:
        label = str(sentiment_result.get("label", ""))
        score = float(sentiment_result.get("score", 0))

        if label.upper() in ["POSITIVE", "LABEL_1"]:
            label_zh = "正向"
        elif label.upper() in ["NEGATIVE", "LABEL_0"]:
            label_zh = "負向"
        else:
            label_zh = label

        return {
            "label": label_zh,
            "score": score,
            "raw_label": label,
        }

    def _clean_entities(self, raw_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entities = []

        for ent in raw_entities:
            word = str(ent.get("word", "")).replace(" ", "")
            entity_type = ent.get("entity_group", "")
            score = round(float(ent.get("score", 0)), 4)

            if word:
                entities.append(
                    {
                        "entity": entity_type,
                        "word": word,
                        "score": score,
                    }
                )

        return entities

    def _extract_order_info(self, message: str) -> dict[str, Any]:
        date_pattern = r"\d{4}/\d{2}/\d{2}"
        order_pattern = r"[A-Z]{2}-\d{6}"

        dates = re.findall(date_pattern, message)
        orders = re.findall(order_pattern, message)

        product = None
        product_match = re.search(r"購買了\s*(.*?)，", message)

        if product_match:
            product = product_match.group(1)

        return {
            "dates": dates,
            "orders": orders,
            "product": product,
        }

    def _summarize(self, message: str) -> str:
        try:
            summary_input = f"摘要以下客服信件重點：{message}"

            summary = self.summarizer(
                summary_input,
                max_length=80,
                min_length=20,
                do_sample=False,
            )[0]["summary_text"]

            if len(summary.strip()) < 5:
                return self._fallback_summary(message)

            return summary

        except Exception:
            return self._fallback_summary(message)

    def _fallback_summary(self, message: str, max_len: int = 100) -> str:
        clean_message = message.replace("\n", " ").strip()
        return clean_message[:max_len] + "..."

    def _generate_reply(
        self,
        intent: str,
        sentiment: str,
        summary: str,
        order_info: dict[str, Any],
    ) -> str:
        order_text = "、".join(order_info.get("orders", [])) or "未提供"
        product_text = order_info.get("product") or "商品"

        if intent == "退貨退款":
            action = "我們會協助您確認退貨退款流程，請您提供商品照片與故障狀況說明，以便後續處理。信用卡退款時間會依銀行作業流程而定。"
        elif intent == "物流配送":
            action = "我們會協助您查詢配送狀態，請您提供訂單編號或收件資訊，以便確認目前物流進度。"
        elif intent == "帳務付款":
            action = "我們會協助您確認付款與帳務狀況，請您提供付款時間與付款方式，以利後續查詢。"
        elif intent == "產品技術問題":
            action = "我們會協助您排查產品問題，請您提供產品型號、故障狀況與相關照片或影片。"
        else:
            action = "我們已收到您的詢問，會盡快協助確認並回覆您後續處理方式。"

        return f"""
您好，感謝您的來信。

很抱歉讓您有不佳的體驗。根據您的描述，您的問題主要屬於「{intent}」，目前系統判斷客戶情緒偏向「{sentiment}」。

我們已注意到您的問題重點：
{summary}

相關訂單資訊：
訂單編號：{order_text}
產品名稱：{product_text}

{action}

感謝您的耐心等候，我們會盡快協助您處理。
""".strip()


if __name__ == "__main__":
    demo_message = """
您好，我在 2026/05/29 購買了 Orbit X1 降噪耳機，訂單編號 OD-583920。
商品昨天到貨後左耳完全沒有聲音，我已經嘗試重置還是無法使用。
這是送給主管的生日禮物，現在真的很尷尬。
請協助我辦理退貨退款，並告知多久會退回信用卡。
"""

    workflow = CustomerSupportPipeline()
    result = workflow.run(demo_message)
    print(result.to_dict())