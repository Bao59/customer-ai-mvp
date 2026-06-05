# 專案進度紀錄

## 已完成

- 確認專題題目：AI 客服信件分析與回覆草稿系統
- 使用 Google Colab 建立 Transformers Pipeline 工作流
- 串接 zero-shot-classification、sentiment-analysis、token-classification / NER、summarization
- 完成測試案例
- 完成專題報告 Word 檔

## 遇到的問題

- 中文 NER 對訂單編號與產品型號辨識不穩
- 摘要模型在短文本上的效果有限

## 解決方式

- 加入規則式資訊抽取，補強日期、訂單編號、產品名稱
- 摘要失敗時使用簡單文字截斷作為備援

## 下一步

- 整理 GitHub README
- 補上 Colab 執行截圖
- 後續改成 Streamlit 網頁版