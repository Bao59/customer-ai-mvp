import streamlit as st

from src.customer_ai_pipeline import CustomerSupportPipeline


st.set_page_config(
    page_title="AI 客服信件分析系統",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def load_workflow():
    return CustomerSupportPipeline()


st.title("🤖 AI 客服信件分析與回覆草稿系統")

st.write(
    "本系統使用 Hugging Face Transformers Pipeline 串接意圖分類、情緒分析、NER 實體抽取與摘要模型，"
    "協助客服人員快速分析客戶信件並產生初步回覆草稿。"
)

sample_message = """
您好，我在 2026/05/29 購買了 Orbit X1 降噪耳機，訂單編號 OD-583920。
商品昨天到貨後左耳完全沒有聲音，我已經嘗試重置還是無法使用。
這是送給主管的生日禮物，現在真的很尷尬。
請協助我辦理退貨退款，並告知多久會退回信用卡。
"""

message = st.text_area(
    "請輸入客服信件內容",
    value=sample_message,
    height=220,
)

analyze_button = st.button("開始分析", type="primary")

if analyze_button:
    if not message.strip():
        st.warning("請先輸入客服信件內容。")
        st.stop()

    with st.spinner("模型分析中，第一次執行會花比較久時間..."):
        workflow = load_workflow()
        result = workflow.run(message)

    st.success("分析完成")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. 意圖分類")
        st.metric(
            label="預測類別",
            value=result.intent["label"],
            delta=f'信心分數 {result.intent["score"]:.4f}',
        )

        st.write("完整分類排序")
        st.dataframe(result.intent["ranking"], use_container_width=True)

    with col2:
        st.subheader("2. 情緒分析")
        st.metric(
            label="情緒判斷",
            value=result.sentiment["label"],
            delta=f'信心分數 {result.sentiment["score"]:.4f}',
        )

        st.write("原始模型標籤")
        st.code(result.sentiment["raw_label"])

    st.subheader("3. NER 實體抽取")
    if result.entities:
        st.dataframe(result.entities, use_container_width=True)
    else:
        st.info("未抽取到明確實體。")

    st.subheader("4. 訂單資訊抽取")
    st.json(result.order_info)

    st.subheader("5. 信件摘要")
    st.info(result.summary)

    st.subheader("6. 客服回覆草稿")
    st.text_area(
        "系統產生的回覆草稿",
        value=result.reply,
        height=260,
    )