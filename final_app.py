# final_app.py - 完整可运行的版本
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re

# 设置页面
st.set_page_config(page_title="微博情感分析系统", layout="wide")
st.title("📊 微博情感分析系统")
st.markdown("---")

# 先检查scikit-learn是否可用
try:
    import sklearn
    st.sidebar.success(f"✅ scikit-learn {sklearn.__version__}")
except ImportError:
    st.sidebar.error("❌ scikit-learn未安装")

# 加载模型 - 先尝试小模型
@st.cache_resource
def load_model():
    try:
        # 先尝试小模型（620KB）
        with open('naive_bayes_best_model.pkl', 'rb') as f:
            model_info = pickle.load(f)
        st.sidebar.success("✅ 基础模型加载成功")
        return model_info
    except:
        try:
            # 再尝试大模型
            with open('adaboost_nb_best_model.pkl', 'rb') as f:
                model_info = pickle.load(f)
            st.sidebar.success("✅ AdaBoost模型加载成功")
            return model_info
        except Exception as e:
            st.error(f"❌ 所有模型加载失败: {e}")
            return None

# 显示加载状态
with st.spinner("正在加载模型..."):
    model_info = load_model()

if model_info is None:
    st.error("""
    ## 模型加载失败
    
    可能原因：
    1. 模型文件不存在
    2. 依赖包版本不匹配
    3. 内存不足
    
    **解决方案：**
    1. 确保 `.pkl` 文件在同一个目录
    2. 检查 requirements.txt 是否正确
    3. 尝试本地运行
    """)
    st.stop()

# 获取模型和向量化器
model = model_info.get('model')
vectorizer = model_info.get('vectorizer')

if model is None or vectorizer is None:
    st.error("模型结构不完整")
    st.stop()

# 标签映射
LABELS = {0: "客观", 1: "积极", 2: "消极"}

# ==================== 主界面 ====================
tab1, tab2, tab3 = st.tabs(["📝 分析", "📊 统计", "ℹ️ 关于"])

with tab1:
    st.header("文本情感分析")
    
    # 单条分析
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "输入微博内容：",
            "今天天气真好，心情愉快！",
            height=120
        )
        
        if st.button("🚀 分析情感", type="primary"):
            # 简单预处理
            text_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', '', user_input)
            
            # 预测
            try:
                features = vectorizer.transform([text_clean])
                pred = model.predict(features)[0]
                sentiment = LABELS.get(pred, "未知")
                
                # 显示结果
                st.markdown("---")
                st.subheader("分析结果")
                
                if sentiment == "积极":
                    st.success(f"✅ **情感：{sentiment}**")
                elif sentiment == "消极":
                    st.error(f"❌ **情感：{sentiment}**")
                else:
                    st.info(f"📄 **情感：{sentiment}**")
                
                # 显示原始文本
                with st.expander("查看处理后的文本"):
                    st.code(text_clean)
                    
            except Exception as e:
                st.error(f"预测失败: {e}")
    
    with col2:
        st.subheader("快速测试")
        test_texts = [
            "太棒了！",
            "很失望。",
            "普通。"
        ]
        
        for text in test_texts:
            if st.button(text, key=text):
                st.session_state.test_text = text

with tab2:
    st.header("统计分析")
    
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # 批量分析
    st.subheader("批量分析")
    batch_input = st.text_area(
        "输入多条文本（每行一条）",
        "今天很开心\n这个产品很糟糕\n天气不错",
        height=150
    )
    
    if st.button("📥 分析所有文本"):
        texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
        results = []
        
        with st.spinner(f"正在分析 {len(texts)} 条文本..."):
            for text in texts:
                try:
                    text_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
                    features = vectorizer.transform([text_clean])
                    pred = model.predict(features)[0]
                    sentiment = LABELS.get(pred, "未知")
                    results.append({
                        'text': text[:50] + "..." if len(text) > 50 else text,
                        'sentiment': sentiment
                    })
                except:
                    results.append({
                        'text': text[:50] + "..." if len(text) > 50 else text,
                        'sentiment': "错误"
                    })
        
        st.session_state.results = results
        
        # 显示结果表格
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # 简单统计
            st.subheader("统计摘要")
            sentiment_counts = df['sentiment'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("积极", sentiment_counts.get("积极", 0))
            with col2:
                st.metric("消极", sentiment_counts.get("消极", 0))
            with col3:
                st.metric("客观", sentiment_counts.get("客观", 0))

with tab3:
    st.header("关于系统")
    
    st.markdown("""
    ## 📋 项目信息
    
    **选题：** 基于机器学习的社交媒体情感分析
    
    **目标：** 分析微博文本的情感倾向，分为积极、消极、客观三类
    
    ## 🛠️ 技术架构
    
    - **算法：** 朴素贝叶斯 + AdaBoost增强
    - **准确率：** 94%（10万条数据集）
    - **框架：** Streamlit交互式Web应用
    - **部署：** Streamlit Cloud
    
    ## 📈 应用场景
    
    1. **品牌声誉监测**
       - 分析社交媒体对品牌的评价倾向
       - 及时发现负面舆情
    
    2. **舆情分析**
       - 监测热点事件的公众情感
       - 支持决策制定
    
    3. **用户反馈分析**
       - 分析产品评论的情感分布
       - 了解用户满意度
    
    4. **市场调研**
       - 了解消费者情感倾向
       - 支持产品改进
    """)
    
    st.markdown("---")
    st.caption("人工智能导论大作业 · 微博情感分析系统")

# ==================== 侧边栏信息 ====================
st.sidebar.header("系统状态")
st.sidebar.info(f"""
**模型：** {'AdaBoost增强' if 'adaboost' in str(model).lower() else '朴素贝叶斯'}
**特征维度：** {vectorizer.vocabulary_.__len__() if hasattr(vectorizer, 'vocabulary_') else '未知'}
**分类：** 积极/消极/客观
""")

st.sidebar.header("使用说明")
st.sidebar.markdown("""
1. 在"分析"标签页输入文本
2. 点击"分析情感"按钮
3. 查看分析结果
4. 可在"统计"标签页进行批量分析
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2024 人工智能导论课程项目")
