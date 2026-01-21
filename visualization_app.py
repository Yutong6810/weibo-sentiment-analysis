import streamlit as st
import pickle
import jieba
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ========== 中文字体显示 ==========
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置页面
st.set_page_config(page_title="情感分析可视化系统", layout="wide")
st.title("📊 微博情感分析可视化系统")
st.markdown("---")

# 初始化session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'results' not in st.session_state:
    st.session_state.results = []

# 加载模型
@st.cache_resource
def load_model():
    with open('adaboost_nb_best_model.pkl', 'rb') as f:
        return pickle.load(f)

model_info = load_model()
model = model_info['model']
vectorizer = model_info['vectorizer']

# 标签映射
LABELS = {0: "Neutral", 1: "Positive", 2: "Negative"}
COLORS = {'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#2196F3'}


# ==================== 辅助函数 ====================
def analyze_text(text):
    """分析单条文本并返回结果"""
    # 预处理
    text_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    words = jieba.lcut(text_clean)
    processed = ' '.join(words)
    
    # 预测
    features = vectorizer.transform([processed])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    sentiment = LABELS.get(pred, "未知")
    confidence = proba[pred]
    
    return {
        'text': text[:100] + "..." if len(text) > 100 else text,
        'sentiment': sentiment,
        'confidence': confidence,
        'full_text': text
    }

# ==================== 侧边栏 ====================
st.sidebar.header("⚙️ 设置")
model_choice = st.sidebar.selectbox(
    "选择模型",
    ["AdaBoost增强朴素贝叶斯", "基础朴素贝叶斯"]
)

st.sidebar.header("📋 批量分析")
batch_input = st.sidebar.text_area(
    "输入多条文本（每行一条）",
    "今天很开心！\n这个产品很糟糕\n天气不错\n服务态度很好\n电影不好看",
    height=150
)

if st.sidebar.button("📥 批量分析", type="secondary"):
    texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
    
    with st.spinner(f"正在批量分析 {len(texts)} 条文本..."):
        for text in texts:
            result = analyze_text(text)
            st.session_state.history.append(text)
            st.session_state.results.append(result)
    
    st.sidebar.success(f"✅ 批量分析完成！分析了 {len(texts)} 条文本")

# 清空历史按钮
if st.sidebar.button("🗑️ 清空历史记录"):
    st.session_state.history = []
    st.session_state.results = []
    st.rerun()

# ==================== 主界面 ====================
tab1, tab2, tab3 = st.tabs(["🔤 单条分析", "📈 可视化", "📋 历史记录"])

with tab1:
    st.subheader("单条文本情感分析")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "请输入微博内容：",
            "今天天气真好，心情愉快！",
            height=120,
            key="single_input"
        )
        
        if st.button("🚀 分析情感", type="primary", key="single"):
            result = analyze_text(user_input)
            st.session_state.history.append(user_input)
            st.session_state.results.append(result)
            
            # 显示结果
            st.markdown("---")
            st.subheader("📊 分析结果")
            
            sentiment = result['sentiment']
            confidence = result['confidence']
            
            if sentiment == "积极":
                st.success(f"✅ **情感：{sentiment}** (置信度：{confidence:.2%})")
            elif sentiment == "消极":
                st.error(f"❌ **情感：{sentiment}** (置信度：{confidence:.2%})")
            else:
                st.info(f"📄 **情感：{sentiment}** (置信度：{confidence:.2%})")
    
    with col2:
        st.subheader("📊 当前分布")
        if st.session_state.results:
            # 统计当前结果的情感分布
            sentiments = [r['sentiment'] for r in st.session_state.results]
            sentiment_counts = Counter(sentiments)
            
            # 创建饼图
            fig, ax = plt.subplots(figsize=(5, 4))
            labels = list(sentiment_counts.keys())
            sizes = list(sentiment_counts.values())
            
            if labels:  # 确保有数据
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                       colors=[COLORS.get(l, '#999') for l in labels])
                ax.set_title("当前情感分布")
                st.pyplot(fig)
            else:
                st.info("暂无分析记录")
        else:
            st.info("暂无分析记录")

with tab2:
    st.header("📊 可视化分析")
    
    if not st.session_state.results:
        st.warning("暂无分析数据，请先分析一些文本")
    else:
        # 准备数据
        df = pd.DataFrame(st.session_state.results)
        
        # 显示统计摘要
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总分析数", len(df))
        with col2:
            positive = len(df[df['sentiment'] == '积极'])
            st.metric("积极", positive)
        with col3:
            negative = len(df[df['sentiment'] == '消极'])
            st.metric("消极", negative)
        with col4:
            neutral = len(df[df['sentiment'] == '客观'])
            st.metric("客观", neutral)
        
        # 图表1：情感分布饼图
        st.subheader("情感分布饼图")
        sentiment_counts = df['sentiment'].value_counts()
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=[COLORS.get(l, '#999') for l in sentiment_counts.index])
        ax1.set_title("情感分布比例")
        st.pyplot(fig1)
        
        # 图表2：情感分布柱状图
        st.subheader("情感分布柱状图")
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=[COLORS.get(l, '#999') for l in sentiment_counts.index])
        ax2.set_xlabel("情感类别")
        ax2.set_ylabel("数量")
        ax2.set_title("各类情感数量统计")
        
        # 在柱子上显示数字
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        st.pyplot(fig2)
        
        # 图表3：置信度分布
        st.subheader("置信度分布")
        
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        
        # 为每种情感创建置信度分布
        for sentiment in df['sentiment'].unique():
            data = df[df['sentiment'] == sentiment]['confidence']
            ax3.hist(data, alpha=0.5, label=sentiment, 
                    color=COLORS.get(sentiment), bins=10)
        
        ax3.set_xlabel("置信度")
        ax3.set_ylabel("频次")
        ax3.set_title("模型预测置信度分布")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        st.pyplot(fig3)
        
        # 应用场景示例
        st.subheader("📱 应用场景示例")
        
        scenario_col1, scenario_col2 = st.columns(2)
        
        with scenario_col1:
            st.markdown("#### 品牌声誉监测")
            st.markdown("""
            - **监测品牌**在社交媒体上的提及
            - **分析用户**对产品的评价倾向
            - **及时发现**负面舆情并预警
            - **跟踪**营销活动的效果
            """)
            
            # 示例品牌数据
            brand_data = pd.DataFrame({
                '情感': ['积极', '消极', '客观'],
                '数量': [positive, negative, neutral]
            })
            
            fig_brand, ax_brand = plt.subplots(figsize=(6, 4))
            ax_brand.bar(brand_data['情感'], brand_data['数量'], 
                         color=['#4CAF50', '#F44336', '#2196F3'])
            ax_brand.set_title("品牌评价情感分布")
            ax_brand.set_ylabel("评论数量")
            st.pyplot(fig_brand)
        
        with scenario_col2:
            st.markdown("#### 舆情分析")
            st.markdown("""
            - **分析热点事件**的公众情感
            - **追踪情感**随时间的变化趋势
            - **识别主要**情感驱动因素
            - **支持决策**制定和危机管理
            """)
            
            # 简单的舆情分析示例
            st.markdown("**舆情分析结果摘要：**")
            st.metric("总体积极率", f"{(positive/len(df)*100):.1f}%")
            st.metric("总体消极率", f"{(negative/len(df)*100):.1f}%")
            st.metric("平均置信度", f"{df['confidence'].mean():.2%}")

with tab3:
    st.header("📋 分析历史记录")
    
    if not st.session_state.results:
        st.info("暂无分析记录")
    else:
        # 显示详细记录
        st.subheader("详细记录")
        
        # 可排序的表格
        df_display = pd.DataFrame(st.session_state.results)
        df_display = df_display[['text', 'sentiment', 'confidence']]
        df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.2%}")
        df_display.columns = ['文本', '情感', '置信度']
        
        st.dataframe(df_display, use_container_width=True)
        
        # 导出选项
        st.markdown("---")
        st.subheader("📤 导出结果")
        
        if st.button("📄 导出为CSV文件"):
            df_export = pd.DataFrame(st.session_state.results)
            csv = df_export.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载CSV文件",
                data=csv,
                file_name="情感分析结果.csv",
                mime="text/csv"
            )

# ==================== 底部信息 ====================
st.sidebar.markdown("---")
st.sidebar.info("""
### 📊 系统信息
- **模型：** AdaBoost增强朴素贝叶斯
- **准确率：** 94%
- **分类：** 积极/消极/客观
- **数据量：** 10万条微博

### 📈 应用场景
- **品牌声誉监测**：分析社交媒体对品牌的评价倾向
- **舆情分析**：监测公众对热点事件的情感态度
- **用户反馈分析**：分析用户评论的情感分布
- **市场调研**：了解消费者对产品的情感倾向
""")

st.sidebar.markdown("---")
st.sidebar.caption("人工智能导论大作业 · 情感分析可视化系统")