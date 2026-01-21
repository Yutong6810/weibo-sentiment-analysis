# visualization_app.py - 统一修改版

# 添加BERT模型需要的库
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
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

# ==================== 模型加载函数 ====================

@st.cache_resource
def load_naive_bayes_model():
    """加载朴素贝叶斯模型"""
    try:
        with open('naive_bayes_best_model.pkl', 'rb') as f:
            model_info = pickle.load(f)
        st.success("✅ 朴素贝叶斯模型加载完成！")
        return {
            'model': model_info['model'],
            'vectorizer': model_info['vectorizer'],
            'model_type': 'naive_bayes'
        }
    except Exception as e:
        st.error(f"❌ 朴素贝叶斯模型加载失败: {e}")
        return None

@st.cache_resource
def load_adaboost_nb_model():
    """加载AdaBoost增强朴素贝叶斯模型"""
    try:
        with open('adaboost_nb_best_model.pkl', 'rb') as f:
            model_info = pickle.load(f)
        st.success("✅ AdaBoost增强朴素贝叶斯模型加载完成！")
        return {
            'model': model_info['model'],
            'vectorizer': model_info['vectorizer'],
            'model_type': 'adaboost_nb'
        }
    except Exception as e:
        st.error(f"❌ AdaBoost模型加载失败: {e}")
        return None

@st.cache_resource
def load_bert_model():
    """加载BERT深度学习模型"""
    st.info("正在加载BERT模型，首次加载可能需要较长时间...")
    try:
        # 定义模型结构
        class BertSentimentClassifier(nn.Module):
            def __init__(self, bert_model_name='bert-base-chinese', num_classes=3):
                super(BertSentimentClassifier, self).__init__()
                self.bert = BertModel.from_pretrained(bert_model_name)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                return logits
        
        # 加载检查点
        checkpoint = torch.load('sentiment_model_fixed.pth', map_location=torch.device('cpu'))
        
        # 初始化模型
        model = BertSentimentClassifier()
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            # 移除"module."前缀（如果存在）
            new_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                else:
                    new_key = key
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # 加载分词器
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        st.success("✅ BERT模型加载完成！")
        return {
            'model': model,
            'tokenizer': tokenizer,
            'model_type': 'bert'
        }
    
    except Exception as e:
        st.error(f"❌ BERT模型加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# 统一标签映射
LABELS = {0: "客观", 1: "积极", 2: "消极"}
COLORS = {'积极': '#4CAF50', '消极': '#F44336', '客观': '#2196F3'}

# ==================== 辅助函数 ====================

def preprocess_text(text):
    """预处理文本"""
    # 去除特殊字符
    text_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    words = jieba.lcut(text_clean)
    processed = ' '.join(words)
    return processed

def analyze_text_naive_bayes(text, model_info):
    """使用朴素贝叶斯模型分析文本"""
    if model_info is None:
        return {'error': '朴素贝叶斯模型未加载成功'}
    
    model = model_info['model']
    vectorizer = model_info['vectorizer']
    
    # 预处理
    processed = preprocess_text(text)
    
    # 提取特征
    features = vectorizer.transform([processed])
    
    # 预测
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    # 使用统一标签映射
    sentiment = LABELS.get(pred, "未知")
    confidence = proba[pred]
    
    # 获取所有类别的概率
    prob_dict = {}
    for i, prob in enumerate(proba):
        label = LABELS.get(i, f"类别{i}")
        prob_dict[label] = prob
    
    return {
        'text': text[:100] + "..." if len(text) > 100 else text,
        'full_text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': prob_dict,
        'model_type': '朴素贝叶斯'
    }

def analyze_text_adaboost_nb(text, model_info):
    """使用AdaBoost增强朴素贝叶斯模型分析文本"""
    if model_info is None:
        return {'error': 'AdaBoost模型未加载成功'}
    
    model = model_info['model']
    vectorizer = model_info['vectorizer']
    
    # 预处理
    processed = preprocess_text(text)
    
    # 提取特征
    features = vectorizer.transform([processed])
    
    # 预测
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    # 使用统一标签映射
    sentiment = LABELS.get(pred, "未知")
    confidence = proba[pred]
    
    # 获取所有类别的概率
    prob_dict = {}
    for i, prob in enumerate(proba):
        label = LABELS.get(i, f"类别{i}")
        prob_dict[label] = prob
    
    return {
        'text': text[:100] + "..." if len(text) > 100 else text,
        'full_text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': prob_dict,
        'model_type': 'AdaBoost增强朴素贝叶斯'
    }

def analyze_text_bert(text, model_info):
    """使用BERT模型分析文本"""
    if model_info is None:
        return {'error': 'BERT模型未加载成功'}
    
    model = model_info['model']
    tokenizer = model_info['tokenizer']
    
    # 使用BERT分词器处理文本
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    # 预测
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # BERT模型的原始标签顺序
    bert_labels = ["消极", "客观", "积极"] 
    
    sentiment = bert_labels[predicted_class]
    
    # 获取所有类别的概率
    prob_list = probabilities[0].tolist()
    prob_dict = {bert_labels[i]: prob_list[i] for i in range(len(bert_labels))}
    
    return {
        'text': text[:100] + "..." if len(text) > 100 else text,
        'full_text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': prob_dict,
        'model_type': 'BERT'
    }

def analyze_text(text, model_choice):
    """根据模型选择分析文本"""
    if model_choice == "朴素贝叶斯 (基础)":
        if 'nb_model' not in st.session_state:
            st.session_state.nb_model = load_naive_bayes_model()
        return analyze_text_naive_bayes(text, st.session_state.nb_model)
    
    elif model_choice == "集成学习: AdaBoost增强朴素贝叶斯":
        if 'adaboost_model' not in st.session_state:
            st.session_state.adaboost_model = load_adaboost_nb_model()
        return analyze_text_adaboost_nb(text, st.session_state.adaboost_model)
    
    elif model_choice == "深度学习: BERT情感分析模型":
        if 'bert_model' not in st.session_state:
            st.session_state.bert_model = load_bert_model()
        return analyze_text_bert(text, st.session_state.bert_model)
    
    else:
        return {'error': '未知模型选择'}

# ==================== 侧边栏 ====================
st.sidebar.header("⚙️ 模型选择")
model_choice = st.sidebar.selectbox(
    "选择分析模型",
    ["朴素贝叶斯 (基础)",
     "集成学习: AdaBoost增强朴素贝叶斯",
     "深度学习: BERT情感分析模型"]
)

st.sidebar.header("📋 批量分析")
batch_input = st.sidebar.text_area(
    "输入多条文本（每行一条）",
    "今天很开心！\n这个产品很糟糕\n天气不错\n服务态度很好\n电影不好看",
    height=150
)

if st.sidebar.button("📥 批量分析", type="secondary"):
    texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
    if texts:
        with st.spinner(f"正在批量分析 {len(texts)} 条文本..."):
            for text in texts:
                result = analyze_text(text, model_choice)
                if 'error' not in result:
                    st.session_state.history.append(text)
                    st.session_state.results.append(result)
            st.sidebar.success(f"✅ 批量分析完成！分析了 {len(texts)} 条文本")
    else:
        st.sidebar.warning("请输入文本内容")

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
            if user_input.strip():
                result = analyze_text(user_input, model_choice)
                
                if 'error' not in result:
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    
                    # 显示结果
                    if sentiment == "积极":
                        st.success(f"✅ **情感：{sentiment}** (置信度：{confidence:.2%})")
                    elif sentiment == "消极":
                        st.error(f"❌ **情感：{sentiment}** (置信度：{confidence:.2%})")
                    else:
                        st.info(f"📄 **情感：{sentiment}** (置信度：{confidence:.2%})")
                    
                    # 显示详细概率
                    with st.expander("查看详细概率"):
                        for label, prob in result['probabilities'].items():
                            st.write(f"{label}: {prob:.2%}")
                    
                    # 保存结果
                    st.session_state.history.append(user_input)
                    st.session_state.results.append(result)
                else:
                    st.error(f"分析失败: {result['error']}")
            else:
                st.warning("请输入文本内容")
    
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
                # 确保颜色顺序一致
                colors = [COLORS.get(l, '#999') for l in labels]
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
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
        
        # 确保标签和颜色顺序
        labels = sentiment_counts.index.tolist()
        sizes = sentiment_counts.values.tolist()
        colors = [COLORS.get(l, '#999') for l in labels]
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax1.set_title("情感分布")
        st.pyplot(fig1)
        
        # 图表2：情感分布柱状图
        st.subheader("情感分布柱状图")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        bars = ax2.bar(labels, sizes, color=colors)
        ax2.set_xlabel("情感")
        ax2.set_ylabel("评论数量")
        ax2.set_title("情感分布统计")
        
        # 在柱子上显示数字
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        st.pyplot(fig2)
        
        # 图表3：置信度分布
        if 'confidence' in df.columns:
            st.subheader("置信度分布")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            
            # 为每种情感创建置信度分布
            for sentiment in df['sentiment'].unique():
                data = df[df['sentiment'] == sentiment]['confidence']
                if len(data) > 0:
                    ax3.hist(data, alpha=0.5, label=sentiment,
                            color=COLORS.get(sentiment), bins=10)
            
            ax3.set_xlabel("置信度")
            ax3.set_ylabel("评论数量")
            ax3.set_title("各情感置信度分布")
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
            ax_brand.set_title("品牌声誉分析")
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
            st.metric("总体积极率", f"{(positive/len(df)*100):.1f}%" if len(df) > 0 else "0%")
            st.metric("总体消极率", f"{(negative/len(df)*100):.1f}%" if len(df) > 0 else "0%")
            if 'confidence' in df.columns:
                st.metric("平均置信度", f"{df['confidence'].mean():.2%}" if len(df) > 0 else "0%")

with tab3:
    st.header("📋 分析历史记录")
    
    if not st.session_state.results:
        st.info("暂无分析记录")
    else:
        # 显示详细记录
        st.subheader("详细记录")
        
        # 可排序的表格
        df_display = pd.DataFrame(st.session_state.results)
        
        # 确保包含必要的列
        if 'text' in df_display.columns and 'sentiment' in df_display.columns and 'confidence' in df_display.columns:
            df_display = df_display[['text', 'sentiment', 'confidence', 'model_type']]
            df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.2%}")
            df_display.columns = ['文本', '情感', '置信度', '模型类型']
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("数据格式不正确")
        
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
- **模型：** 朴素贝叶斯 / AdaBoost增强 / BERT
- **分类：** 积极 / 消极 / 客观
- **统一标签：** 所有模型使用中文标签

### 📈 应用场景
- **品牌声誉监测**：分析社交媒体对品牌的评价倾向
- **舆情分析**：监测公众对热点事件的情感态度
- **用户反馈分析**：分析用户评论的情感分布
- **市场调研**：了解消费者对产品的情感倾向
""")

st.sidebar.markdown("---")
st.sidebar.caption("人工智能导论大作业 · 情感分析可视化系统")