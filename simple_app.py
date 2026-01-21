# simple_app.py - 极简版本，确保能部署成功
import streaml# 在 simple_app.py 最开头添加
import sys
import subprocess
import pkg_resources

# 检查并安装必要包
required = {
    'scikit-learn': '1.3.2',
    'pandas': '2.0.3',
    'numpy': '1.24.3',
    'jieba': '0.42.1'
}

for package, version in required.items():
    try:
        dist = pkg_resources.get_distribution(package)
        if dist.version != version:
            print(f"更新 {package} 从 {dist.version} 到 {version}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
    except pkg_resources.DistributionNotFound:
        print(f"安装 {package}=={version}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])

# 现在导入
import pickle
import pandas as pd
import numpy as npit as st
import pickle
import pandas as pd
import re

# 设置页面
st.set_page_config(page_title="情感分析系统", layout="wide")
st.title("📊 微博情感分析系统")
st.markdown("---")

# 尝试加载模型
@st.cache_resource
def load_model():
    try:
        with open('adaboost_nb_best_model.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model_info
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model_info = load_model()

if model_info is None:
    st.error("无法加载模型文件")
    st.stop()

model = model_info['model']
vectorizer = model_info['vectorizer']

# 标签映射
LABELS = {0: "客观", 1: "积极", 2: "消极"}

# ==================== 辅助函数 ====================
def simple_tokenize(text):
    """极简分词函数，替代 jieba"""
    # 移除标点符号，按空格分词
    text_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    # 按字符分割，但保留常用连接
    words = []
    current_word = ""
    for char in text_clean:
        if char.strip():  # 不是空格
            current_word += char
        else:
            if current_word:
                words.append(current_word)
                current_word = ""
    if current_word:
        words.append(current_word)
    
    # 过滤停用词
    stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人'}
    words = [w for w in words if w not in stopwords and len(w) > 0]
    
    return ' '.join(words)

# ==================== 主界面 ====================
st.header("🔤 文本情感分析")

# 输入区域
user_input = st.text_area(
    "请输入微博内容：",
    "今天天气真好，心情愉快！",
    height=100
)

if st.button("🚀 分析情感", type="primary"):
    # 处理文本
    processed = simple_tokenize(user_input)
    
    try:
        # 向量化并预测
        features = vectorizer.transform([processed])
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        # 显示结果
        sentiment = LABELS.get(pred, "未知")
        confidence = proba[pred]
        
        st.markdown("---")
        st.subheader("📊 分析结果")
        
        if sentiment == "积极":
            st.success(f"✅ **情感：{sentiment}** (置信度：{confidence:.2%})")
        elif sentiment == "消极":
            st.error(f"❌ **情感：{sentiment}** (置信度：{confidence:.2%})")
        else:
            st.info(f"📄 **情感：{sentiment}** (置信度：{confidence:.2%})")
        
        # 显示概率
        st.markdown("**各类别概率：**")
        for i, prob in enumerate(proba):
            label = LABELS.get(i, f"类别{i}")
            st.write(f"{label}: {prob:.2%}")
            
    except Exception as e:
        st.error(f"预测失败: {e}")

# ==================== 批量分析 ====================
st.markdown("---")
st.header("📋 批量分析")

batch_input = st.text_area(
    "输入多条文本（每行一条）",
    "今天很开心！\n这个产品很糟糕\n天气不错",
    height=150
)

if st.button("📥 批量分析", type="secondary"):
    texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
    results = []
    
    with st.spinner(f"正在分析 {len(texts)} 条文本..."):
        for text in texts:
            processed = simple_tokenize(text)
            features = vectorizer.transform([processed])
            pred = model.predict(features)[0]
            sentiment = LABELS.get(pred, "未知")
            results.append((text[:50] + "..." if len(text) > 50 else text, sentiment))
    
    # 显示结果
    st.markdown("### 批量分析结果")
    for i, (text, sentiment) in enumerate(results, 1):
        st.write(f"{i}. `{text}` → **{sentiment}**")
    
    # 统计
    sentiments = [r[1] for r in results]
    st.markdown(f"**统计：** 积极: {sentiments.count('积极')}, 消极: {sentiments.count('消极')}, 客观: {sentiments.count('客观')}")

# ==================== 底部信息 ====================
st.sidebar.header("ℹ️ 系统信息")
st.sidebar.info("""
### 模型信息
- **算法：** AdaBoost增强朴素贝叶斯
- **准确率：** 94%
- **训练数据：** 10万条微博
- **分类：** 积极/消极/客观

### 应用场景
- 品牌声誉监测
- 社交媒体舆情分析
- 用户反馈情感分析
- 市场调研情感倾向
""")

st.sidebar.markdown("---")
st.sidebar.caption("人工智能导论大作业 · 微博情感分析系统")
