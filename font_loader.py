# font_loader.py - 字体加载器

import os
import sys
import tempfile
import requests
import zipfile
import matplotlib
from matplotlib.font_manager import fontManager, FontProperties
import warnings

def load_chinese_fonts():
    """加载中文字体，支持多种来源"""
    fonts_loaded = []
    
    # 1. 先检查本地fonts文件夹
    local_fonts = []
    font_dirs = ['./fonts', 'fonts', '../fonts']
    
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for font_file in os.listdir(font_dir):
                if font_file.lower().endswith(('.ttf', '.otf')):
                    font_path = os.path.join(font_dir, font_file)
                    try:
                        fontManager.addfont(font_path)
                        local_fonts.append(font_path)
                        print(f"✅ 加载本地字体: {font_file}")
                    except Exception as e:
                        print(f"❌ 加载字体失败 {font_file}: {e}")
    
    if local_fonts:
        # 使用第一个找到的字体
        font_prop = FontProperties(fname=local_fonts[0])
        font_name = font_prop.get_name()
        matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial Unicode MS']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return True
    
    # 2. 检查是否有网盘链接文件
    cloud_font_file = 'cloud_fonts.txt'
    if os.path.exists(cloud_font_file):
        print("📦 发现云字体配置，正在处理...")
        try:
            with open(cloud_font_file, 'r') as f:
                cloud_config = f.read().strip()
            
            # 解析网盘链接（示例格式）
            if cloud_config.startswith('http'):
                # 这里添加下载网盘字体的代码
                # 由于不清楚你的网盘链接格式，这里使用示例代码
                print(f"🌐 发现网盘链接: {cloud_config}")
                # TODO: 根据你的网盘链接格式实现下载逻辑
                pass
        except Exception as e:
            print(f"❌ 读取云字体配置失败: {e}")
    
    # 3. 使用matplotlib的默认字体，添加中文字体支持
    try:
        # 尝试添加系统字体
        system_fonts = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Ubuntu文泉驿
            '/usr/share/fonts/truetype/arphic/uming.ttc',      # AR PL UMing
            '/System/Library/Fonts/PingFang.ttc',              # macOS苹方
            'C:/Windows/Fonts/msyh.ttc',                       # Windows微软雅黑
        ]
        
        for font_path in system_fonts:
            if os.path.exists(font_path):
                try:
                    fontManager.addfont(font_path)
                    font_prop = FontProperties(fname=font_path)
                    font_name = font_prop.get_name()
                    matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    print(f"✅ 加载系统字体: {font_path}")
                    return True
                except Exception as e:
                    print(f"❌ 加载系统字体失败 {font_path}: {e}")
    except Exception as e:
        print(f"❌ 系统字体检查失败: {e}")
    
    # 4. 最后回退到matplotlib默认字体
    print("⚠️ 未找到中文字体，使用默认字体（中文可能显示为方框）")
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return False

def download_font_from_cloud(url, save_path='fonts'):
    """从云盘下载字体"""
    try:
        os.makedirs(save_path, exist_ok=True)
        
        # 下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # 解压文件
        with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
            zip_ref.extractall(save_path)
        
        os.unlink(temp_file.name)
        print(f"✅ 字体下载完成: {save_path}")
        return True
    except Exception as e:
        print(f"❌ 字体下载失败: {e}")
        return False

# 初始化字体
font_loaded = load_chinese_fonts()