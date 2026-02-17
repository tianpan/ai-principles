import os
import time
import urllib.request
import ssl
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 忽略SSL证书错误
ssl._create_default_https_context = ssl._create_unverified_context

# 确保目标目录存在
target_dir = 'assets/images/players'
os.makedirs(target_dir, exist_ok=True)

# 扑克玩家列表及其搜索关键词
players = {
    'daniel_negreanu': 'Daniel Negreanu poker player',
    'phil_ivey': 'Phil Ivey poker player',
    'phil_hellmuth': 'Phil Hellmuth poker player',
    'doyle_brunson': 'Doyle Brunson poker player',
    'johnny_chan': 'Johnny Chan poker player',
    'tom_dwan': 'Tom Dwan poker player', 
    'patrik_antonius': 'Patrik Antonius poker player',
    'fedor_holz': 'Fedor Holz poker player',
    'justin_bonomo': 'Justin Bonomo poker player',
    'jennifer_tilly': 'Jennifer Tilly poker player',
    'maria_ho': 'Maria Ho poker player',
    'vanessa_selbst': 'Vanessa Selbst poker player',
    'liv_boeree': 'Liv Boeree poker player',
    'jason_mercier': 'Jason Mercier poker player',
    'kristen_bicknell': 'Kristen Bicknell poker player',
    'phil_galfond': 'Phil Galfond poker player',
    'celina_lin': 'Celina Lin poker player',
    'xuan_liu': 'Xuan Liu poker player',
    'elton_tsang': 'Elton Tsang poker player',
    'jennifer_harman': 'Jennifer Harman poker player'
}

def download_images():
    """使用Selenium从Google搜索下载扑克玩家真实头像"""
    
    print("正在设置Chrome WebDriver...")
    
    # 设置Chrome选项
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 无头模式
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # 初始化WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    for player_name, search_query in players.items():
        print(f"\n正在搜索 {player_name} 的图片...")
        
        # 构建Google图片搜索URL
        search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
        
        try:
            # 打开Google图片搜索
            driver.get(search_url)
            time.sleep(2)  # 等待页面加载
            
            # 定位第一张图片
            images = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
            if not images:
                print(f"未找到 {player_name} 的图片")
                continue
                
            # 点击第一张图片获取高清版本
            images[0].click()
            time.sleep(1)
            
            # 获取大图
            big_img = driver.find_elements(By.CSS_SELECTOR, "img.r48jcc")
            if big_img:
                img_url = big_img[0].get_attribute('src')
                
                if img_url.startswith('data:'):
                    print(f"跳过 {player_name} 的Base64编码图片")
                    continue
                
                # 下载图片
                img_path = os.path.join(target_dir, f"{player_name}.jpg")
                
                # 备份原始文件
                if os.path.exists(img_path):
                    os.rename(img_path, f"{img_path}.bak")
                
                print(f"下载 {player_name} 的图片: {img_url}")
                urllib.request.urlretrieve(img_url, img_path)
                print(f"成功保存 {player_name} 的图片到 {img_path}")
            else:
                print(f"无法获取 {player_name} 的高清图片")
                
        except Exception as e:
            print(f"下载 {player_name} 的图片时出错: {e}")
    
    # 关闭WebDriver
    driver.quit()
    print("\n所有图片下载完成！")

if __name__ == "__main__":
    print("开始下载扑克玩家的真实头像...")
    download_images()
    print("\n提示: 如果部分图片下载失败，您可以考虑:")
    print("1. 手动从Google或Bing图片搜索下载")
    print("2. 从扑克网站如PokerNews, WSOP等获取官方照片")
    print("3. 购买商业图库如Shutterstock, Getty Images的照片") 