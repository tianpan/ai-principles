import os
import requests
from urllib.parse import quote

# 确保目标目录存在
target_dir = 'assets/images/players'
os.makedirs(target_dir, exist_ok=True)

# 扑克玩家列表及其对应的图片搜索关键词
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

# 使用unsplash API来获取头像图片
def download_image(name, keyword):
    print(f"Downloading image for {name}...")
    try:
        # 使用Unsplash Source API获取随机人物照片
        # 由于我们不能直接搜索特定人物，这里使用一个通用的方式获取与关键词相关的图片
        url = f"https://source.unsplash.com/featured/?{quote(keyword)}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(os.path.join(target_dir, f"{name}.jpg"), 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Successfully downloaded image for {name}")
        else:
            print(f"Failed to download image for {name}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading image for {name}: {e}")

# 下载所有玩家头像
for name, keyword in players.items():
    download_image(name, keyword)

print("All images downloaded successfully!") 