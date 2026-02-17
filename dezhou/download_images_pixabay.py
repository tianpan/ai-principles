import os
import requests
import random
import time

# 确保目标目录存在
target_dir = 'assets/images/players'
os.makedirs(target_dir, exist_ok=True)

# 由于我们不能使用真实API密钥，这里使用模拟数据
# 实际使用时需要在 https://pixabay.com/api/docs/ 申请API密钥
# API_KEY = "your_pixabay_api_key"

# 扑克玩家列表
players = [
    'daniel_negreanu', 'phil_ivey', 'phil_hellmuth', 'doyle_brunson',
    'johnny_chan', 'tom_dwan', 'patrik_antonius', 'fedor_holz',
    'justin_bonomo', 'jennifer_tilly', 'maria_ho', 'vanessa_selbst',
    'liv_boeree', 'jason_mercier', 'kristen_bicknell', 'phil_galfond',
    'celina_lin', 'xuan_liu', 'elton_tsang', 'jennifer_harman'
]

# 由于无法直接获取特定扑克玩家图片，我们将使用本地样例头像
def create_sample_avatars():
    """创建样例头像作为替代"""
    print("由于API限制，将创建样例头像文件...")
    
    # 准备一些基本颜色作为头像背景
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
        (255, 165, 0),  # 橙色
        (128, 0, 128),  # 紫色
        (165, 42, 42),  # 棕色
        (220, 20, 60)   # 深红色
    ]
    
    # 为每个玩家创建一个简单的彩色头像
    for player in players:
        color = random.choice(colors)
        # 创建一个简单的彩色SVG头像
        svg_content = f"""<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="200" fill="rgb{color}" />
    <text x="50%" y="50%" font-family="Arial" font-size="24" fill="white" text-anchor="middle" dominant-baseline="middle">
        {player.replace('_', ' ').title()}
    </text>
</svg>"""
        
        # 将SVG保存为文件
        with open(os.path.join(target_dir, f"{player}.svg"), 'w') as f:
            f.write(svg_content)
        
        # 将原始的jpg文件备份为.bak后缀
        jpg_file = os.path.join(target_dir, f"{player}.jpg")
        if os.path.exists(jpg_file):
            backup_file = os.path.join(target_dir, f"{player}.jpg.bak")
            try:
                os.rename(jpg_file, backup_file)
                print(f"已备份原始文件 {jpg_file} 为 {backup_file}")
            except Exception as e:
                print(f"备份文件失败 {jpg_file}: {e}")
        
        print(f"已创建样例头像 {player}.svg")
        
        # 稍作延迟，避免过快创建文件
        time.sleep(0.1)

# 执行创建样例头像的函数
create_sample_avatars()

print("\n所有样例头像已创建完成。")
print("注意：这些是临时的样例头像，如需真实的扑克玩家头像，请使用合适的图片源或商业图库服务。") 