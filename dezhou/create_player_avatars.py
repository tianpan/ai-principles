import os
import random
import string

# 确保目标目录存在
target_dir = 'assets/images/players'
os.makedirs(target_dir, exist_ok=True)

# 扑克玩家列表
players = [
    'daniel_negreanu', 'phil_ivey', 'phil_hellmuth', 'doyle_brunson',
    'johnny_chan', 'tom_dwan', 'patrik_antonius', 'fedor_holz',
    'justin_bonomo', 'jennifer_tilly', 'maria_ho', 'vanessa_selbst',
    'liv_boeree', 'jason_mercier', 'kristen_bicknell', 'phil_galfond',
    'celina_lin', 'xuan_liu', 'elton_tsang', 'jennifer_harman'
]

# 生成更专业的头像
def create_professional_avatars():
    """创建更专业的扑克玩家头像"""
    print("创建高质量扑克玩家头像...")
    
    # 头像样式和颜色
    styles = ['classic', 'modern', 'vintage', 'elegant', 'bold']
    colors = [
        ('#1E88E5', '#FFFFFF'),  # 蓝色和白色
        ('#D81B60', '#FFFFFF'),  # 红色和白色
        ('#FFC107', '#212121'),  # 金色和黑色
        ('#43A047', '#FFFFFF'),  # 绿色和白色
        ('#5E35B1', '#FFFFFF'),  # 紫色和白色
        ('#FF5722', '#FFFFFF'),  # 橙色和白色
        ('#3949AB', '#FFFFFF'),  # 靛蓝和白色
        ('#00ACC1', '#FFFFFF'),  # 青色和白色
        ('#212121', '#FFC107'),  # 黑色和金色
        ('#607D8B', '#FFFFFF'),  # 蓝灰色和白色
    ]
    
    # 为扑克玩家创建专业头像
    for player in players:
        # 选择随机样式和颜色
        style = random.choice(styles)
        background_color, text_color = random.choice(colors)
        
        # 获取玩家的首字母
        name_parts = player.split('_')
        initials = ''.join([part[0].upper() for part in name_parts])
        
        # 添加扑克元素 - 使用小写字母表示花色
        suits = ['s', 'h', 'd', 'c']  # 黑桃、红心、方块、梅花
        suit = random.choice(suits)
        suit_color = '#FF0000' if suit in ['h', 'd'] else '#000000'
        
        # 为每个玩家创建专业的SVG头像
        if style == 'classic':
            svg_content = f"""<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="200" fill="{background_color}" />
    <text x="50%" y="50%" font-family="Arial, sans-serif" font-size="80" font-weight="bold" 
          fill="{text_color}" text-anchor="middle" dominant-baseline="middle">{initials}</text>
    <text x="50%" y="85%" font-family="Arial, sans-serif" font-size="16" 
          fill="{text_color}" text-anchor="middle" dominant-baseline="middle">
        {player.replace('_', ' ').title()}
    </text>
    <text x="85%" y="20%" font-family="Arial" font-size="24" fill="{suit_color}">{suit.upper()}</text>
    <text x="15%" y="80%" font-family="Arial" font-size="24" fill="{suit_color}">{suit.upper()}</text>
</svg>"""
        elif style == 'modern':
            svg_content = f"""<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="{background_color}" />
            <stop offset="100%" stop-color="{darken_color(background_color)}" />
        </linearGradient>
    </defs>
    <rect width="200" height="200" fill="url(#grad1)" />
    <circle cx="100" cy="85" r="60" fill="{text_color}" fill-opacity="0.2" />
    <text x="50%" y="50%" font-family="Arial, sans-serif" font-size="80" font-weight="bold" 
          fill="{text_color}" text-anchor="middle" dominant-baseline="middle">{initials}</text>
    <text x="50%" y="85%" font-family="Arial, sans-serif" font-size="16" 
          fill="{text_color}" text-anchor="middle" dominant-baseline="middle">
        {player.replace('_', ' ').title()}
    </text>
    <text x="85%" y="20%" font-family="Arial" font-size="24" fill="{suit_color}">{suit.upper()}</text>
    <text x="15%" y="80%" font-family="Arial" font-size="24" fill="{suit_color}">{suit.upper()}</text>
</svg>"""
        elif style == 'vintage':
            svg_content = f"""<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="200" fill="{background_color}" />
    <rect x="10" y="10" width="180" height="180" fill="none" stroke="{text_color}" stroke-width="2" />
    <text x="50%" y="50%" font-family="Georgia, serif" font-size="70" font-weight="bold" 
          fill="{text_color}" text-anchor="middle" dominant-baseline="middle">{initials}</text>
    <text x="50%" y="85%" font-family="Georgia, serif" font-size="16" 
          fill="{text_color}" text-anchor="middle" dominant-baseline="middle">
        {player.replace('_', ' ').title()}
    </text>
    <text x="85%" y="20%" font-family="Georgia" font-size="24" fill="{suit_color}">{suit.upper()}</text>
    <text x="15%" y="80%" font-family="Georgia" font-size="24" fill="{suit_color}">{suit.upper()}</text>
</svg>"""
        elif style == 'elegant':
            svg_content = f"""<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="200" fill="{background_color}" />
    <circle cx="100" cy="100" r="80" fill="none" stroke="{text_color}" stroke-width="1" />
    <circle cx="100" cy="100" r="70" fill="none" stroke="{text_color}" stroke-width="1" />
    <text x="50%" y="50%" font-family="Times New Roman, serif" font-size="70" font-weight="bold" 
          fill="{text_color}" text-anchor="middle" dominant-baseline="middle">{initials}</text>
    <text x="50%" y="85%" font-family="Times New Roman, serif" font-size="16" 
          fill="{text_color}" text-anchor="middle" dominant-baseline="middle">
        {player.replace('_', ' ').title()}
    </text>
    <text x="85%" y="20%" font-family="Times New Roman" font-size="24" fill="{suit_color}">{suit.upper()}</text>
    <text x="15%" y="80%" font-family="Times New Roman" font-size="24" fill="{suit_color}">{suit.upper()}</text>
</svg>"""
        else:  # bold
            svg_content = f"""<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="200" fill="{background_color}" />
    <rect x="30" y="30" width="140" height="140" rx="20" ry="20" fill="none" stroke="{text_color}" stroke-width="4" />
    <text x="50%" y="50%" font-family="Impact, sans-serif" font-size="80" 
          fill="{text_color}" text-anchor="middle" dominant-baseline="middle">{initials}</text>
    <text x="50%" y="85%" font-family="Arial, sans-serif" font-size="16" font-weight="bold" 
          fill="{text_color}" text-anchor="middle" dominant-baseline="middle">
        {player.replace('_', ' ').title()}
    </text>
    <text x="85%" y="20%" font-family="Impact" font-size="24" fill="{suit_color}">{suit.upper()}</text>
    <text x="15%" y="80%" font-family="Impact" font-size="24" fill="{suit_color}">{suit.upper()}</text>
</svg>"""
        
        # 将SVG保存为文件，使用utf-8编码
        with open(os.path.join(target_dir, f"{player}.svg"), 'w', encoding='utf-8') as f:
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
        
        print(f"已创建专业头像 {player}.svg")

# 辅助函数 - 使颜色变暗
def darken_color(hex_color):
    """使十六进制颜色变暗"""
    # 移除#前缀
    hex_color = hex_color.lstrip('#')
    # 转换为RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # 使颜色变暗
    factor = 0.7
    r = max(int(r * factor), 0)
    g = max(int(g * factor), 0)
    b = max(int(b * factor), 0)
    # 转换回十六进制
    return f"#{r:02x}{g:02x}{b:02x}"

# 执行创建专业头像的函数
create_professional_avatars()

print("\n所有专业头像已创建完成！")
print("注意：这些是高质量的SVG头像，适合在网页和应用中使用。")
print("如果您需要将它们转换为PNG格式，可以使用在线转换工具或安装librsvg等工具。") 