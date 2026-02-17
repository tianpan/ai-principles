import os
import sys
import urllib.request
import ssl

# 忽略SSL证书错误
ssl._create_default_https_context = ssl._create_unverified_context

def download_image(url, filename):
    """从URL下载图片并保存为指定文件名"""
    target_dir = 'assets/images/players'
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, filename)
    
    # 备份原始文件
    if os.path.exists(target_path):
        backup_path = f"{target_path}.bak"
        try:
            os.rename(target_path, backup_path)
            print(f"已备份原始文件 {target_path} 为 {backup_path}")
        except Exception as e:
            print(f"备份文件失败 {target_path}: {e}")
    
    print(f"正在从 {url} 下载图片...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"成功保存图片到 {target_path}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("用法: python download_image_from_url.py <图片URL> <保存的文件名>")
        print("例如: python download_image_from_url.py https://example.com/image.jpg daniel_negreanu.jpg")
        return
    
    url = sys.argv[1]
    filename = sys.argv[2]
    
    success = download_image(url, filename)
    if success:
        print(f"图片 {filename} 下载成功!")
    else:
        print("下载失败，请检查URL或网络连接。")

if __name__ == "__main__":
    main() 