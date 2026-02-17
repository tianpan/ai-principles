/**
 * 德州扑克头像下载工具
 * 
 * 此脚本用于下载著名扑克玩家的头像到本地，确保游戏中能够显示真实的玩家头像
 */

// 著名扑克玩家头像URL
const POKER_AVATARS = [
    {
        name: 'daniel_negreanu',
        url: 'https://www.cardplayer.com/assets/players/000/033/714/profile/daniel-negreanu.jpg'
    },
    {
        name: 'phil_ivey',
        url: 'https://www.pokernews.com/img/incl/photos/phil-ivey-wsop-2019-nolan-dalla.jpg'
    },
    {
        name: 'phil_hellmuth',
        url: 'https://www.pokernews.com/img/incl/photos/phil-hellmuth-1-big.jpg'
    },
    {
        name: 'doyle_brunson',
        url: 'https://www.pokernews.com/img/incl/photos/doyle-brunson-2.jpg'
    },
    {
        name: 'johnny_chan',
        url: 'https://www.pokernews.com/img/incl/photos/johnny-chan-wsop.jpg'
    },
    {
        name: 'vanessa_selbst',
        url: 'https://www.pokernews.com/img/incl/photos/vanessaselbst-big.jpg'
    },
    {
        name: 'tom_dwan',
        url: 'https://www.pokernews.com/img/incl/photos/tom-dwan-poker-after-dark.jpg'
    },
    {
        name: 'fedor_holz',
        url: 'https://www.pokernews.com/img/incl/photos/fedor-holz-2018-wsop-chidwick.jpg'
    },
    {
        name: 'maria_ho',
        url: 'https://www.pokernews.com/img/incl/photos/maria-ho-2019-wsop-main-event.jpg'
    },
    {
        name: 'main_player',
        url: 'https://img.freepik.com/free-photo/young-bearded-man-with-striped-shirt_273609-5677.jpg'
    },
    {
        name: 'default',
        url: 'https://img.freepik.com/free-vector/businessman-character-avatar-isolated_24877-60111.jpg'
    }
];

// 存储位置
const SAVE_PATH = '../assets/images/players/';

/**
 * 下载单个头像
 * 
 * @param {Object} avatar 包含头像名称和URL的对象
 * @returns {Promise} 下载完成的Promise
 */
async function downloadAvatar(avatar) {
    console.log(`开始下载 ${avatar.name} 的头像...`);
    
    try {
        // 获取头像数据
        const response = await fetch(avatar.url);
        
        if (!response.ok) {
            throw new Error(`Download failed: ${response.status} ${response.statusText}`);
        }
        
        // 转换为Blob
        const imageBlob = await response.blob();
        
        // 创建下载链接
        const imageUrl = URL.createObjectURL(imageBlob);
        const link = document.createElement('a');
        link.href = imageUrl;
        link.download = `${avatar.name}.jpg`;
        
        // 添加到文档并触发点击
        document.body.appendChild(link);
        link.click();
        
        // 清理
        document.body.removeChild(link);
        URL.revokeObjectURL(imageUrl);
        
        console.log(`${avatar.name} 的头像下载完成`);
        return true;
    } catch (error) {
        console.error(`下载 ${avatar.name} 的头像时出错:`, error);
        return false;
    }
}

/**
 * 下载所有头像
 */
async function downloadAllAvatars() {
    console.log('开始下载所有头像...');
    
    // 创建下载状态显示
    const statusDiv = document.createElement('div');
    statusDiv.id = 'downloadStatus';
    statusDiv.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 20px;
        border-radius: 8px;
        font-family: Arial, sans-serif;
        z-index: 1000;
        max-width: 80%;
        max-height: 80%;
        overflow-y: auto;
    `;
    document.body.appendChild(statusDiv);
    
    // 创建进度信息
    const progressInfo = document.createElement('div');
    progressInfo.innerHTML = '<h2>下载头像中...</h2><p>请等待下载完成，并将下载的图片移动到 assets/images/players 目录</p>';
    statusDiv.appendChild(progressInfo);
    
    // 创建进度列表
    const progressList = document.createElement('ul');
    statusDiv.appendChild(progressList);
    
    let successCount = 0;
    
    // 下载每个头像
    for (let i = 0; i < POKER_AVATARS.length; i++) {
        const avatar = POKER_AVATARS[i];
        
        // 添加到进度列表
        const listItem = document.createElement('li');
        listItem.textContent = `${avatar.name}: 下载中...`;
        listItem.id = `avatar-${avatar.name}`;
        progressList.appendChild(listItem);
        
        // 下载头像
        const success = await downloadAvatar(avatar);
        
        // 更新进度
        if (success) {
            listItem.textContent = `${avatar.name}: 下载完成`;
            listItem.style.color = 'lightgreen';
            successCount++;
        } else {
            listItem.textContent = `${avatar.name}: 下载失败`;
            listItem.style.color = 'salmon';
        }
        
        // 延迟一下，避免同时触发太多下载
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    // 更新最终状态
    progressInfo.innerHTML = `
        <h2>下载完成</h2>
        <p>成功: ${successCount} / ${POKER_AVATARS.length}</p>
        <p>请将下载的图片移动到 assets/images/players 目录</p>
        <button id="closeStatusBtn" style="padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px;">关闭</button>
    `;
    
    // 添加关闭按钮功能
    document.getElementById('closeStatusBtn').addEventListener('click', function() {
        document.body.removeChild(statusDiv);
    });
}

// 创建下载按钮
function createDownloadButton() {
    const button = document.createElement('button');
    button.textContent = '下载玩家头像';
    button.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 10px 16px;
        background: #0d47a1;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
        z-index: 1000;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    `;
    
    button.addEventListener('click', downloadAllAvatars);
    document.body.appendChild(button);
}

// 当文档加载完成时创建下载按钮
document.addEventListener('DOMContentLoaded', createDownloadButton); 