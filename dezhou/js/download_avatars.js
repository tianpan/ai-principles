const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

// 创建目录函数
function ensureDirectoryExists(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`创建目录: ${dirPath}`);
  }
}

// 下载图片函数
function downloadImage(url, filePath) {
  return new Promise((resolve, reject) => {
    // 如果文件已存在，跳过下载
    if (fs.existsSync(filePath)) {
      console.log(`文件已存在，跳过下载: ${filePath}`);
      return resolve();
    }

    // 选择http或https
    const client = url.startsWith('https') ? https : http;
    
    const request = client.get(url, (response) => {
      // 处理重定向
      if (response.statusCode === 301 || response.statusCode === 302) {
        console.log(`重定向: ${url} -> ${response.headers.location}`);
        return downloadImage(response.headers.location, filePath)
          .then(resolve)
          .catch(reject);
      }
      
      // 检查状态码
      if (response.statusCode !== 200) {
        return reject(new Error(`下载失败，状态码: ${response.statusCode}, URL: ${url}`));
      }
      
      // 保存文件
      const fileStream = fs.createWriteStream(filePath);
      response.pipe(fileStream);
      
      fileStream.on('finish', () => {
        fileStream.close();
        console.log(`下载成功: ${filePath}`);
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(filePath, () => {}); // 删除未完成的文件
      console.error(`下载失败: ${url}, 错误: ${err.message}`);
      reject(err);
    });

    request.setTimeout(10000, () => {
      request.abort();
      fs.unlink(filePath, () => {});
      reject(new Error(`下载超时: ${url}`));
    });
  });
}

// 主函数
async function main() {
  // 确保玩家图片目录存在
  const playerImagesDir = path.join('assets', 'images', 'players');
  ensureDirectoryExists(playerImagesDir);

  // 定义要下载的图片
  const images = [
    // 主玩家
    { 
      url: 'https://m.media-amazon.com/images/M/MV5BMjA2NTI1ODk3N15BMl5BanBnXkFtZTcwODUyMTM1MQ@@._V1_.jpg', 
      file: 'main_player.jpg',
      name: '吴彦祖'
    },
    
    // 男性玩家
    { 
      url: 'https://www.pokertube.com/wp-content/uploads/2020/04/Negreanu.jpg', 
      file: 'daniel_negreanu.jpg',
      name: 'Daniel Negreanu'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/phil-ivey-2019-wsop.jpg', 
      file: 'phil_ivey.jpg',
      name: 'Phil Ivey'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/doyle-brunson-wsop-1.jpg', 
      file: 'doyle_brunson.jpg',
      name: 'Doyle Brunson'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/phil-hellmuth-15-bracelets.jpg', 
      file: 'phil_hellmuth.jpg',
      name: 'Phil Hellmuth'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/johnny-chan-wsop-2010.jpg', 
      file: 'johnny_chan.jpg',
      name: 'Johnny Chan'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/tom-dwan-triton.jpg', 
      file: 'tom_dwan.jpg',
      name: 'Tom Dwan'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/fedor-holz-wsop.jpg', 
      file: 'fedor_holz.jpg',
      name: 'Fedor Holz'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/justin-bonomo-2018-one-drop.jpg', 
      file: 'justin_bonomo.jpg',
      name: 'Justin Bonomo'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/patrik-antonius-poker-after-dark.jpg', 
      file: 'patrik_antonius.jpg',
      name: 'Patrik Antonius'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/phil-galfond-2017-wsop.jpg', 
      file: 'phil_galfond.jpg',
      name: 'Phil Galfond'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/jason-mercier-trophy.jpg', 
      file: 'jason_mercier.jpg',
      name: 'Jason Mercier'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/elton-tsang-one-drop.jpg', 
      file: 'elton_tsang.jpg',
      name: 'Elton Tsang'
    },
    
    // 女性玩家
    { 
      url: 'https://www.pokernews.com/img/incl/photos/vanessa-selbst-2013-wsop.jpg', 
      file: 'vanessa_selbst.jpg',
      name: 'Vanessa Selbst'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/liv-boeree-2015-wsop.jpg', 
      file: 'liv_boeree.jpg',
      name: 'Liv Boeree'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/jennifer-harman-wsop.jpg', 
      file: 'jennifer_harman.jpg',
      name: 'Jennifer Harman'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/maria-ho-2019-wsop.jpg', 
      file: 'maria_ho.jpg',
      name: 'Maria Ho'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/kristen-bicknell-partypoker.jpg', 
      file: 'kristen_bicknell.jpg',
      name: 'Kristen Bicknell'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/celina-lin-macau.jpg', 
      file: 'celina_lin.jpg',
      name: 'Celina Lin'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/xuan-liu-poker.jpg', 
      file: 'xuan_liu.jpg',
      name: 'Xuan Liu'
    },
    { 
      url: 'https://www.pokernews.com/img/incl/photos/jennifer-tilly-wpt.jpg', 
      file: 'jennifer_tilly.jpg',
      name: 'Jennifer Tilly'
    }
  ];

  // 为所有玩家数据库中的玩家创建条目
  const pokerPlayersDatabase = [
    { name: 'Daniel Negreanu' },
    { name: 'Phil Ivey' },
    { name: 'Doyle Brunson' },
    { name: 'Phil Hellmuth' },
    { name: 'Patrik Antonius' },
    { name: 'Vanessa Selbst' },
    { name: 'Liv Boeree' },
    { name: 'Jennifer Harman' },
    { name: 'Maria Ho' },
    { name: 'Johnny Chan' },
    { name: 'Fedor Holz' },
    { name: 'Justin Bonomo' },
    { name: 'Kristen Bicknell' },
    { name: 'Phil Galfond' },
    { name: 'Jason Mercier' },
    { name: 'Elton Tsang' },
    { name: 'Celina Lin' },
    { name: 'Xuan Liu' },
    { name: 'Jennifer Tilly' }
  ];

  // 下载图片
  const downloadPromises = images.map(image => {
    const filePath = path.join(playerImagesDir, image.file);
    console.log(`开始下载 ${image.name} 的头像: ${image.url}`);
    return downloadImage(image.url, filePath).catch(err => {
      console.error(`下载 ${image.name} 头像失败: ${err.message}`);
    });
  });

  try {
    await Promise.all(downloadPromises);
    console.log('所有头像下载完成！');
  } catch (error) {
    console.error('下载过程中发生错误:', error);
  }
}

// 执行主函数
main().catch(err => {
  console.error('程序执行失败:', err);
}); 