// 添加位置因素影响函数
function evaluatePosition(position) {
    // 晚位置加成，早位置减弱
    if (position === 'late') return 0.15;
    if (position === 'middle') return 0.05;
    if (position === 'early') return -0.05;
    if (position === 'blinds') return -0.1;
    return 0;
}

// 根据庄家位置计算相对位置
function getRelativePositions(dealerPosition, playerCount) {
    const positions = [];
    for (let i = 0; i < playerCount; i++) {
        // 计算相对位置
        const relativePosition = (i - dealerPosition - 3 + playerCount) % playerCount;
        const positionPercentile = relativePosition / playerCount;
        
        // 根据位置百分比分配位置名称
        let positionName;
        if (i === (dealerPosition + 1) % playerCount || i === (dealerPosition + 2) % playerCount) {
            positionName = 'blinds'; // 小盲和大盲
        } else if (positionPercentile < 0.3) {
            positionName = 'early'; // 早位
        } else if (positionPercentile < 0.7) {
            positionName = 'middle'; // 中位
        } else {
            positionName = 'late'; // 晚位
        }
        positions[i] = positionName;
    }
    return positions;
}
