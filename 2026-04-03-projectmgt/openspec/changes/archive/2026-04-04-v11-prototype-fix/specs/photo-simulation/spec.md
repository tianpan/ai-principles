## ADDED Requirements

### Requirement: Simulated photo capture
系统 SHALL 提供模拟拍照组件 `PhotoCapture`，用户点击后模拟拍照并显示预设图片。

#### Scenario: User taps photo area for first time
- **WHEN** 用户点击照片占位区域
- **THEN** 系统弹出底部选择面板，显示"拍照"和"从相册选择"两个选项

#### Scenario: User selects "拍照"
- **WHEN** 用户在底部面板中选择"拍照"
- **THEN** 面板关闭，显示 300ms 加载动画，然后展示一张预设图片（picsum.photos），图片上叠加当前日期时间水印

#### Scenario: User selects "从相册选择"
- **WHEN** 用户在底部面板中选择"从相册选择"
- **THEN** 面板关闭，显示 300ms 加载动画，然后展示一张预设图片

#### Scenario: Photo already taken
- **WHEN** 照片已拍摄且显示
- **THEN** 用户可点击照片区域重新拍照，或点击"+"追加更多照片（最多3张）

### Requirement: Photo component integration
PhotoCapture 组件 SHALL 接受 `onPhotoTaken: (urls: string[]) => void` 回调，返回已拍摄的图片 URL 列表。

#### Scenario: Multiple photos
- **WHEN** 用户连续拍摄3张照片
- **THEN** 组件显示3张缩略图，`onPhotoTaken` 回调返回包含3个 URL 的数组

#### Scenario: Component used in safety page
- **WHEN** 安全签到页面渲染
- **THEN** 照片区域使用 PhotoCapture 组件，拍照后 `canNext` 验证包含照片检查
