## ADDED Requirements

### Requirement: Astro + Starlight 项目初始化
系统 SHALL 使用 Astro + Starlight 作为内容框架，支持 MDX 和 React 组件。

#### Scenario: 项目可运行
- **WHEN** 执行 `npm run dev`
- **THEN** 开发服务器在 localhost:4321 启动

### Requirement: Tailwind CSS 配置
系统 SHALL 使用 Tailwind CSS 进行样式管理，并通过 CSS Variables 定义 Design Tokens。

#### Scenario: 样式生效
- **WHEN** 使用 Tailwind 类名
- **THEN** 样式正确应用

### Requirement: React 交互组件支持
系统 SHALL 支持 React 组件作为 Islands 按需加载。

#### Scenario: 组件按需加载
- **WHEN** 页面包含交互组件
- **THEN** 组件 JS 仅在该页面加载
