# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

港华智慧能源工程管理 APP — 面向"港华→卓裕→金卓"工程服务链条的项目部精细化管理移动端系统。核心用户是**班组长和技术人员**（非一线工人）。

## 业务模块（四大领域）

1. **安全与考勤** — 班前早会考勤、安全交底、高风险作业视频监控
2. **进度与工效** — 每日施工人数/进度登记、人员备案核查、关键岗位持证验证
3. **材料与设备** — 二级库领料/退料、设备租赁台班、乙供材采购登记
4. **现场记录与竣工资料** — 施工记录（工程量+影像）、草图绘制、竣工图、签字确认

## 三方角色体系

| 角色 | 系统定位 |
|------|---------|
| 港华/卓裕 | 集团管理决策层 — 项目立项审批、预算控制、全局报表 |
| 金卓项目经理 | 项目级管理 — 每日确认、统筹安排、竣工管理 |
| 班组长 | 现场执行 — 每日登记（考勤、材料、进度、施工记录） |
| 资料员 | 资料管理 — 资料整理、竣工图绘制、签字跟踪 |

## 当前状态

项目处于**需求分析阶段**。业务需求文档和澄清问题清单在 `docs/` 目录。多个关键问题（考勤验证方式、角色权限细节、施工图格式、签字方式、范围边界）待业务部门确认。系统当前设计为**独立系统**，未来可能对接 ERP。

## 工作流工具

### OpenSpec（Spec-Driven Development）

规范驱动开发框架，通过 `openspec` CLI 管理变更：

```bash
openspec new change "<name>"          # 创建新变更
openspec status --change "<name>"     # 查看变更状态
openspec instructions <artifact> --change "<name>"  # 获取工件模板
openspec list                         # 列出所有变更
openspec update                       # 更新 OpenSpec 本身
```

快捷命令（`.claude/commands/opsx/`）：
- `/opsx:new` — 创建变更
- `/opsx:continue` — 推进工件
- `/opsx:apply` — 实施任务
- `/opsx:verify` — 验证变更
- `/opsx:archive` — 归档变更
- `/opsx:explore` — 探索代码库

变更工件存放在 `openspec/changes/`，规范存放在 `openspec/specs/`。

**重要**：`openspec/` 目录位于项目根目录（`/Users/admin/Documents/00-Projects/2026-04-03-projectmgt/openspec/`），不在 `app/openspec/`。创建或操作 OpenSpec 工件时，必须使用项目根目录的 `openspec/` 路径。

### gstack

所有 web browsing 必须使用 `/browse`，禁止使用 `mcp__claude-in-chrome__*` 工具。

主要 skills：`/review` `/ship` `/qa` `/browse` `/investigate` `/retro` `/office-hours` `/plan-ceo-review` `/plan-eng-review` `/plan-design-review` `/design-review` `/design-consultation` `/careful` `/freeze` `/guard` `/unfreeze` `/gstack-upgrade`

如果 gstack skills 异常：`cd .claude/skills/gstack && ./setup`

## 关键文件

- `docs/项目部工程管理业务需求.md` — 完整业务需求
- `docs/需求澄清问题清单.md` — 待确认问题及已有回答
- `openspec/` — OpenSpec 变更和规范目录
- `.claude/skills/` — gstack skills（symlink）+ OpenSpec skills
- `.claude/commands/opsx/` — OpenSpec 快捷命令
