## 1. Project Setup

> **Skill**: 完成后调用 `browse` skill 对首页做视觉验证

- [x] 1.1 Initialize Next.js 14 project with App Router and Tailwind CSS
- [x] 1.2 Configure project structure: `/app/(app)/` for mobile, `/app/(pc)/` for PC, `/src/mock/` for data
- [x] 1.3 Define Tailwind design tokens (colors: blue #1677ff, orange #fa8c16, green #52c41a, red #ff4d4f; border-radius: 8px/4px)

## 2. Mock Data Layer

> **Skill**: 使用 `frontend-patterns` skill 的 Context+Reducer 模式实现 MockDataProvider

- [x] 2.1 Create `src/mock/users.json` — 1 PM (王经理), 3 team leaders (张三/李四/王五), 1 admin
- [x] 2.2 Create `src/mock/projects.json` — 1 department (金卓南京项目部), 3 projects with 5-6 work items each (weights sum 100%)
- [x] 2.3 Create `src/mock/workers.json` — 10-15 registered workers with certifications
- [x] 2.4 Create `src/mock/templates.json` — 6 safety briefing templates
- [x] 2.5 Create `src/mock/daily-records.json` — 2-3 days of records across 4 modules with mixed statuses
- [x] 2.6 Implement `MockDataProvider` React Context — loads JSON, provides getters, handles in-memory mutations

## 3. APP Shell (app-shell spec)

> **Skill**: 使用 `design-html` skill 生成页面 HTML，`browse` skill 截图验证

- [x] 3.1 APP layout — max-width 430px centered, background gray, bottom tab bar fixed
- [x] 3.2 APP login page `/app/login` — phone + password + department dropdown, click login → `/app/home`
- [x] 3.3 APP home page `/app/home` — current project info, 4 module cards with ⬜/✅ status, recent review list
- [x] 3.4 Bottom tab navigation — Home / History / Notifications / Profile, active tab highlighted
- [x] 3.5 History page `/app/history` — past records grouped by date, filterable by module
- [x] 3.6 Profile page `/app/profile` — user info, pending count, offline data entry link

## 4. APP Safety Attendance (app-safety spec)

> **Skill**: 使用 `design-html` skill 生成 4 步向导页面，`frontend-patterns` skill 的 compound component 模式

- [x] 4.1 Safety wizard shell `/app/safety` — 4-step indicator (Photo → Briefing → Signature → Confirm)
- [x] 4.2 Step 1: Photo placeholder + GPS display + timestamp + attendee checkbox list
- [x] 4.3 Step 2: 6 safety briefing template radio buttons with content display
- [x] 4.4 Step 3: Canvas-based handwritten signature with Clear/Confirm buttons
- [x] 4.5 Step 4: Summary display + optional high-risk work form + Submit button
- [x] 4.6 Step navigation (Next/Back) and state preservation across steps

## 5. APP Progress Record (app-progress spec)

> **Skill**: 使用 `design-html` skill 生成页面，`frontend-patterns` skill 的 custom hook 模式封装计算逻辑

- [x] 5.1 Progress page `/app/progress` — list all work items with target/cumulative/input fields
- [x] 5.2 Real-time calculation: update cumulative + percentage on input change
- [x] 5.3 Overall progress bar — weighted sum display with visual bar
- [x] 5.4 Photo upload area — placeholder thumbnails, max 5
- [x] 5.5 Submit for review — mark module complete, update home card status

## 6. APP Material Registration (app-material spec)

> **Skill**: 使用 `design-html` skill 生成 Tab 表单页面，`frontend-patterns` skill 的 form handling 模式

- [x] 6.1 Material page `/app/material` — 4 tabs (领料/退料/设备/采买) with underline indicator
- [x] 6.2 Requisition form — dynamic add/delete rows (单号/品名/规格/数量)
- [x] 6.3 "Copy Yesterday" button — pre-fill from previous day's records
- [x] 6.4 Equipment rental form — equipment type + shift count
- [x] 6.5 Daily summary counts at bottom
- [x] 6.6 Submit for review — mark module complete

## 7. APP Construction Record (app-site-record spec)

> **Skill**: 使用 `design-html` skill 生成表单页面

- [x] 7.1 Construction record page `/app/site-record` — form with record type dropdown + dimension fields
- [x] 7.2 Hidden work checkbox — toggle description textarea
- [x] 7.3 Photo upload area — placeholder thumbnails, max 10
- [x] 7.4 Completion sketch upload — camera placeholder + PDF annotation placeholder
- [x] 7.5 Submit for review — mark module complete

## 8. APP Review (app-review spec)

> **Skill**: 使用 `design-html` skill 生成列表页面，`browse` skill 验证超时提醒样式

- [x] 8.1 Review list page `/app/review` — pending items grouped by team leader with module status checkboxes
- [x] 8.2 Timeout warning section — items pending >7 days highlighted with overdue count
- [x] 8.3 Review detail view — expand to show submitted data summary
- [x] 8.4 Action buttons — Approve / Return (with reason) / Supplement
- [x] 8.5 Module-separate review — each module independently actionable

## 9. PC Shell (pc-shell spec)

> **Skill**: 使用 `design-html` skill 生成 PC 布局，`browse` skill 截图验证宽屏效果

- [x] 9.1 PC layout — left sidebar 220px, main content area, top header with user info
- [x] 9.2 PC login page `/pc/login` — phone + password, click login → `/pc/dashboard` (merged with APP login at /login)
- [x] 9.3 Left sidebar navigation — dynamic menu based on mock user role
- [x] 9.4 Dashboard home `/pc/dashboard` — project overview card, pending alerts, project progress grid, safety summary, completion countdown timers

## 10. PC Project Management (pc-project spec)

> **Skill**: 使用 `design-html` skill 生成表格和表单页面

- [x] 10.1 Project list page `/pc/projects` — table with name/department/leader/dates/status/progress
- [x] 10.2 Project create form `/pc/projects/new` — all required fields with validation
- [x] 10.3 Work item editor — flat list, add/delete rows, weight sum validation (must = 100%)

## 11. PC Review Workstation (pc-review spec)

> **Skill**: 使用 `design-html` skill 生成审核工作台，`frontend-patterns` skill 的 modal 模式

- [x] 11.1 Review workstation `/pc/review` — filterable table (project/status/date), module checkboxes per row
- [x] 11.2 Review detail modal — expand submitted data, photos, signatures
- [x] 11.3 Action buttons in modal — Approve / Return / Supplement
- [x] 11.4 Batch approve — "Approve All" for a team leader's daily submissions

## 12. PC Reports (pc-reports spec)

> **Skill**: 使用 `design-html` skill 生成报表页面，`browse` skill 验证数据可视化效果

- [x] 12.1 Progress overview table `/pc/reports` — planned vs actual vs deviation, color-coded status
- [x] 12.2 Work item progress bars — horizontal bars per project showing individual completion rates
- [x] 12.3 Safety summary — metrics with month-over-month trend arrows
- [x] 12.4 Completion submission tracker — deadline dates, status indicators, countdown

## 13. Integration & Polish

> **Skill**: 使用 `browse` skill 端到端视觉验证，`design-review` skill 做最终设计审查

- [x] 13.1 Verify full navigation flow: APP login → home → each module → submit → review
- [x] 13.2 Verify PC flow: login → dashboard → projects → review → reports
- [x] 13.3 Ensure all form submissions update Context state and reflect on home page
- [x] 13.4 Responsive check: APP pages at 375px/430px, PC pages at 1280px+
