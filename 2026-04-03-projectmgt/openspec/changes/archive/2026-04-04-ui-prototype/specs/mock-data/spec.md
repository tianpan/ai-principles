## ADDED Requirements

### Requirement: Mock Data Structure
The system SHALL provide JSON-based mock data files that align with the PRD Chapter 6 data model, covering: 1 project department, 1 PM, 3 team leaders, 3 projects, registered workers, safety briefing templates, and 2-3 days of historical daily records.

#### Scenario: Load mock data
- **WHEN** the application starts
- **THEN** the system loads all mock JSON files and makes data available through React Context

### Requirement: Project Department and Users
The mock data SHALL include: 金卓南京项目部 (1 department), 王经理 (PM), 张三/李四/王五 (3 team leaders), each assigned to a separate project.

#### Scenario: User role data
- **WHEN** the application loads user data
- **THEN** each user has id, name, phone, password, role, departmentId, status fields matching the PRD User entity definition

### Requirement: Project and Work Item Data
The mock data SHALL include 3 projects: XX路中压燃气管道工程, YY路支管连接工程, ZZ小区改造工程. Each project has 5-6 work items with target quantities, units, and weights summing to 100%.

#### Scenario: Work item data
- **WHEN** the application loads project data
- **THEN** each project's work items have name, targetQuantity, unit, and weight fields matching the WorkItem entity

### Requirement: Safety Briefing Templates
The mock data SHALL include 6 safety briefing templates: 高空作业, 动火作业, 有限空间, 基坑开挖, 管线焊接, 通用安全交底. Each has a name, category, and sample content text.

#### Scenario: Template data
- **WHEN** the application loads template data
- **THEN** all 6 templates are available with id, name, category, and content fields

### Requirement: Historical Daily Records
The mock data SHALL include 2-3 days of historical daily records across all 4 modules (attendance, progress, material, construction record) with mixed statuses: approved, pending, returned.

#### Scenario: Historical record data
- **WHEN** the application loads historical records
- **THEN** records exist for past dates with varied statuses enabling the review workflow demonstration

### Requirement: In-Memory State Management
The system SHALL use React Context to serve mock data. Form submissions update Context state in memory. Page refresh resets all data to original mock state.

#### Scenario: Submit form updates state
- **WHEN** user submits a form (e.g., progress entry)
- **THEN** the Context state updates to reflect the new data without page refresh

#### Scenario: Refresh resets state
- **WHEN** user refreshes the browser
- **THEN** all data returns to the original mock JSON state
