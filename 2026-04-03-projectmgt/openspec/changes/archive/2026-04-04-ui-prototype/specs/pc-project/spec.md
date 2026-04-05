## ADDED Requirements

### Requirement: Project List Page
The system SHALL display a project list at `/pc/projects` showing all projects in a table with columns: name, department, team leader, dates, status, progress percentage.

#### Scenario: View project list
- **WHEN** user navigates to `/pc/projects`
- **THEN** the system displays a table of all projects with key information columns

### Requirement: Project Create/Edit Form
The system SHALL provide a form at `/pc/projects/new` for creating projects with fields: name (required), department, start date (required), end date (required), team leader (required), special notes (required), subcontractor flag.

#### Scenario: Create project
- **WHEN** user fills in the project form and clicks Save
- **THEN** the system adds the project to the list and navigates back

### Requirement: Work Item Configuration
The system SHALL provide a flat work item list editor within each project, where each work item has: name, target quantity, unit, weight (percentage). Total weight MUST equal 100%.

#### Scenario: Add work item
- **WHEN** user clicks "Add Work Item" in project configuration
- **THEN** the system appends a new work item row with editable fields

#### Scenario: Weight validation
- **WHEN** user enters weights that don't sum to 100%
- **THEN** the system displays a warning showing the current total and difference from 100%
