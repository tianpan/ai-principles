## ADDED Requirements

### Requirement: Progress Overview Report
The system SHALL display a progress overview report at `/pc/reports` showing all projects with planned vs actual progress, deviation percentage, and status indicators (green=normal, yellow=slight delay, red=severe delay).

#### Scenario: View progress report
- **WHEN** management user navigates to `/pc/reports`
- **THEN** the system displays a table with project name, planned progress, actual progress, deviation, and color-coded status

### Requirement: Work Item Progress Comparison
The system SHALL display horizontal bar charts for each project showing individual work item completion rates with target vs actual quantities.

#### Scenario: View work item details
- **WHEN** management selects a project in the report
- **THEN** the system shows horizontal bars for each work item with completion percentage, cumulative quantity, and target quantity

### Requirement: Safety Summary Report
The system SHALL display safety metrics: monthly briefing completion rate, safety briefing count, high-risk work count, with month-over-month trend indicators.

#### Scenario: View safety report
- **WHEN** management views the safety section
- **THEN** the system shows current month metrics with comparison to previous month and up/down arrows

### Requirement: Completion Document Submission Progress
The system SHALL display a timeline tracker for each completed project showing: completion date, 30-day deadline for completion docs, 45-day deadline for settlement docs, current status (pending/submitted/overdue).

#### Scenario: View submission progress
- **WHEN** management views the completion section
- **THEN** the system shows each completed project with deadline dates and status (green=submitted, red=overdue, orange=approaching)
