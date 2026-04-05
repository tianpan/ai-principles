## ADDED Requirements

### Requirement: PC Review Workstation
The system SHALL display a review workstation at `/pc/review` with filterable table of all pending reviews across projects, with filters: project, status, date range. Each row shows team leader, project, date, and module completion status (checkboxes for attendance/progress/material/construction record).

#### Scenario: Filter reviews
- **WHEN** user selects a project filter and date range
- **THEN** the system updates the table to show only matching records

#### Scenario: View review details
- **WHEN** user clicks "View Details" on a review row
- **THEN** the system opens a detail modal showing submitted data, photos, signatures

### Requirement: Review Actions in Modal
The system SHALL provide Approve, Return, and Supplement buttons in the review detail modal.

#### Scenario: Approve from modal
- **WHEN** PM clicks "Approve" in the detail modal
- **THEN** the system marks the item as approved and updates the list

#### Scenario: Batch approve
- **WHEN** PM clicks "Approve All" for a team leader's submissions
- **THEN** the system approves all pending modules for that team leader on that date
