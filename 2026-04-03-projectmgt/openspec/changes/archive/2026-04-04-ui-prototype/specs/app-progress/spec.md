## ADDED Requirements

### Requirement: Daily Progress Entry Page
The system SHALL display a progress entry page at `/app/progress` showing all work items for the current project. Each work item displays: name, target quantity, cumulative completed, completion percentage, and a numeric input for today's completed amount.

#### Scenario: View work items
- **WHEN** team leader navigates to `/app/progress`
- **THEN** the system displays all configured work items with target, cumulative, percentage, and input fields for today's amount

#### Scenario: Enter daily quantity
- **WHEN** team leader enters a numeric value in a work item's "Today's Completed" field
- **THEN** the system updates the cumulative total and completion percentage in real-time

### Requirement: Auto-Calculated Overall Progress
The system SHALL calculate and display the overall project progress as a weighted sum: overall = Σ(work item completion rate × weight). A visual progress bar reflects the percentage.

#### Scenario: Progress bar update
- **WHEN** team leader enters values in one or more work items
- **THEN** the system recalculates the weighted overall progress and updates the progress bar

### Requirement: Construction Photo Upload Area
The system SHALL provide a photo upload area with a limit of 5 photos per day for the progress module. Photos show as placeholder thumbnails.

#### Scenario: Add photo placeholders
- **WHEN** team leader clicks the add photo button
- **THEN** the system adds a placeholder thumbnail in the photo grid (up to 5)

### Requirement: Submit Progress for Review
The system SHALL provide a "Submit for Review" button that marks the progress module as complete for today.

#### Scenario: Submit progress
- **WHEN** team leader clicks "Submit for Review"
- **THEN** the system marks the progress module as complete and navigates to home page with progress card showing ✅
