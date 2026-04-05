## ADDED Requirements

### Requirement: PM Review List Page
The system SHALL display a review list page at `/app/review` showing pending and completed review items. Items are grouped by team leader, showing module type, date, and summary. Items pending over 7 days are highlighted with a timeout warning.

#### Scenario: View pending reviews
- **WHEN** PM navigates to `/app/review`
- **THEN** the system displays pending reviews grouped by team leader, each showing module type, date, and key data summary

#### Scenario: Timeout warning
- **WHEN** a review item has been pending for more than 7 days
- **THEN** the system displays it in a highlighted "Timeout Alert" section with the number of days overdue

### Requirement: Review Detail and Actions
The system SHALL provide review actions for each pending item: Approve, Return for Revision, and Supplement. Each action updates the item's status.

#### Scenario: Approve review
- **WHEN** PM clicks "Approve" on a pending item
- **THEN** the system changes the item status to "Approved" and moves it to the completed section

#### Scenario: Return for revision
- **WHEN** PM clicks "Return" on a pending item
- **THEN** the system prompts for a reason and changes the item status to "Returned"

#### Scenario: Supplement data
- **WHEN** PM clicks "Supplement" on a pending item
- **THEN** the system opens an edit form allowing PM to add supplementary information

### Requirement: Module-Separate Review
The system SHALL allow PM to review each module (attendance, progress, material, construction record) independently, without requiring all modules to be reviewed at once.

#### Scenario: Review single module
- **WHEN** PM approves one module for a team leader
- **THEN** only that module is marked as approved; other modules remain pending
