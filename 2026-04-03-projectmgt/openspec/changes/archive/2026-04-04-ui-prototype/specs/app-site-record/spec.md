## ADDED Requirements

### Requirement: Construction Record Form
The system SHALL display a construction record page at `/app/site-record` with form fields: record type (dropdown), station/location (text), pipe diameter, pipe material, length, depth, width. Form structure follows the enterprise's own construction record template format.

#### Scenario: Fill construction record
- **WHEN** team leader navigates to `/app/site-record`
- **THEN** the system displays the construction record form with all fields and placeholder text

#### Scenario: Select record type
- **WHEN** team leader selects a record type from dropdown
- **THEN** the system updates the form context accordingly

### Requirement: Hidden Work Marking
The system SHALL provide a checkbox for "Involves Hidden Work" (涉及隐蔽工程). When checked, an additional required text area appears for hidden work description.

#### Scenario: Mark hidden work
- **WHEN** team leader checks "Involves Hidden Work"
- **THEN** the system shows a required description text area

#### Scenario: Unmark hidden work
- **WHEN** team leader unchecks "Involves Hidden Work"
- **THEN** the system hides the description text area

### Requirement: Photo Upload Area
The system SHALL provide a photo upload area with a limit of 10 photos per day for the construction record module.

#### Scenario: Add photos
- **WHEN** team leader clicks add photo button
- **THEN** the system adds a placeholder thumbnail (up to 10)

### Requirement: Completion Sketch Upload
The system SHALL provide two methods for uploading completion sketches: (1) Camera capture of hand-drawn sketches (primary method), (2) PDF annotation of construction drawings (secondary method).

#### Scenario: Upload hand-drawn sketch
- **WHEN** team leader clicks "Capture Hand-drawn Sketch"
- **THEN** the system shows a placeholder for the captured sketch image

#### Scenario: PDF annotation entry
- **WHEN** team leader clicks "Annotate Construction Drawing"
- **THEN** the system shows a placeholder PDF viewer with simple annotation capability (draw circle/line)

### Requirement: Submit Construction Record for Review
The system SHALL provide a "Submit for Review" button.

#### Scenario: Submit construction record
- **WHEN** team leader clicks "Submit for Review"
- **THEN** the system marks the construction record module as complete and navigates to home page with the card showing ✅
