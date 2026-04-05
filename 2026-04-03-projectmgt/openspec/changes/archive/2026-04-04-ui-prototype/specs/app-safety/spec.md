## ADDED Requirements

### Requirement: Safety Attendance 4-Step Wizard
The system SHALL implement a 4-step wizard at `/app/safety` with a step indicator showing progress: Photo+GPS → Safety Briefing → Signature → Confirmation.

#### Scenario: Enter safety module
- **WHEN** team leader clicks the Safety Attendance card from home page
- **THEN** the system navigates to `/app/safety` showing step 1 of the wizard with a step indicator

### Requirement: Step 1 - Morning Meeting Photo and Attendee Selection
The system SHALL display a photo capture area with auto-recorded GPS and timestamp, plus a checklist of registered workers for attendance selection.

#### Scenario: Capture morning meeting photo
- **WHEN** team leader clicks the camera button
- **THEN** the system shows a placeholder photo area and displays auto-filled GPS coordinates and current time

#### Scenario: Select attendees
- **WHEN** team leader views the attendee list
- **THEN** the system shows all registered workers for the project department with checkboxes; team leader is pre-checked

#### Scenario: Duplicate attendance warning
- **WHEN** team leader selects a worker who was already checked in at another project today
- **THEN** the system displays a warning: "This person has already been checked in at [Project X] today"

### Requirement: Step 2 - Safety Briefing Template Selection
The system SHALL display 6 safety briefing template categories: High Altitude, Hot Work, Confined Space, Foundation Excavation, Pipeline Welding, General. Selecting one displays the briefing content.

#### Scenario: Select briefing type
- **WHEN** team leader selects a briefing type (e.g., "Hot Work Safety Briefing")
- **THEN** the system highlights the selection and displays the corresponding briefing content text

### Requirement: Step 3 - Handwritten Signature
The system SHALL provide a canvas area for the team leader's handwritten signature with Clear and Confirm buttons. Only the team leader signs; workers do not sign individually.

#### Scenario: Draw signature
- **WHEN** team leader draws on the signature canvas
- **THEN** the system captures the handwriting and displays it on the canvas

#### Scenario: Clear signature
- **WHEN** team leader clicks "Clear"
- **THEN** the system resets the signature canvas to blank

#### Scenario: Confirm signature
- **WHEN** team leader clicks "Confirm"
- **THEN** the system locks the signature and enables proceeding to step 4

### Requirement: Step 4 - Summary and Submission
The system SHALL display a summary of all collected data: photo uploaded, GPS, time, attendee count, briefing type, signature status. It includes an optional "Record High-Risk Work" entry and a "Submit for Review" button.

#### Scenario: View summary
- **WHEN** team leader reaches step 4
- **THEN** the system displays: meeting photo status, GPS coordinates, time, attendee count, briefing type, signature confirmation

#### Scenario: Submit for review
- **WHEN** team leader clicks "Submit for Review"
- **THEN** the system marks the safety attendance module as complete for today and navigates back to home page with the module card showing ✅

#### Scenario: Record high-risk work
- **WHEN** team leader clicks "Record High-Risk Work"
- **THEN** the system shows a form with start time, end time, and description fields
