## ADDED Requirements

### Requirement: Material Registration Page with Tab Switching
The system SHALL display a material registration page at `/app/material` with 4 tabs: Requisition (领料), Return (退料), Equipment (设备), Purchase (采买). Each tab shows corresponding entry forms.

#### Scenario: Tab switching
- **WHEN** team leader clicks a tab (e.g., "Equipment")
- **THEN** the system switches to show the equipment rental entry form

### Requirement: Requisition Entry Form
The system SHALL provide a form for each requisition record with fields: requisition number (text input), material name (text input), specification (text input), quantity (numeric input) with unit. Records can be added and deleted dynamically.

#### Scenario: Add requisition record
- **WHEN** team leader clicks "Add Requisition Record"
- **THEN** the system appends a new blank requisition form row

#### Scenario: Delete requisition record
- **WHEN** team leader clicks "Delete" on a record
- **THEN** the system removes that record row from the list

### Requirement: Copy Yesterday's Records
The system SHALL provide a "Copy Yesterday's Records" button that pre-fills today's list with yesterday's material names and specifications, leaving quantities blank for editing.

#### Scenario: Copy yesterday
- **WHEN** team leader clicks "Copy Yesterday's Records"
- **THEN** the system populates the form with yesterday's material entries with blank quantity fields

### Requirement: Equipment Rental Entry
The system SHALL provide a form under the Equipment tab for recording equipment type and shift count (台班).

#### Scenario: Record equipment rental
- **WHEN** team leader enters equipment type and shift count
- **THEN** the system adds the equipment rental record to the list

### Requirement: Daily Summary
The system SHALL display a summary at the bottom: total requisitions count, returns count, equipment shifts, and purchases count for the current day.

#### Scenario: View daily summary
- **WHEN** team leader views the material page
- **THEN** the system displays counts for each category in the summary area

### Requirement: Submit Material for Review
The system SHALL provide a "Submit for Review" button that marks the material module as complete for today.

#### Scenario: Submit materials
- **WHEN** team leader clicks "Submit for Review"
- **THEN** the system marks the material module as complete and navigates to home page with material card showing ✅
