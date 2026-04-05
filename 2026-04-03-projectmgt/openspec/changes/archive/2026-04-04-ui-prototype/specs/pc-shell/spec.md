## ADDED Requirements

### Requirement: PC Login Page
The system SHALL display a PC login page at `/pc/login` with phone number, password, and a "Login" button. Clicking login navigates to `/pc/dashboard` without authentication.

#### Scenario: PC user opens login
- **WHEN** user navigates to `/pc/login`
- **THEN** the system displays login form with phone, password, and Login button

#### Scenario: PC login success
- **WHEN** user clicks Login
- **THEN** the system navigates to `/pc/dashboard`

### Requirement: PC Left Sidebar Navigation
The system SHALL display a left sidebar (220px width) with navigation items that change based on user role: Admin sees all items, PM sees dashboard/projects/review, Document Clerk sees document workspace, Management sees reports only.

#### Scenario: Admin views navigation
- **WHEN** admin user views the sidebar
- **THEN** the system shows: Dashboard, Projects, Personnel, Review, Documents, Reports, System menu items

#### Scenario: PM views navigation
- **WHEN** PM user views the sidebar
- **THEN** the system shows: Dashboard, Projects, Review menu items

### Requirement: PC Dashboard Home
The system SHALL display a management dashboard at `/pc/dashboard` with: project overview card, pending tasks alerts, project progress grid, safety status summary, and completion countdown tracker.

#### Scenario: View dashboard
- **WHEN** user navigates to `/pc/dashboard`
- **THEN** the system displays: project department name, active project count, total personnel count, monthly completion count, pending review count with urgency indicators, project progress cards, safety metrics, and completion countdown timers

#### Scenario: Completion countdown
- **WHEN** a project is completed but documentation deadline is approaching
- **THEN** the system displays countdown timers: 30-day deadline for completion docs, 45-day deadline for settlement docs, with color-coded urgency (red for imminent, green for completed)
