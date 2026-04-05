## ADDED Requirements

### Requirement: APP Login Page
The system SHALL display a login page at `/app/login` with phone number and password fields, and a project department selector dropdown. The login page SHALL be purely presentational — clicking login navigates to `/app/home` without authentication.

#### Scenario: User opens the app
- **WHEN** user navigates to `/app/login`
- **THEN** the system displays phone number input, password input, project department dropdown, and a "Login" button

#### Scenario: User clicks login
- **WHEN** user fills in any text and clicks "Login"
- **THEN** the system navigates to `/app/home` with the selected project department context

### Requirement: APP Home Page with Daily Task Guide
The system SHALL display a home page at `/app/home` showing "Today's Tasks" guide with 4 module cards: Safety Attendance, Progress Record, Material Registration, Construction Record. Each card shows its completion status (incomplete ⬜ / complete ✅). The page also shows current project info, recent review status, and a bottom tab navigation.

#### Scenario: Team leader opens home page
- **WHEN** team leader navigates to `/app/home`
- **THEN** the system displays: current date, current project name with schedule dates and overall progress percentage, 4 module cards with status indicators, recent review status list, and bottom tab bar (Home/History/Notifications/Profile)

#### Scenario: Module card click
- **WHEN** team leader clicks an incomplete module card
- **THEN** the system navigates to the corresponding module page

#### Scenario: Completed module display
- **WHEN** a module has been filled for the current day
- **THEN** its card shows ✅ status with "Completed" text instead of ⬜ "Incomplete"

### Requirement: APP Bottom Tab Navigation
The system SHALL provide a fixed bottom tab bar with 4 tabs: Home, History, Notifications, Profile. The active tab is visually highlighted.

#### Scenario: Tab navigation
- **WHEN** user clicks any tab
- **THEN** the system navigates to the corresponding page and highlights the active tab

### Requirement: APP History Page
The system SHALL display a history page at `/app/history` showing past daily records grouped by date, filterable by module type.

#### Scenario: View history
- **WHEN** user navigates to `/app/history`
- **THEN** the system displays a list of past records grouped by date, each showing module type, submission time, and review status

### Requirement: APP Profile Page
The system SHALL display a profile page at `/app/profile` showing user info, pending/anomaly list, and offline data view entry.

#### Scenario: View profile
- **WHEN** user navigates to `/app/profile`
- **THEN** the system displays user name, role, project department, a pending tasks count, and a link to offline data view

### Requirement: APP Layout Constraint
The APP pages SHALL be constrained to max-width 430px, centered on screen, simulating a mobile phone viewport.

#### Scenario: Desktop browser viewing
- **WHEN** APP pages are viewed on a desktop browser
- **THEN** content is centered with max-width 430px, with empty space on both sides
