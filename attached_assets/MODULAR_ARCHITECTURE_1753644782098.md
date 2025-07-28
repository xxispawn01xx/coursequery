# Modular Architecture Documentation

## Overview

This document describes the modular architecture implemented throughout the application. The architecture follows the guidelines in PLANNING.md, ensuring separation of concerns, maintainability, and scalability.

## Architectural Principles

### Domain-Driven Design

Our modular architecture is based on domain-driven design principles, organizing code around business domains rather than technical concerns:

- **Bounded Contexts**: Each module represents a distinct business domain with clear boundaries
- **Ubiquitous Language**: Consistent terminology used throughout the code and documentation
- **Entity-Focused**: Core business entities drive the design of modules

### Module Structure

Each module follows a consistent structure:

```
module-name/
├── storage/          # Data access and persistence
│   ├── interfaces/   # Storage interfaces
│   └── impl/         # Storage implementations
├── routes/           # API endpoints and controllers
├── services/         # Business logic
├── models/           # Domain models and types
└── utils/            # Module-specific utilities
```

### Separation of Concerns

The architecture enforces strict separation between:

1. **Storage Layer**: Responsible for data persistence and retrieval
2. **Service Layer**: Contains business logic and domain rules
3. **Route Layer**: Handles HTTP requests, validation, and responses
4. **Model Layer**: Defines domain entities and data structures

## Module Implementations

### Authentication Module

The Authentication module handles user identity, authentication, and sessions.

**Key Components:**
- `AuthStorage`: Manages user credentials and session persistence
- `AuthRoutes`: Provides login, logout, and registration endpoints
- `TokenService`: Handles JWT generation and validation

**Usage Example:**
```typescript
// Authenticate a user
router.post('/login', validateRequest(loginSchema), async (req, res) => {
  const { username, password } = req.body;
  const result = await authStorage.authenticateUser(username, password);
  
  if (result.success) {
    req.session.userId = result.user.id;
    res.json({ user: result.user });
  } else {
    res.status(401).json({ error: result.error });
  }
});
```

### Search Module

The Search module provides global and entity-specific search functionality across resources.

**Key Components:**
- `SearchStorage`: Implements search indexes and queries
- `SearchRoutes`: Exposes search endpoints with filtering and pagination
- `SearchService`: Handles search logic, scoring, and relevance

**Usage Example:**
```typescript
// Perform a global search
router.get('/search', async (req, res) => {
  const { query, filters, limit, offset } = req.query;
  
  const results = await searchStorage.search({
    query,
    filters: parseFilters(filters),
    limit: parseInt(limit) || 10,
    offset: parseInt(offset) || 0
  });
  
  res.json(results);
});
```

### Files Module

The Files module manages file uploads, storage, and retrieval.

**Key Components:**
- `FileStorage`: Handles file persistence and metadata
- `FileRoutes`: Provides file upload, download, and management endpoints
- `FileService`: Processes files, handles permissions, and manages lifecycle

**Usage Example:**
```typescript
// Upload a file
router.post('/files', 
  requireAuth, 
  upload.single('file'), 
  async (req, res) => {
    const file = req.file;
    const { description, tags } = req.body;
    
    const fileRecord = await fileStorage.storeFile({
      originalName: file.originalname,
      mimeType: file.mimetype,
      size: file.size,
      path: file.path,
      userId: req.user.id,
      description,
      tags: tags ? JSON.parse(tags) : []
    });
    
    res.status(201).json(fileRecord);
});
```

### Reports Module

The Reports module generates and manages usage, financial, and performance reports.

**Key Components:**
- `ReportStorage`: Stores report definitions and results
- `ReportRoutes`: Exposes endpoints for report generation and retrieval
- `ReportService`: Implements report calculation logic

**Usage Example:**
```typescript
// Generate a usage report
router.post('/reports/usage', 
  requireAuth, 
  requirePermission('reports:generate'), 
  async (req, res) => {
    const { startDate, endDate, metrics } = req.body;
    
    const report = await reportService.generateUsageReport({
      startDate: new Date(startDate),
      endDate: new Date(endDate),
      metrics,
      userId: req.user.id
    });
    
    res.json(report);
});
```

### Notifications Module

The Notifications module handles multi-channel notification delivery and management.

**Key Components:**
- `NotificationStorage`: Persists notification records and delivery status
- `NotificationRoutes`: Provides endpoints for sending and managing notifications
- `NotificationService`: Implements delivery logic for various channels

**Usage Example:**
```typescript
// Send a notification
router.post('/notifications', 
  requireAuth, 
  validateRequest(notificationSchema), 
  async (req, res) => {
    const { recipients, message, channel, priority } = req.body;
    
    const notification = await notificationService.sendNotification({
      senderId: req.user.id,
      recipients,
      message,
      channel,
      priority
    });
    
    res.status(201).json(notification);
});
```

### Audit Log Module

The Audit Log module provides comprehensive system activity tracking and compliance reporting.

**Key Components:**
- `AuditStorage`: Records and retrieves audit events
- `AuditRoutes`: Exposes endpoints for audit log querying and management
- `AuditService`: Implements audit logging and compliance reporting logic

**Usage Example:**
```typescript
// Record an audit event
await auditStorage.createAuditLog({
  userId: req.user.id,
  action: 'UPDATE',
  resourceType: 'patient',
  resourceId: patientId,
  details: { changedFields: ['name', 'address'] },
  ipAddress: req.ip
});

// Query audit logs
router.get('/audit/logs', 
  requireAuth, 
  requirePermission('audit:view'), 
  async (req, res) => {
    const { userId, action, resourceType, startDate, endDate } = req.query;
    
    const logs = await auditStorage.getAuditLogs({
      userId: userId ? parseInt(userId) : undefined,
      action,
      resourceType,
      startDate: startDate ? new Date(startDate) : undefined,
      endDate: endDate ? new Date(endDate) : undefined
    });
    
    res.json(logs);
});
```

### API Token Module

The API Token module manages secure token generation and validation for external API access.

**Key Components:**
- `APITokenStorage`: Manages token persistence and validation
- `APITokenRoutes`: Provides endpoints for token creation and management
- `TokenService`: Implements secure token generation and validation logic

**Usage Example:**
```typescript
// Create a new API token
router.post('/tokens', 
  requireAuth, 
  validateRequest(createTokenSchema), 
  async (req, res) => {
    const { name, scopes, expiresInDays } = req.body;
    
    let expiresAt = null;
    if (expiresInDays) {
      expiresAt = new Date();
      expiresAt.setDate(expiresAt.getDate() + expiresInDays);
    }
    
    const result = await apiTokenStorage.createToken({
      userId: req.user.id,
      name,
      scopes,
      expiresAt
    });
    
    res.status(201).json({
      token: result.token,
      plainTextToken: result.plainTextToken,
      message: 'API token created successfully. The plain text token will only be shown once.'
    });
});
```

## Transition System

The application includes a transition system to gradually migrate from legacy monolithic code to the modular architecture.

### Key Components

- `RouteVersion`: Enum defining possible route states (LEGACY, MODULAR, BOTH)
- `setRouteVersion`: Function to configure which system handles a module's requests
- `markExclusiveRoute`: Middleware to enforce route ownership
- `configureTransitionSystem`: Initializes and configures the transition system

### Usage Example

```typescript
// In transition-config.ts
setRouteVersion('auth', RouteVersion.MODULAR);
setRouteVersion('search', RouteVersion.MODULAR);

// In route definition
router.get('/api/search', markExclusiveRoute, async (req, res) => {
  // This route will only be called if search module is set to MODULAR
  // ...implementation
});
```

## Best Practices

### Writing Storage Implementations

1. **Interface First**: Always define the storage interface before implementation
2. **Error Handling**: Use consistent error handling patterns across implementations
3. **Transactions**: Use database transactions for operations that modify multiple records
4. **Validation**: Validate inputs at the storage boundary
5. **Logging**: Include appropriate logging for storage operations

### Creating Route Handlers

1. **Input Validation**: Use validation middleware (validateRequest) for all inputs
2. **Authentication**: Use requireAuth middleware for protected routes
3. **Authorization**: Use requirePermission middleware to enforce access control
4. **Error Handling**: Implement consistent error responses
5. **Response Formatting**: Follow consistent response format patterns

### Implementing Business Logic

1. **Single Responsibility**: Keep service functions focused on a single task
2. **Domain Rules**: Encapsulate business rules in the service layer
3. **Pure Functions**: Prefer pure functions where possible
4. **Testability**: Design for testability with dependency injection
5. **Idempotency**: Ensure operations are idempotent when appropriate

## Testing Strategy

Each module should include comprehensive tests:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Verify interactions between components
3. **API Tests**: Ensure API endpoints function correctly
4. **Storage Tests**: Verify data persistence and retrieval

## Migration Guide

For guidance on migrating existing code to the modular architecture, refer to [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md).

## References

- [PLANNING.md](../PLANNING.md) - Original architectural guidelines
- [MODULAR_MIGRATION_PROGRESS.md](../MODULAR_MIGRATION_PROGRESS.md) - Current migration status