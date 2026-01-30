-- Medical AI Gateway - Database Initialization Script
-- This runs automatically when PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE medical_ai TO medai;

-- Create indexes for common queries (tables created by SQLAlchemy)
-- These will be created after the application initializes the tables
