/* Step 1: Create the database */
CREATE DATABASE IF NOT EXISTS commissionWorkAccountingDB;

USE commissionWorkAccountingDB;

CREATE TABLE IF NOT EXISTS jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    job_date DATE NOT NULL,
    job_name VARCHAR(50) NOT NULL,
    job_type VARCHAR(50) NOT NULL,
    hours_worked DECIMAL(5,2) NOT NULL,
    pay DECIMAL(10,2)
);

/* Step 2: Create limited-privilige user(s) */
CREATE USER IF NOT EXISTS 'user1'@'localhost' IDENTIFIED BY 'user1password'; /* leave testpassword as blank if no password */
GRANT SELECT, INSERT, UPDATE, DELETE, ALTER ON commissionWorkAccountingDB.* TO 'user1'@'localhost'; /* granting only some privileges */
/* ...create other users */