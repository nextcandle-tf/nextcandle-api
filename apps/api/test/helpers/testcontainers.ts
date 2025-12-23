/**
 * Testcontainers helpers for integration tests
 */
import { PostgreSqlContainer, StartedPostgreSqlContainer } from '@testcontainers/postgresql';
import { MikroORM } from '@mikro-orm/core';
import { PostgreSqlDriver } from '@mikro-orm/postgresql';

let postgresContainer: StartedPostgreSqlContainer | null = null;

/**
 * Start PostgreSQL container for integration tests
 */
export async function startPostgresContainer(): Promise<StartedPostgreSqlContainer> {
  if (postgresContainer) {
    return postgresContainer;
  }

  postgresContainer = await new PostgreSqlContainer('postgres:16-alpine')
    .withDatabase('nextcandle_test')
    .withUsername('test')
    .withPassword('test')
    .start();

  return postgresContainer;
}

/**
 * Stop PostgreSQL container
 */
export async function stopPostgresContainer(): Promise<void> {
  if (postgresContainer) {
    await postgresContainer.stop();
    postgresContainer = null;
  }
}

/**
 * Create MikroORM instance connected to test container
 */
export async function createTestORM(
  container: StartedPostgreSqlContainer,
): Promise<MikroORM<PostgreSqlDriver>> {
  const orm = await MikroORM.init<PostgreSqlDriver>({
    driver: PostgreSqlDriver,
    host: container.getHost(),
    port: container.getMappedPort(5432),
    user: container.getUsername(),
    password: container.getPassword(),
    dbName: container.getDatabase(),
    entities: ['./src/**/*.entity.ts'],
    debug: false,
    allowGlobalContext: true,
  });

  // Run migrations
  const migrator = orm.getMigrator();
  await migrator.up();

  return orm;
}

/**
 * Get connection string for test container
 */
export function getTestConnectionString(container: StartedPostgreSqlContainer): string {
  return container.getConnectionUri();
}
