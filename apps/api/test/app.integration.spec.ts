/**
 * Integration tests for App module with Testcontainers
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Test, TestingModule } from '@nestjs/testing';
import { FastifyAdapter, NestFastifyApplication } from '@nestjs/platform-fastify';
import { StartedPostgreSqlContainer } from '@testcontainers/postgresql';
import { AppModule } from '../src/app.module';
import { startPostgresContainer, stopPostgresContainer } from './helpers/testcontainers';

describe('AppController (Integration)', () => {
  let app: NestFastifyApplication;
  let container: StartedPostgreSqlContainer;

  beforeAll(async () => {
    // Start PostgreSQL container
    container = await startPostgresContainer();

    // Set environment variables for the test
    process.env.DB_HOST = container.getHost();
    process.env.DB_PORT = container.getMappedPort(5432).toString();
    process.env.DB_USER = container.getUsername();
    process.env.DB_PASSWORD = container.getPassword();
    process.env.DB_NAME = container.getDatabase();

    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication<NestFastifyApplication>(
      new FastifyAdapter(),
    );

    await app.init();
    await app.getHttpAdapter().getInstance().ready();
  });

  afterAll(async () => {
    await app?.close();
    await stopPostgresContainer();
  });

  it('GET / - should return health status', async () => {
    const result = await app.inject({
      method: 'GET',
      url: '/',
    });

    expect(result.statusCode).toBe(200);
    const body = JSON.parse(result.payload);
    expect(body.status).toBe('ok');
    expect(body.timestamp).toBeDefined();
  });

  it('GET /health - should return detailed health', async () => {
    const result = await app.inject({
      method: 'GET',
      url: '/health',
    });

    expect(result.statusCode).toBe(200);
    const body = JSON.parse(result.payload);
    expect(body.status).toBe('ok');
    expect(body.version).toBe('1.0.0');
  });
});
