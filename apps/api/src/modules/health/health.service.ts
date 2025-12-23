import { Injectable } from '@nestjs/common';
import { MikroORM } from '@mikro-orm/core';
import { HealthCheckResponse, DetailedHealthResponse, ServiceStatus } from './health.dto';

@Injectable()
export class HealthService {
  constructor(private readonly orm: MikroORM) {}

  getHealth(): HealthCheckResponse {
    return {
      status: 'ok',
      timestamp: new Date().toISOString(),
    };
  }

  async getDetailedHealth(): Promise<DetailedHealthResponse> {
    const dbStatus = await this.checkDatabase();

    return {
      status: dbStatus.status === 'healthy' ? 'ok' : 'degraded',
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0',
      uptime: process.uptime(),
      services: {
        database: dbStatus,
      },
    };
  }

  private async checkDatabase(): Promise<ServiceStatus> {
    try {
      const connection = this.orm.em.getConnection();
      await connection.execute('SELECT 1');
      return {
        status: 'healthy',
        latency: 0,
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
}
