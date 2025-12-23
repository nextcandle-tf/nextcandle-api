import { ApiProperty } from '@nestjs/swagger';

export class HealthCheckResponse {
  @ApiProperty({ example: 'ok' })
  status: string;

  @ApiProperty({ example: '2024-12-24T12:00:00.000Z' })
  timestamp: string;
}

export class ServiceStatus {
  @ApiProperty({ example: 'healthy', enum: ['healthy', 'unhealthy'] })
  status: 'healthy' | 'unhealthy';

  @ApiProperty({ example: 10, required: false })
  latency?: number;

  @ApiProperty({ required: false })
  error?: string;
}

export class ServicesHealth {
  @ApiProperty({ type: ServiceStatus })
  database: ServiceStatus;
}

export class DetailedHealthResponse {
  @ApiProperty({ example: 'ok', enum: ['ok', 'degraded'] })
  status: 'ok' | 'degraded';

  @ApiProperty({ example: '2024-12-24T12:00:00.000Z' })
  timestamp: string;

  @ApiProperty({ example: '1.0.0' })
  version: string;

  @ApiProperty({ example: 12345.67, description: 'Uptime in seconds' })
  uptime: number;

  @ApiProperty({ type: ServicesHealth })
  services: ServicesHealth;
}
