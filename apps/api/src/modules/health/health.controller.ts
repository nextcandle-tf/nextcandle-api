import { Controller, Get, VERSION_NEUTRAL } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { HealthService } from './health.service';
import { HealthCheckResponse, DetailedHealthResponse } from './health.dto';

@ApiTags('Health')
@Controller({ path: 'health', version: VERSION_NEUTRAL })
export class HealthController {
  constructor(private readonly healthService: HealthService) {}

  @Get()
  @ApiOperation({ summary: 'Health check' })
  @ApiResponse({ status: 200, description: 'API is healthy', type: HealthCheckResponse })
  getHealth(): HealthCheckResponse {
    return this.healthService.getHealth();
  }

  @Get('detailed')
  @ApiOperation({ summary: 'Detailed health check' })
  @ApiResponse({ status: 200, description: 'Detailed health information', type: DetailedHealthResponse })
  async getDetailedHealth(): Promise<DetailedHealthResponse> {
    return this.healthService.getDetailedHealth();
  }
}
