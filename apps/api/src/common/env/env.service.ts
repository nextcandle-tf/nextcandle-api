import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class EnvService {
  private readonly nodeEnv: string;

  constructor(configService: ConfigService) {
    this.nodeEnv = configService.get('NODE_ENV') ?? 'local';
  }

  get currentEnv(): string {
    return this.nodeEnv;
  }

  get isLocal(): boolean {
    return this.currentEnv === 'local';
  }

  get isDevelopment(): boolean {
    return this.currentEnv === 'development';
  }

  get isStaging(): boolean {
    return this.currentEnv === 'staging';
  }

  get isProduction(): boolean {
    return this.currentEnv === 'production';
  }

  get isTest(): boolean {
    return this.currentEnv === 'test';
  }
}
