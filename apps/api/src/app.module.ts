import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { MikroOrmModule } from '@mikro-orm/nestjs';
import { EnvModule } from './common/env';
import { HealthModule } from './modules/health';
import { UserModule } from './modules/user';
import mikroOrmConfig from './mikro-orm.config';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: ['.env.local', '.env'],
    }),
    EnvModule,
    MikroOrmModule.forRootAsync({
      inject: [ConfigService],
      useFactory: (config: ConfigService) => ({
        ...mikroOrmConfig,
        host: config.get('DB_HOST', 'localhost'),
        port: config.get('DB_PORT', 5432),
        user: config.get('DB_USER', 'postgres'),
        password: config.get('DB_PASSWORD', 'postgres'),
        dbName: config.get('DB_NAME', 'nextcandle'),
      }),
    }),
    HealthModule,
    UserModule,
  ],
})
export class AppModule {}
