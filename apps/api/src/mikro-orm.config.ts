import 'dotenv/config';
import { defineConfig } from '@mikro-orm/postgresql';
import { Migrator } from '@mikro-orm/migrations';
import { TsMorphMetadataProvider } from '@mikro-orm/reflection';
import { EntityCaseNamingStrategy } from '@mikro-orm/core';
import { SoftDeleteHandler } from 'mikro-orm-soft-delete';
import { User } from './modules/user';

const entities = [User];

export default defineConfig({
  host: process.env.DB_HOST || 'localhost',
  port: Number(process.env.DB_PORT) || 5432,
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  dbName: process.env.DB_NAME || 'nextcandle',
  entities,
  metadataProvider: TsMorphMetadataProvider,
  extensions: [Migrator, SoftDeleteHandler],
  migrations: {
    path: './src/migrations',
    pathTs: './src/migrations',
  },
  namingStrategy: EntityCaseNamingStrategy,
  debug: process.env.NODE_ENV !== 'production',
});
