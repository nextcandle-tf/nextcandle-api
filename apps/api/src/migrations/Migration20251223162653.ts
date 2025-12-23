import { Migration } from '@mikro-orm/migrations';

export class Migration20251223162653 extends Migration {

  override async up(): Promise<void> {
    this.addSql(`create table "users" ("id" serial primary key, "createdAt" timestamptz not null, "updatedAt" timestamptz not null, "role" text check ("role" in ('user', 'admin')) not null default 'user', "tier" text check ("tier" in ('guest', 'member', 'premium', 'max', 'beta')) not null default 'member', "deletedAt" timestamptz null);`);
  }

  override async down(): Promise<void> {
    this.addSql(`drop table if exists "users" cascade;`);
  }

}
