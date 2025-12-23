import { Entity, Enum, Property } from '@mikro-orm/core';
import { SoftDeletable } from 'mikro-orm-soft-delete';
import { BaseEntity } from '../../../entities/base.entity';

export enum UserTier {
  GUEST = 'guest',
  MEMBER = 'member',
  PREMIUM = 'premium',
  MAX = 'max',
  BETA = 'beta',
}

export enum UserRole {
  USER = 'user',
  ADMIN = 'admin',
}

@SoftDeletable(() => User, 'deletedAt', () => new Date())
@Entity({ tableName: 'users' })
export class User extends BaseEntity {
  @Enum(() => UserRole)
  role: UserRole = UserRole.USER;

  @Enum(() => UserTier)
  tier: UserTier = UserTier.MEMBER;

  @Property({ nullable: true })
  deletedAt?: Date;
}
