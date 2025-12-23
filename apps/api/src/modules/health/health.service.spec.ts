import { describe, it, expect, beforeEach, vi } from 'vitest';
import { HealthService } from './health.service';
import { MikroORM } from '@mikro-orm/core';

describe('HealthService', () => {
  let service: HealthService;
  let mockOrm: Partial<MikroORM>;

  beforeEach(() => {
    mockOrm = {
      em: {
        getConnection: vi.fn().mockReturnValue({
          execute: vi.fn().mockResolvedValue([{ '?column?': 1 }]),
        }),
      } as any,
    };
    service = new HealthService(mockOrm as MikroORM);
  });

  describe('getHealth', () => {
    it('should return status ok', () => {
      const result = service.getHealth();

      expect(result.status).toBe('ok');
      expect(result.timestamp).toBeDefined();
    });

    it('should return valid ISO timestamp', () => {
      const result = service.getHealth();
      const timestamp = new Date(result.timestamp);

      expect(timestamp.toString()).not.toBe('Invalid Date');
    });
  });

  describe('getDetailedHealth', () => {
    it('should return ok status when database is healthy', async () => {
      const result = await service.getDetailedHealth();

      expect(result.status).toBe('ok');
      expect(result.timestamp).toBeDefined();
      expect(result.version).toBeDefined();
      expect(result.uptime).toBeGreaterThanOrEqual(0);
      expect(result.services.database.status).toBe('healthy');
    });

    it('should return degraded status when database is unhealthy', async () => {
      mockOrm.em = {
        getConnection: vi.fn().mockReturnValue({
          execute: vi.fn().mockRejectedValue(new Error('Connection failed')),
        }),
      } as any;

      const result = await service.getDetailedHealth();

      expect(result.status).toBe('degraded');
      expect(result.services.database.status).toBe('unhealthy');
      expect(result.services.database.error).toBe('Connection failed');
    });
  });
});
