/**
 * Shared types and utilities for NextCandle
 */

// User tiers
export enum UserTier {
  GUEST = 'guest',
  MEMBER = 'member',
  PREMIUM = 'premium',
  ADMIN = 'admin',
}

// Pattern detection types
export interface CandleData {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp?: string;
}

export interface PatternMatch {
  similarity: number;
  startIndex: number;
  endIndex: number;
  timestamp?: string;
}

export interface PatternDetectionRequest {
  candles: CandleData[];
  topK?: number;
  targetLength?: number;
}

export interface PatternDetectionResponse {
  success: boolean;
  matches: PatternMatch[];
  queryLength: number;
}

// API Response types
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}
