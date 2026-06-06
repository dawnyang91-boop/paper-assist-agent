import type { AskResult, BackgroundTask, ChatEvent, SessionSummary, UploadedRagFile, UserProfile } from './types';

const USER_ID_KEY = 'user_id';
const USER_PROFILE_KEY = 'user_profile';
const API_BASE = normalizeBasePath(import.meta.env.VITE_API_BASE || '/chatbot');

function normalizeBasePath(value: string): string {
  const trimmed = value.trim();
  if (!trimmed || trimmed === '/') return '';
