const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init.headers || {}),
    },
    cache: 'no-store',
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API ${path} failed: ${res.status} ${text}`)
  }
  return (await res.json()) as T
}

export type ConfigResponse = {
  demo_mode: boolean
  signal_mode: string
  instrument: string
  granularity: string
  allow_live: boolean
}

export type AccountResponse = Record<string, unknown> & { demo?: boolean }

export type SignalItem = {
  ts: string
  y_hat: number
  action: string
  confidence: number
  payload?: Record<string, unknown>
}

export type SignalGenerateResponse = {
  ts: string
  action: string
  y_hat: number
  confidence: number
  explanation: string
  meta: Record<string, unknown>
}

export async function getConfig() {
  return request<ConfigResponse>('/config')
}

export async function getAccount() {
  return request<AccountResponse>('/account')
}

export async function getRecentSignals(limit = 20) {
  return request<{ items: SignalItem[] }>(`/signals/recent?limit=${limit}`)
}

export async function generateSignal() {
  return request<SignalGenerateResponse>('/signals/generate', { method: 'POST' })
}

export async function submitOrder(direction: 'long' | 'short', units = 1) {
  return request<{ ok: boolean; response: Record<string, unknown> }>('/orders/market', {
    method: 'POST',
    body: JSON.stringify({ direction, units }),
  })
}

export async function getNews(limit = 5) {
  return request<{ items: any[] }>(`/news?limit=${limit}`)
}
