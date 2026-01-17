/* eslint-disable @next/next/no-img-element */
'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

type AccountResponse = {
  account?: {
    id?: string
    balance?: number | string
    alias?: string
    currency?: string
    marginRate?: number | string
  }
  demo?: boolean
  [key: string]: any
}

type SignalItem = {
  ts: string
  action: string
  y_hat: number
  confidence: number
  payload?: Record<string, any>
}

type SignalGenerateResponse = {
  ts: string
  action: string
  y_hat: number
  confidence: number
  explanation?: string
  meta?: Record<string, any>
}

type Status = 'idle' | 'loading' | 'error' | 'success'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'
const AUTO_KEY = 'autoSignalMode'

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
    throw new Error(text || `Request failed: ${res.status}`)
  }
  return (await res.json()) as T
}

export default function Home() {
  const pathname = usePathname()
  const [account, setAccount] = useState<AccountResponse | null>(null)
  const [lastSignal, setLastSignal] = useState<SignalItem | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [orderStatus, setOrderStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  const [autoMode, setAutoMode] = useState(false)

  const accountFields = useMemo(() => {
    if (!account?.account) return []
    const acc = account.account
    return [
      { label: 'Alias', value: acc.alias },
      { label: 'Balance', value: acc.balance },
      { label: 'Currency', value: acc.currency },
      { label: 'Margin rate', value: acc.marginRate },
    ].filter((x) => x.value !== undefined && x.value !== null)
  }, [account])

  const loadAccount = async () => {
    try {
      const data = await request<AccountResponse>('/account')
      setAccount(data)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const loadLastSignal = async () => {
    try {
      const data = await request<{ items: SignalItem[] }>('/signals/recent?limit=1')
      setLastSignal(data.items?.[0] ?? null)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const handleGenerate = async () => {
    setStatus('loading')
    setError(null)
    try {
      const sig = await request<SignalGenerateResponse>('/signals/generate', { method: 'POST' })
      const normalized: SignalItem = {
        ts: sig.ts,
        action: sig.action,
        y_hat: sig.y_hat,
        confidence: sig.confidence,
        payload: sig.meta,
      }
      setLastSignal(normalized)
      setStatus('success')
    } catch (err) {
      setStatus('error')
      setError((err as Error).message)
    }
  }

  const handleOrder = async (direction: 'long' | 'short') => {
    setOrderStatus('loading')
    setError(null)
    try {
      await request('/orders/market', {
        method: 'POST',
        body: JSON.stringify({ direction }),
      })
      setOrderStatus('success')
    } catch (err) {
      setOrderStatus('error')
      setError((err as Error).message)
    }
  }

  useEffect(() => {
    void loadAccount()
    void loadLastSignal()
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const saved = localStorage.getItem(AUTO_KEY)
    setAutoMode(saved === '1')
    const onStorage = () => {
      const next = localStorage.getItem(AUTO_KEY)
      setAutoMode(next === '1')
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  useEffect(() => {
    if (!autoMode) return
    const id = setInterval(() => {
      if (status !== 'loading') {
        void handleGenerate()
      }
    }, 60_000)
    return () => clearInterval(id)
  }, [autoMode, status])

  const signalAction = lastSignal?.action?.toLowerCase?.() || ''
  const allowLong = signalAction === 'long'
  const allowShort = signalAction === 'short'

  return (
    <main className="relative isolate min-h-screen overflow-hidden bg-gradient-to-br from-[#f9fafc] via-[#f4f6fb] to-[#eef2f7] text-neutral-900">
      <div className="absolute inset-0 -z-20 bg-[radial-gradient(circle_at_1px_1px,#e4e7ee_1.2px,transparent_0)] [background-size:22px_22px]" />
      <div className="relative mx-auto flex max-w-6xl flex-col gap-8 px-6 pb-16 pt-10">
        <nav className="flex items-center gap-3">
          {[
            { href: '/', label: 'Dashboard' },
            { href: '/news', label: 'Wiadomości' },
            { href: '/signals', label: 'Dziennik' },
            { href: '/stats', label: 'Statystyka' },
            { href: '/settings', label: 'Ustawienia' },
          ].map((item) => {
            const active = pathname === item.href
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-full border-2 px-4 py-2 text-sm font-semibold transition-all duration-150 ${
                  active
                    ? 'border-neutral-900 bg-neutral-900 text-white shadow-[0_4px_0_#0f172a]'
                    : 'border-neutral-200 bg-white text-neutral-700 hover:border-neutral-300'
                }`}
              >
                {item.label}
              </Link>
            )
          })}
        </nav>

        <section className="relative overflow-hidden rounded-[36px] border-[3px] border-neutral-900 bg-white px-8 py-10 shadow-[0_30px_95px_rgba(0,0,0,0.16)]">
          <div className="absolute -left-10 top-6 h-14 w-3 rounded-full bg-neutral-900/70 blur-[12px]" aria-hidden />
          <p className="text-lg font-semibold text-neutral-700">Trader CRM</p>
          <h1 className="mt-2 text-4xl font-black leading-tight tracking-tight text-neutral-900 sm:text-5xl">
            Open-Source CRM dla tradingu
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-neutral-600">
            LIVE tryb (practice) z OANDA. Zarządzaj sygnałami i wysyłaj zlecenia bezpośrednio z CRM.
          </p>
          <div className="mt-6 flex flex-wrap gap-3 text-sm text-neutral-700">
            <span className="rounded-full bg-neutral-100 px-3 py-1 font-semibold">EUR_USD/M15</span>
            <span className="rounded-full bg-neutral-100 px-3 py-1 font-semibold">
              {account?.demo ? 'DEMO' : 'LIVE'}
            </span>
            <span className="rounded-full bg-neutral-100 px-3 py-1 font-semibold">API-driven</span>
            <span className="rounded-full bg-neutral-100 px-3 py-1 font-semibold">
              {API_BASE.replace(/^https?:\/\//, '')}
            </span>
          </div>
        </section>

        <section className="flex flex-col gap-6 rounded-[32px] border border-neutral-200 bg-white/80 p-6 shadow-[0_18px_60px_rgba(0,0,0,0.08)] backdrop-blur">
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Last signal</p>
                  <p className="text-xl font-bold text-neutral-900">{lastSignal?.action ?? '—'}</p>
                  <p className="text-sm text-neutral-600">{lastSignal?.ts ?? 'Brak sygnałów'}</p>
                </div>
                <span className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-semibold text-neutral-700">
                  {status === 'loading' ? 'Loading…' : 'Ready'}
                </span>
              </div>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={handleGenerate}
                  className="inline-flex items-center justify-center rounded-full border-2 border-neutral-900 bg-neutral-900 px-5 py-2 text-sm font-semibold text-white shadow-[0_6px_0_#0f172a] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_9px_0_#0f172a] active:translate-y-0 active:shadow-[0_3px_0_#0f172a] disabled:opacity-60"
                  disabled={status === 'loading'}
                >
                  Pobierz sygnał
                </button>
                <button
                  onClick={() => handleOrder('long')}
                  className="inline-flex items-center justify-center rounded-full border-2 border-emerald-700 bg-emerald-600 px-5 py-2 text-sm font-semibold text-white shadow-[0_6px_0_#065f46] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_9px_0_#065f46] active:translate-y-0 active:shadow-[0_3px_0_#065f46] disabled:opacity-60"
                  disabled={orderStatus === 'loading' || !allowLong}
                >
                  LONG
                </button>
                <button
                  onClick={() => handleOrder('short')}
                  className="inline-flex items-center justify-center rounded-full border-2 border-rose-700 bg-rose-600 px-5 py-2 text-sm font-semibold text-white shadow-[0_6px_0_#b91c1c] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_9px_0_#b91c1c] active:translate-y-0 active:shadow-[0_3px_0_#b91c1c] disabled:opacity-60"
                  disabled={orderStatus === 'loading' || !allowShort}
                >
                  SHORT
                </button>
              </div>
              {lastSignal && (
                <dl className="grid grid-cols-2 gap-3 text-sm text-neutral-700">
                  <div>
                    <dt className="text-xs uppercase tracking-wide text-neutral-500">y_hat</dt>
                    <dd className="font-semibold">{lastSignal.y_hat?.toFixed?.(6)}</dd>
                  </div>
                  <div>
                    <dt className="text-xs uppercase tracking-wide text-neutral-500">confidence</dt>
                    <dd className="font-semibold">{lastSignal.confidence?.toFixed?.(3)}</dd>
                  </div>
                </dl>
              )}
            </div>

            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Konto</h3>
                <span className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-semibold text-neutral-700">
                  {account?.demo ? 'DEMO' : 'LIVE'}
                </span>
              </div>
              {accountFields.length ? (
                <div className="grid grid-cols-2 gap-3 text-sm text-neutral-800">
                  {accountFields.map((item) => (
                    <div key={item.label} className="rounded-xl bg-neutral-50 px-3 py-2">
                      <p className="text-xs uppercase tracking-wide text-neutral-500">{item.label}</p>
                      <p className="font-semibold text-neutral-900">{item.value as string}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-neutral-600">Brak danych konta.</p>
              )}
              <button
                onClick={() => {
                  void loadAccount()
                  void loadLastSignal()
                }}
                className="self-start rounded-xl border border-neutral-200 bg-neutral-50 px-3 py-2 text-sm font-semibold text-neutral-700 hover:border-neutral-300"
              >
                Odśwież dane
              </button>
            </div>
          </div>
        </section>

        {error && (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-rose-700 shadow-sm">
            {error}
          </div>
        )}
      </div>
    </main>
  )
}
